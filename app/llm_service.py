# whatsapp_client/app/llm_service.py

import logging
import time
from typing import Callable, Awaitable, Any, Optional
from .db_manager import ConversationDBManager
from .llm_client import GeminiLLMClient
from .models import IncomingMessageData, SendTextMessageRequest
from .config import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

WhatsAppTextSender = Callable[[str, SendTextMessageRequest], Awaitable[Any]]

class WhatsAppLLMService:
    """
    Orchestrates the interaction between incoming WhatsApp messages,
    the database manager, the LLM client, and the WhatsApp message sender.
    """
    def __init__(
        self,
        db_manager: ConversationDBManager,
        llm_client: GeminiLLMClient, # LLM Client now manages its own chat state
        whatsapp_text_sender: WhatsAppTextSender
    ):
        """
        Initializes the service with necessary dependencies.

        Args:
            db_manager: An instance of ConversationDBManager.
            llm_client: An instance of GeminiLLMClient.
            whatsapp_text_sender: An async function to send text messages via WhatsApp API.
        """
        self.db_manager = db_manager
        self.llm_client = llm_client
        self.send_text = whatsapp_text_sender
        logger.info("WhatsAppLLMService initialized.")

    async def handle_incoming_message(self, message_data: IncomingMessageData):
        """
        Handles a single incoming WhatsApp message received via webhook.

        1. Ignores messages sent by the bot itself.
        2. Saves the incoming user message to the database.
        3. Prepares input and calls the LLM client (which manages its own chat session).
        4. Saves the LLM's response to the database.
        5. Sends the LLM's response back to the user via WhatsApp.

        Args:
            message_data: The parsed IncomingMessageData object from the webhook,
                          potentially including 'local_file_path' if media was downloaded.
        """
        try:
            # --- Step 1: Ignore Own Messages ---
            if message_data.is_mine:
                logger.debug(f"Ignoring own message (ID: {message_data.id})")
                return

            sender_phone = message_data.sender_phone
            logger.info(f"Processing incoming message ID {message_data.id} from {sender_phone}")

            # --- Step 2: Save Incoming User Message ---
            try:
                await self.db_manager.save_message(
                    sender_phone=sender_phone,
                    role='user',
                    message_id=message_data.id,
                    type=message_data.type,
                    text_content=message_data.body,
                    local_file_path=message_data.local_file_path,
                    timestamp=message_data.timestamp
                )
            except Exception as db_error:
                logger.error(f"Failed to save incoming message for {sender_phone}: {db_error}", exc_info=True)
                # Continue processing even if save fails

            # --- Step 3: Prepare LLM Input ---
            mime_type = message_data.media.type if message_data.media else None
            llm_input_data = {
                'type': message_data.type,
                'text_content': message_data.body,
                'local_file_path': message_data.local_file_path,
                'mime_type': mime_type
            }

            # --- Step 4: Call LLM ---
            logger.info(f"Sending message content (type: {message_data.type}) to LLM for {sender_phone}...")
            # Pass only the current input data, client handles history via chat session
            ai_response_text = await self.llm_client.process_user_message(
                sender_phone=sender_phone,
                input_data=llm_input_data
            )

            # --- Step 5: Validate LLM Response ---
            if not ai_response_text or ai_response_text == "I cannot respond to that prompt due to safety guidelines.":
                logger.warning(f"LLM returned no valid response for {sender_phone}. Not sending reply.")
                return # Stop processing this message

            logger.info(f"LLM generated response for {sender_phone}: '{ai_response_text[:100]}...'")

            # --- Step 6: Save AI Response ---
            try:
                await self.db_manager.save_message(
                    sender_phone=sender_phone,
                    role='assistant',
                    message_id=None,
                    type='text',
                    text_content=ai_response_text,
                    local_file_path=None,
                    timestamp=int(time.time())
                )
            except Exception as db_error:
                logger.error(f"Failed to save AI response for {sender_phone}: {db_error}", exc_info=True)
                # Continue anyway

            # --- Step 7: Send Reply via WhatsApp ---
            try:
                logger.info(f"Sending AI response back to {sender_phone}...")
                await self.send_text(
                    sender_phone,
                    SendTextMessageRequest(message=ai_response_text)
                )
                logger.info(f"Successfully sent AI response to {sender_phone}.")
            except Exception as send_error:
                logger.error(f"Failed to send AI response to {sender_phone}: {send_error}", exc_info=True)

        except Exception as e:
            # --- Step 8: Catch All Other Processing Errors ---
            logger.error(f"Unhandled error processing message ID {message_data.id if message_data else 'N/A'} from {sender_phone if message_data else 'N/A'}: {e}", exc_info=True)
            # Optionally send a generic error message
            try:
                if message_data and not message_data.is_mine and message_data.sender_phone:
                    await self.send_text(
                        message_data.sender_phone,
                        SendTextMessageRequest(message="Sorry, I encountered an internal error.")
                    )
                    logger.info("Sent generic error message to user.")
            except Exception as final_error:
                logger.error(f"Failed even to send generic error message: {final_error}", exc_info=True)
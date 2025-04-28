# whatsapp_client/app/llm_client.py

import google.generativeai as genai
import os
import logging
import mimetypes
import base64
from typing import Optional, List, Dict, Any
from fastapi import HTTPException, status
import asyncio
from .db_manager import ConversationDBManager # <-- Added Import

# Import settings (assuming this contains API key etc.)
from .config import settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration Constants ---
# Size threshold for sending files inline vs. uploading (in bytes)
# A common practical limit for inline data is around a few MB.
# Adjust this based on API documentation or empirical testing.
INLINE_FILE_SIZE_THRESHOLD_MB = 4
INLINE_FILE_SIZE_THRESHOLD_BYTES = INLINE_FILE_SIZE_THRESHOLD_MB * 1024 * 1024

# Supported MIME types (remains the same)
SUPPORTED_GENAI_MIMETYPES = {
    # ... (keep the list from the previous version) ...
    # Images
    'image/png', 'image/jpeg', 'image/webp', 'image/heic', 'image/heif',
    # Audio
    'audio/wav', 'audio/mp3', 'audio/aiff', 'audio/aac', 'audio/ogg', 'audio/flac',
    # Video
    'video/mp4', 'video/mpeg', 'video/mov', 'video/avi', 'video/x-flv', 'video/mpg',
    'video/webm', 'video/wmv', 'video/3gpp',
    # Text/Documents
    'text/plain', 'text/html', 'text/css', 'text/javascript', 'application/x-javascript',
    'text/x-python', 'application/x-python', 'text/x-java-source', 'text/x-java',
    'text/x-c', 'text/x-csrc', 'text/x-cpp', 'text/x-c++src', 'application/json',
    'text/markdown', 'text/csv', 'application/pdf', 'text/rtf', 'application/rtf',
    'text/xml', 'application/xml',
}
# ---------------------------------


class GeminiLLMClient:
    """
    Handles asynchronous communication with the Gemini API using google-generativeai,
    supporting multimodal inputs (text, images, audio, documents) using
    both INLINE DATA (for smaller files) and FILE UPLOAD (for larger files).
    Manages multi-turn chat history manually for different users.
    Uses the model name and system instruction provided during initialization.
    Constructs input parts as dictionaries.
    """
    def __init__(
        self,
        db_manager: ConversationDBManager,
        api_key: str = settings.GEMINI_API_KEY,
        model_name: str = settings.GEMINI_MODEL_NAME, 
        system_instruction: str = settings.system_instruction 
    ):
        """
        Initializes the Google GenAI client and chat history storage.
        Args:
            api_key: Your Gemini API key.
            model_name: The name of the Gemini model to use (e.g., "gemini-2.0-flash").
            system_instruction: An optional system instruction for the model.
                        **Warning:** Ensure this model supports multimodal input
                        and is suitable for the system instruction.
        """
        self.model_name = model_name
        self.api_key = api_key
        self.system_instruction = system_instruction # Store system instruction
        self.db_manager = db_manager

        try:
            genai.configure(api_key=api_key)
            # Pass system_instruction to the GenerativeModel
            self.model = genai.GenerativeModel(self.model_name, system_instruction=self.system_instruction)
            logger.info(f"Initialized Google GenAI GenerativeModel: {self.model_name}")
            if self.system_instruction:
                logger.info("Initialized model with a system instruction.")
        except Exception as e:
            logger.critical(f"CRITICAL: Failed to initialize Google GenAI with model {self.model_name}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize GenAI with model {self.model_name}: {e}") from e

        # Storing dicts for user input and Content objects from model response
        # Note: Storing large inline data in history might consume significant memory.
        self.chat_histories: Dict[str, List[Any]] = {}

        self._supported_mimetypes = SUPPORTED_GENAI_MIMETYPES
        logger.warning(f"GeminiLLMClient initialized with model '{self.model_name}'. Files will be sent as INLINE DATA (under {INLINE_FILE_SIZE_THRESHOLD_MB}MB) or via UPLOAD (over {INLINE_FILE_SIZE_THRESHOLD_MB}MB).")


    async def _format_db_history_for_genai(self, db_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Formats DB history (list of dicts) into GenAI's input history format
        (list of dicts with 'role' and 'parts' keys).
        Includes only text messages for context.
        """
        genai_formatted_history = []
        for turn in db_history:
            role = turn.get('role') # 'user' or 'assistant'
            msg_type = turn.get('type')
            text_content = turn.get('text_content')

            # Map DB role ('assistant') to GenAI role ('model')
            genai_role = 'model' if role == 'assistant' else 'user'

            parts = []
            # Only include text content in the history parts for context
            if msg_type == 'text' and text_content:
                parts.append({"text": text_content}) # Part is a dictionary with 'text' key

            if parts: # Only add turns with valid text parts
                genai_formatted_history.append({"role": genai_role, "parts": parts})
            else:
                 logger.debug(f"Skipping non-text/empty turn from DB history: Role={role}, Type={msg_type}")

        logger.debug(f"Formatted {len(genai_formatted_history)} turns from DB history for GenAI input.")
        return genai_formatted_history

    async def _get_or_create_chat_history(self, sender_phone: str) -> List[Dict[str, Any]]:
        """
        Retrieves an existing in-memory chat history list or creates a new one,
        attempting to load initial history from the database.
        """
        if sender_phone not in self.chat_histories:
            logger.info(f"No active chat history for {sender_phone}. Checking database...")
            formatted_history: List[Dict[str, Any]] = [] # Default to empty list
            try:
                # 1. Fetch history using db_manager
                db_history_rows = await self.db_manager.get_history(sender_phone) # Limit is handled in db_manager
                # 2. Format it if found
                if db_history_rows:
                    formatted_history = await self._format_db_history_for_genai(db_history_rows)
                    logger.info(f"Loaded {len(formatted_history)} history turns from DB for {sender_phone}.")
                else:
                    logger.info(f"No history found in DB for {sender_phone}.")
            except Exception as e:
                logger.error(f"Failed to retrieve or format DB history for {sender_phone}: {e}. Starting chat with empty history.", exc_info=True)
                formatted_history = [] # Ensure it's empty on error

            # 3. Store the loaded (or empty) history in the in-memory cache
            self.chat_histories[sender_phone] = formatted_history

        return self.chat_histories[sender_phone]

    # --- Methods for file processing (_get_supported_mime_type, _prepare_inline_file_part, _upload_file, _process_file_for_api) ---
    # (These remain unchanged from the previous "working code" version)

    async def _get_supported_mime_type(self, local_file_path: str) -> str:
        """
        Determines the MIME type of a file and checks if it's supported.
        Raises HTTPException if type cannot be determined or is unsupported.
        """
        mime_type, _ = mimetypes.guess_type(local_file_path)
        if not mime_type:
            ext = os.path.splitext(local_file_path)[1].lower()
            # Add explicit checks for common extensions mimetypes might miss
            if ext == ".heic": mime_type = "image/heic"
            elif ext == ".heif": mime_type = "image/heif"
            # Add other specific extensions if needed, e.g., for documents not always guessed
            # elif ext == ".docx": mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            # etc.
            else:
                 logger.error(f"Could not determine MIME type for file: {local_file_path}")
                 raise HTTPException(
                     status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                     detail=f"Could not determine MIME type for file: {os.path.basename(local_file_path)}"
                 )

        if mime_type not in self._supported_mimetypes:
             logger.error(f"Unsupported MIME type '{mime_type}' for file: {local_file_path}")
             raise HTTPException(
                 status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                 detail=f"Unsupported file type: {mime_type}. Please provide a supported format."
             )
        return mime_type


    async def _prepare_inline_file_part(self, local_file_path: str, mime_type: str) -> Dict[str, Any]:
        """
        Reads file content, base64 encodes it, and prepares a dictionary
        for inline data suitable for the API request.
        Runs the synchronous file read in a thread pool.
        """
        try:
            logger.info(f"Reading file for inline data: {local_file_path}")

            # Read file content in binary mode - must be done in a thread for async
            file_bytes = await asyncio.to_thread(lambda: open(local_file_path, 'rb').read())

            # Base64 encode the file content
            base64_data = base64.b64encode(file_bytes).decode('utf-8')

            logger.info(f"Prepared inline data ({len(file_bytes)} bytes) for file: {os.path.basename(local_file_path)}")

            # --- Construct the inline data part as a dictionary ---
            return {
                "inline_data": {
                    "mime_type": mime_type,
                    "data": base64_data
                }
            }
            # ---

        except Exception as e:
            logger.error(f"Failed to prepare inline data for {local_file_path}: {type(e).__name__} - {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to prepare inline file data for: {os.path.basename(local_file_path)}")

    # --- Re-added function for file upload ---
    async def _upload_file(self, local_file_path: str, mime_type: str) -> Dict[str, Any]:
        """
        Uploads a file using genai.upload_file.
        Returns a dictionary representing the file part.
        Runs the synchronous upload in a thread pool.
        """
        try:
            logger.info(f"Uploading file: {local_file_path} with mime_type: {mime_type}")
            # genai.upload_file is synchronous, so we use asyncio.to_thread
            uploaded_file = await asyncio.to_thread(
                genai.upload_file,
                path=local_file_path,
                mime_type=mime_type
            )
            logger.info(f"File uploaded successfully: URI={uploaded_file.uri}")

            # --- Construct the file part as a dictionary using file_data ---
            return {
                "file_data": {
                    "mime_type": uploaded_file.mime_type, # Use mime type from upload response
                    "file_uri": uploaded_file.uri
                }
            }
            # ---

        except Exception as e:
            logger.error(f"File upload failed for {local_file_path}: {type(e).__name__} - {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to upload file: {os.path.basename(local_file_path)}")

    # --- New function to decide and process file ---
    async def _process_file_for_api(self, local_file_path: str) -> Dict[str, Any]:
        """
        Checks file size and decides whether to send as inline data or upload.
        Calls the appropriate method (_prepare_inline_file_part or _upload_file)
        and returns the file part dictionary.
        Handles file existence and MIME type checks first.
        """
        if not os.path.exists(local_file_path):
            logger.warning(f"File not found at provided path: {local_file_path}")
            raise HTTPException(status_code=404, detail=f"File not found: {os.path.basename(local_file_path)}")

        try:
            # 1. Get and validate MIME type
            mime_type = await self._get_supported_mime_type(local_file_path) # This already raises HTTPException if needed

            # 2. Get file size (must be done in thread)
            file_size = await asyncio.to_thread(os.path.getsize, local_file_path)
            logger.info(f"File size for {os.path.basename(local_file_path)}: {file_size} bytes")


            # 3. Decide based on size
            if file_size <= INLINE_FILE_SIZE_THRESHOLD_BYTES:
                # Use inline data for smaller files
                logger.info(f"File {os.path.basename(local_file_path)} is <= {INLINE_FILE_SIZE_THRESHOLD_MB}MB, preparing inline data.")
                return await self._prepare_inline_file_part(local_file_path, mime_type)
            else:
                # Use file upload for larger files
                logger.info(f"File {os.path.basename(local_file_path)} is > {INLINE_FILE_SIZE_THRESHOLD_MB}MB, initiating upload.")
                return await self._upload_file(local_file_path, mime_type)

        except HTTPException:
             # Re-raise HTTPExceptions from _get_supported_mime_type or raised above
             raise
        except Exception as e:
            # Catch any unexpected errors during size check or decision
            logger.error(f"Unexpected error processing file {local_file_path}: {type(e).__name__} - {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to process file {os.path.basename(local_file_path)}")


    async def process_user_message(
        self,
        sender_phone: str,
        input_data: Dict[str, Any]
    ) -> str:
        """
        Processes a user message which can include text and/or a file path.
        Decides whether to send file as inline data or upload based on size,
        constructs the prompt parts, sends to the Gemini API, manages history,
        and returns the text response.
        """
        user_prompt_text: Optional[str] = input_data.get('text_content')
        local_file_path: Optional[str] = input_data.get('local_file_path')

        if not user_prompt_text and not local_file_path:
             logger.warning(f"Received request from {sender_phone} with no text or file path.")
             return "Please provide a message or a file to process."

        try:
            history = await self._get_or_create_chat_history(sender_phone)
            # --- input_parts will now be a list of dictionaries (can include inline_data or file_data) ---
            input_parts: List[Dict[str, Any]] = []

            # --- Process file if a file path is provided ---
            if local_file_path:
                # Call the new function that handles size check and calls inline/upload
                file_part_dict = await self._process_file_for_api(local_file_path)
                input_parts.append(file_part_dict)
            # ---

            # --- Construct the text part as a dictionary ---
            if user_prompt_text:
                input_parts.append({"text": user_prompt_text})
            # ---

            if not input_parts:
                 # This should ideally not happen if either text or file is present
                 logger.error(f"Internal error: No input parts created for {sender_phone}")
                 raise HTTPException(status_code=500, detail="Internal error processing input.")

            # Construct the current user message as a dictionary containing the list of part dictionaries
            current_user_content = {"role": "user", "parts": input_parts}

            # Combine history with the current user message for the API call
            messages_to_send = history + [current_user_content]

            logger.info(f"Sending request to model '{self.model_name}' for {sender_phone} with {len(input_parts)} parts (history length: {len(history)}).")

            # Use the asynchronous version of generate_content
            response = await self.model.generate_content_async(messages_to_send)

            logger.info(f"Received response from model '{self.model_name}' for {sender_phone}")

            try:
                ai_response_text = response.text
                # Append the actual Content object returned by the model
                model_response_content = response.candidates[0].content

                # Append the user's content (as a dict, now potentially containing inline_data or file_data)
                # and the model's response content (as a Content object) to history
                # CAUTION: Storing inline data in history can still consume significant memory.
                # The upload_file method is better for history efficiency with large files.
                history.append(current_user_content)
                history.append(model_response_content)
                # Optional: Implement history trimming logic here if needed (e.g., self.chat_histories[sender_phone] = history[-N:])

            except (AttributeError, IndexError, StopIteration, Exception) as response_error:
                # This block handles cases where the response doesn't have expected structure (e.g., empty candidates, no text)
                logger.warning(f"Could not extract valid response/content for {sender_phone} from model {self.model_name}. Error: {response_error}", exc_info=True)
                try:
                    # Check for safety block reason
                    if response and hasattr(response, 'prompt_feedback') and response.prompt_feedback and hasattr(response.prompt_feedback, 'block_reason'):
                         block_reason = response.prompt_feedback.block_reason
                         if block_reason:
                             logger.warning(f"Response blocked for {sender_phone} due to: {block_reason}")
                             return f"My response was blocked due to safety reasons ({block_reason}). Please modify your prompt or file."

                    # Check if candidates list is empty or invalid even if no explicit block
                    elif response and (not hasattr(response, 'candidates') or not response.candidates):
                         logger.warning(f"Model response had no candidates for {sender_phone}.")
                         return "Sorry, I couldn't generate a response for that input."

                except Exception as check_error:
                    # Catch errors while checking feedback/candidates
                    logger.error(f"Error checking response feedback/candidates for {sender_phone}: {check_error}", exc_info=True)
                    pass # Fall through to generic error message

                # Generic fallback message if no specific reason found
                return "Sorry, I encountered an issue generating or understanding the response."

            # Final check on the generated text content
            if ai_response_text is None or ai_response_text.strip() == "":
                logger.warning(f"LLM for {sender_phone} (model {self.model_name}) returned an empty or null response text.")
                return "Sorry, I couldn't generate a meaningful response for that."

            return ai_response_text.strip()

        except HTTPException as e:
            # Re-raise HTTPExceptions (like unsupported media type, file not found, or file processing errors)
            logger.warning(f"HTTPException during processing for {sender_phone}: {e.status_code} - {e.detail}")
            raise e
        except Exception as e:
            # Catch any other unexpected errors during the entire process
            logger.error(f"An unexpected error occurred during LLM processing for {sender_phone} with model {self.model_name}: {type(e).__name__} - {e}", exc_info=True)
            # Re-raise as HTTPException to be caught by the outer handler (e.g., in llm_service.py)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"An unexpected error occurred while processing your request."
            )
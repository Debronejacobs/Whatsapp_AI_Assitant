# whatsapp_client/app/models.py

from pydantic import BaseModel, Field
from typing import Any, Dict, Optional, List

# --- Models for Receiving Webhooks ---

# Model for the 'media' object within the webhook payload data
class ReceivedMediaInfo(BaseModel):
    url: str = Field(..., description="URL to download the media file from the API server.")
    type: str = Field(..., description="MIME type of the media file.")
    extension: str = Field(..., description="File extension of the media.")
    filename: Optional[str] = Field(None, description="Original filename of the media.")

# Based on the ACTUAL payload structure observed in the Node.js log
class IncomingMessageData(BaseModel):
     id: str = Field(..., description="The unique ID of the message (string).")
     type: str = Field(..., description="The type of message (e.g., text, image, video, document, etc.).")

     sender_phone: str = Field(..., alias="from", description="The sender's phone number in international format.")
     recipient_phone: str = Field(..., alias="to", description="The recipient's phone number in international format.")

     body: Optional[str] = Field(None, description="The message body (for text messages). Can be null/missing for media, etc.")
     date_str: str = Field(..., alias="date", description="Date string of the message.")
     timestamp: int = Field(..., description="Unix timestamp of the message.")
     is_temporary: bool = Field(..., description="Indicates if the message is temporary.")
     is_forwarded: bool = Field(..., description="Indicates if the message was forwarded.")
     is_mine: bool = Field(..., description="Indicates if the message was sent by the connected account.")
     is_broadcast: bool = Field(..., description="Indicates if the message was a broadcast.")

     # Add the optional media field using the new model
     media: Optional[ReceivedMediaInfo] = Field(None, description="Media information for media messages.")

     # Add a field to store the local path AFTER download in main.py
     local_file_path: Optional[str] = Field(None, description="Local file path if media was downloaded.")

     class Config:
        extra = "allow"


# WebhookPayload structure itself is correct (type and data)
class WebhookPayload(BaseModel):
    """
    Model for the expected webhook payload from the WhatsApp API server.
    """
    type: str = Field(..., description="The type of webhook event (e.g., message_received).")
    data: IncomingMessageData # Uses the updated IncomingMessageData model structure

# --- Models for Sending Messages (These remain the same) ---

class SendTextMessageRequest(BaseModel):
    """
    Model for sending a text message.
    """
    message: str = Field(..., description="The text message content.")
    reply_to: Optional[str] = Field(None, description="ID of the message to reply to (should be the message ID string from webhook).")

class SendMediaMessageParams(BaseModel):
    """
    Model for optional parameters when sending media messages via form data.
    """
    message: Optional[str] = Field(None, description="Caption for the media.")
    view_once: Optional[bool] = Field(None, description="Send media as view once.")
    as_document: Optional[bool] = Field(None, description="Send media as document.")
    as_voice: Optional[bool] = Field(None, description="Send audio file as voice message.")
    as_gif: Optional[bool] = Field(None, description="Send video file as GIF.")
    as_sticker: Optional[bool] = Field(None, description="Send image file as sticker.")
    reply_to: Optional[str] = Field(None, description="ID of another message to quote (should be the message ID string from webhook).")

    class Config:
        extra = "allow"

# --- Models for Database Storage ---

# Represents a message or AI turn stored in the database
class ConversationMessage(BaseModel):
    # Note: 'id' would be the DB primary key, usually auto-generated
    id: Optional[int] = Field(None, description="Database primary key (auto-generated).")
    sender_phone: str = Field(..., description="The phone number identifying the conversation.")
    role: str = Field(..., description="Role of the message sender ('user' or 'assistant').")
    message_id: Optional[str] = Field(None, description="Original WhatsApp message ID if applicable.") # Optional for AI messages
    type: str = Field(..., description="Message type ('text', 'image', 'document', etc.).")
    text_content: Optional[str] = Field(None, description="Text content of the message.")
    local_file_path: Optional[str] = Field(None, description="Local file path for media/documents.")
    timestamp: int = Field(..., description="Unix timestamp of the message.")

    class Config:
        from_attributes = True # Allows creating model instances from ORM objects or dicts

# --- Models for LLM Interaction Formatting (No longer needed as internal helper removed) ---
# We can remove LLMPart, LLMInlineData, LLMMessage if they aren't used elsewhere
# as the genai library types are used directly now.
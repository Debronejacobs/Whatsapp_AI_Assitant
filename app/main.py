# whatsapp_client/app/main.py

import httpx
from fastapi import FastAPI, Request, HTTPException, status, UploadFile, File, Form
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import os
import aiofiles
import pathlib
from typing import Optional, Dict, Any, List
import uuid

# --- Import Application Modules ---
from .config import settings
from .models import (
    WebhookPayload,
    SendTextMessageRequest,
    SendMediaMessageParams,
    IncomingMessageData,
    ReceivedMediaInfo
)
from .db_manager import ConversationDBManager
from .llm_client import GeminiLLMClient
from .llm_service import WhatsAppLLMService

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Variables for Application Components ---
client: httpx.AsyncClient
db_manager: ConversationDBManager
llm_client: GeminiLLMClient
llm_service: WhatsAppLLMService


# --- Application Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global client, db_manager, llm_client, llm_service
    logger.info("Application startup sequence initiated...")

    # 1. Initialize WhatsApp API HTTP Client
    auth_headers = {}
    if settings.AUTH_TOKEN: auth_headers["Authorization"] = f"Bearer {settings.AUTH_TOKEN}"
    client = httpx.AsyncClient(base_url=settings.API_URL, headers=auth_headers, follow_redirects=True, timeout=30.0)
    logger.info(f"WhatsApp API HTTP Client initialized (Base URL: {settings.API_URL})")

    # 2. Initialize and Connect Database Manager
    try:
        db_manager = ConversationDBManager(settings.DATABASE_URL)
        await db_manager.connect()
        logger.info("Database Manager connected successfully.")
    except Exception as e:
        logger.critical(f"CRITICAL: Failed to connect to database: {e}", exc_info=True)
        raise RuntimeError(f"Database connection failed: {e}") from e

    # 3. Initialize LLM Client
    llm_client = GeminiLLMClient(db_manager=db_manager)

    logger.info("Gemini LLM Client initialized.")

    # 4. Initialize LLM Service Layer
    llm_service = WhatsAppLLMService(
        db_manager=db_manager,
        llm_client=llm_client,
        whatsapp_text_sender=send_text_message
    )
    logger.info("WhatsApp LLM Service initialized.")

    # 5. Create Local Media Storage Directory
    try:
        media_dir = settings.MEDIA_STORAGE_DIR
        os.makedirs(media_dir, exist_ok=True)
        logger.info(f"Media storage directory ensured at: {media_dir}")
    except Exception as e:
        logger.error(f"Failed to create media storage directory {settings.MEDIA_STORAGE_DIR}: {e}", exc_info=True)

    # 6. Set WhatsApp Presence to Online (via API Server)
    try:
        logger.info("Attempting to set WhatsApp presence to online via API server...")
        api_response = await call_whatsapp_api(method="POST", endpoint="/set-online")
        if api_response and isinstance(api_response, dict) and api_response.get("status") is True:
             logger.info("Successfully requested WhatsApp presence to be set online.")
        else:
             logger.warning(f"API call to set online successful, but unexpected response body: {api_response}")
    except Exception as e:
         logger.error(f"An unexpected error occurred while setting WhatsApp presence online via API: {e}", exc_info=True)

    logger.info("Application startup complete. Ready to receive requests.")
    yield # Application runs

    # --- Shutdown Logic ---
    logger.info("Application shutdown sequence initiated...")
    if 'db_manager' in globals() and db_manager:
        try: await db_manager.close()
        except Exception as e: logger.error(f"Error closing database: {e}", exc_info=True)
    if 'client' in globals() and client:
        try: await client.aclose()
        except Exception as e: logger.error(f"Error closing HTTP client: {e}", exc_info=True)
    logger.info("Application shutdown complete.")

app = FastAPI(
    title="WhatsApp Client with LLM",
    description="Receives WhatsApp messages, processes with Gemini, stores history, and replies.",
    version="1.2.1", # Version bump
    lifespan=lifespan
)

async def call_whatsapp_api(method: str, endpoint: str, **kwargs):
    
    if 'client' not in globals() or not client:
        logger.error("WhatsApp API client is not initialized.")
        raise HTTPException(status_code=503, detail="WhatsApp API client not ready.")
    try:
        log_level = logging.DEBUG
        if endpoint in ["/status", "/check-login", "/set-online", "/set-offline"]: log_level = logging.INFO
        logging.log(log_level, f"Calling WhatsApp API Server: {method.upper()} {endpoint}")

        response = await client.request(method=method.upper(), url=endpoint, **kwargs)
        response.raise_for_status()
        logging.log(log_level, f"WhatsApp API Server call successful: {method.upper()} {endpoint} -> Status {response.status_code}")
        try: return response.json()
        except Exception:
             logging.warning(f"WhatsApp API Server response for {endpoint} not JSON. Status: {response.status_code}, Body: '{response.text[:100]}...'")
             return {"detail": "Non-JSON response", "status_code": response.status_code, "body": response.text}
    except httpx.ConnectError as e:
        logger.error(f"Connection error calling WhatsApp API Server at {settings.API_URL}{endpoint}: {e}")
        raise HTTPException(status_code=503, detail=f"Could not connect to WhatsApp API server. Error: {e}")
    except httpx.HTTPStatusError as e:
        logging.error(f"HTTP error calling WhatsApp API Server at {settings.API_URL}{endpoint}: {e.response.status_code} - {e.response.text}")
        try: error_detail = e.response.json()
        except Exception: error_detail = {"detail": e.response.text}
        api_error_message = error_detail.get('error', error_detail.get('detail', e.response.text))
        raise HTTPException(status_code=e.response.status_code, detail=f"WhatsApp API Server error: {api_error_message}")
    except Exception as e:
        logger.error(f"Unexpected error calling WhatsApp API Server at {settings.API_URL}{endpoint}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error calling WhatsApp API Server: {e}")

# --- Webhook Endpoint for Receiving Messages ---
@app.post("/webhook", status_code=status.HTTP_200_OK)
async def receive_webhook(payload: WebhookPayload, request: Request):
    global llm_service
    if 'llm_service' not in globals() or not llm_service:
        logger.error("LLM Service not initialized. Cannot process webhook.")
        raise HTTPException(status_code=503, detail="LLM Service not ready.")

    logger.info(f"Webhook received: Type = {payload.type}")
    if payload.type == "message_received":
        message_data: IncomingMessageData = payload.data
        sender = message_data.sender_phone
        msg_type = message_data.type
        logger.info(f"Processing message_received from {sender} (ID: {message_data.id}, Type: {msg_type})")

        # --- Media Download Logic ---
        local_file_path = None
        if not message_data.is_mine:
            media_types = ['image', 'video', 'document', 'sticker', 'audio']
            if message_data.type in media_types and message_data.media and message_data.media.url:
                media_url = message_data.media.url
                try:
                    original_filename = message_data.media.filename or f"file.{message_data.media.extension or 'bin'}"
                    safe_filename_base = "".join(c for c in os.path.splitext(original_filename)[0] if c.isalnum() or c in ('_', '-')).rstrip('._-') or str(uuid.uuid4())
                    safe_extension = "".join(c for c in os.path.splitext(original_filename)[1] if c.isalnum() or c == '.').lstrip('.') or message_data.media.extension or "bin"
                    # --- FIX: Use message_data.id directly ---
                    unique_filename = f"{message_data.id}_{safe_filename_base}.{safe_extension}"
                    # --- End FIX ---
                    local_file_path = os.path.join(settings.MEDIA_STORAGE_DIR, unique_filename)

                    logger.info(f"Attempting download: {media_url} -> {local_file_path}...")
                    async with client.stream("GET", media_url, timeout=60.0) as response:
                        response.raise_for_status()
                        async with aiofiles.open(local_file_path, mode='wb') as f:
                            async for chunk in response.aiter_bytes(): await f.write(chunk)
                        logger.info(f"Media saved to {local_file_path}")
                        message_data.local_file_path = local_file_path # Add path to data object
                except Exception as e:
                    logger.error(f"Failed media download/save from {media_url}: {e}", exc_info=True)
                    local_file_path = None # Ensure path is None if download fails
                    message_data.local_file_path = None # Also ensure it's None on the object
            # --- End Media Download ---

            # --- Delegate to LLM Service ---
            try:
                await llm_service.handle_incoming_message(message_data)
            except Exception as service_error:
                logger.error(f"Error processing message in LLM Service: {service_error}", exc_info=True)
            # --- End Delegate ---

    return {"status": "success", "message": "Webhook received"}



# --- Endpoint for Sending Text Messages (Used by Service and potentially externally) ---
@app.post("/send/text/{number}", status_code=status.HTTP_200_OK, include_in_schema=False)
async def send_text_message(number: str, request_body: SendTextMessageRequest):
    # (This function remains unchanged from the previous correct version)
    logger.info(f"Initiating send text message to WhatsApp API Server for {number}")
    cleaned_number_for_url = number.lstrip('+')
    if not cleaned_number_for_url:
        logger.error(f"Invalid number format for sending text: '{number}'")
        raise HTTPException(status_code=400, detail="Invalid number format for sending.")
    payload = { "message": request_body.message }
    if request_body.reply_to: payload["reply_to"] = request_body.reply_to
    api_endpoint = f"/send-message/{cleaned_number_for_url}"
    logger.info(f"Calling WhatsApp API Server: POST {api_endpoint} with payload: {payload}")
    try:
        api_response = await call_whatsapp_api(method="POST", endpoint=api_endpoint, json=payload)
        logger.info(f"WhatsApp API Server response for sending text: {api_response}")
        return {"status": "success", "message": "Text message request forwarded to API", "api_response": api_response}
    except HTTPException as e:
        logger.error(f"Failed to forward text message request to API: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in send_text_message endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during text send.")





# --- Optional:  Status Endpoint ---
@app.get("/status", status_code=status.HTTP_200_OK)
async def get_status():
    # (This function remains unchanged from the previous correct version)
    logger.info("Checking connectivity status to WhatsApp API server...")
    try:
        api_status = await call_whatsapp_api(method="GET", endpoint="/check-login")
        return {"client_status": "running", "whatsapp_api_server_status": api_status}
    except HTTPException as e:
         logger.error(f"Failed to get WhatsApp API server status: {e.detail}")
         return JSONResponse(status_code=e.status_code, content={"client_status": "running", "whatsapp_api_server_status": "error", "detail": e.detail})
    except Exception as e:
         logger.error(f"Unexpected error checking WhatsApp API server status: {e}", exc_info=True)
         return JSONResponse(status_code=500, content={"client_status": "running", "whatsapp_api_server_status": "error", "detail": "An unexpected internal error occurred."})


# --- Root Endpoint ---
@app.get("/", status_code=status.HTTP_200_OK, include_in_schema=False)
async def read_root():
    # (This function remains unchanged from the previous correct version)
    return {"message": "WhatsApp Client with LLM Integration is Running"}





# --- Endpoint for Sending Media Messages (External use) ---
# @app.post("/send/media/{number}", status_code=status.HTTP_200_OK)
# async def send_media_message(
#     number: str, file: UploadFile = File(...), message: Optional[str] = Form(None),
#     view_once: Optional[bool] = Form(None), as_document: Optional[bool] = Form(None),
#     as_voice: Optional[bool] = Form(None), as_gif: Optional[bool] = Form(None),
#     as_sticker: Optional[bool] = Form(None), reply_to: Optional[str] = Form(None),
# ):
#     # (This function remains unchanged from the previous correct version)
#     logger.info(f"Received external request to send media message to {number} with file: {file.filename}")
#     cleaned_number_for_url = number.lstrip('+')
#     if not cleaned_number_for_url: raise HTTPException(status_code=400, detail="Invalid number format.")
#     try: file_content = await file.read()
#     except Exception as e:
#         logger.error(f"Failed to read uploaded file for external send: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail="Could not read uploaded file.")
#     files = {"file": (file.filename, file_content, file.content_type)}
#     data = {}
#     if message is not None: data["message"] = message
#     if view_once is not None: data["view_once"] = str(view_once).lower()
#     if as_document is not None: data["as_document"] = str(as_document).lower()
#     if as_voice is not None: data["as_voice"] = str(as_voice).lower()
#     if as_gif is not None: data["as_gif"] = str(as_gif).lower()
#     if as_sticker is not None: data["as_sticker"] = str(as_sticker).lower()
#     if reply_to is not None: data["reply_to"] = reply_to
#     api_endpoint = f"/send-media/{cleaned_number_for_url}"
#     logger.info(f"Forwarding external media message request to WhatsApp API Server: POST {api_endpoint}")
#     try:
#         api_response = await call_whatsapp_api(method="POST", endpoint=api_endpoint, files=files, data=data)
#         logger.info(f"WhatsApp API Server response for sending media: {api_response}")
#         return {"status": "success", "message": "Media message request forwarded to API", "api_response": api_response}
#     except HTTPException as e:
#         logger.error(f"Failed to forward media message request to API: {e.detail}")
#         raise e
#     except Exception as e:
#         logger.error(f"Unexpected error in send_media_message endpoint: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail="Internal server error during media send.")
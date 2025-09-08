# routes/voice.py
import os
import urllib.parse
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, WebSocket
import asyncio
from sqlalchemy.orm import Session

from config import settings
from db.database import Chat, get_db, get_or_create_user_by_external_id
from logger_config import logger
from models.schemas import ChatResponse
from services.llm import generate_response
from services.redis_service import retrieve_context
from services.background import index_chat, update_chat_response
from services.stt import transcribe_audio
from services.tts import generate_speech
from services.streaming import websocket_stream

router = APIRouter()


@router.post("/upload", response_model=ChatResponse)
async def upload_voice(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    generate_audio: bool = Form(False),
    db: Session = Depends(get_db),
):
    """Accept an audio upload, transcribe it, generate an LLM response, and
    schedule background tasks to index the transcript and persist the LLM
    response so the HTTP response is fast.
    """
    logger.info("Voice upload requested by user %s, generate_audio=%s", user_id, generate_audio)

    if not file or file.filename is None:
        raise HTTPException(status_code=400, detail="No file provided")

    if not file.filename.lower().endswith((".wav", ".mp3", ".m4a")):
        raise HTTPException(status_code=400, detail="Only audio files (wav, mp3, m4a) are supported")

    # Ensure audio dir exists
    audio_dir = os.path.join(settings.assets_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    # Persist uploaded file to disk
    file_extension = file.filename.split(".")[-1].lower()
    unique_filename = f"{uuid.uuid4()}.{file_extension}"
    file_path = os.path.join(audio_dir, unique_filename)

    try:
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
            f.flush()
            os.fsync(f.fileno())
        if os.path.getsize(file_path) == 0:
            raise ValueError("Saved file is empty")
    except Exception as exc:
        logger.exception("Failed saving uploaded file: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to save audio file")

    # Transcribe (synchronous helper) - keep in thread if it's blocking elsewhere
    try:
        transcription = transcribe_audio(file_path)
        logger.info("Transcription completed length=%d", len(transcription or ""))
    except Exception as exc:
        logger.exception("Transcription failed: %s", exc)
        raise HTTPException(status_code=500, detail="Transcription failed")

    # Resolve canonical DB user id (create if missing)
    try:
        db_user_id = get_or_create_user_by_external_id(db, user_id)
    except Exception as exc:
        logger.exception("Failed to resolve/create user: %s", exc)
        raise HTTPException(status_code=500, detail="User resolution failed")

    # Persist chat synchronously so we have chat.id for background tasks
    try:
        chat = Chat(user_id=db_user_id, message=transcription, response=None)
        db.add(chat)
        db.commit()
        db.refresh(chat)
        logger.info("Stored chat id=%s for user=%s", chat.id, db_user_id)
    except Exception as exc:
        logger.exception("Failed to persist chat: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to store chat")

    # Determine concrete chat id and schedule background indexing (non-blocking)
    chat_id = None
    try:
        chat_id = getattr(chat, "id", None)
        if chat_id is not None:
            try:
                chat_id = int(chat_id)
            except Exception:
                # fallback: leave as-is (background functions will validate)
                pass
            asyncio.create_task(asyncio.to_thread(index_chat, db_user_id, transcription, chat_id))
    except Exception:
        logger.exception("Failed to schedule background indexing")

    # Retrieve any existing context for the user (will be limited if indexing is async)
    try:
        context = retrieve_context(transcription, db_user_id)
        logger.info("Context retrieval completed length=%d", len(context or ""))
    except Exception:
        logger.exception("Context retrieval failed; continuing without it")
        context = ""

    # Generate LLM response (synchronous)
    try:
        response_text = generate_response(transcription, context)
    except Exception:
        logger.exception("LLM generation failed")
        response_text = ""

    # Schedule background update to persist the LLM response
    try:
        if chat_id is None:
            chat_id = getattr(chat, "id", None)
            try:
                chat_id = int(chat_id) if chat_id is not None else None
            except Exception:
                pass
        if chat_id is not None:
            asyncio.create_task(asyncio.to_thread(update_chat_response, chat_id, response_text))
    except Exception:
        logger.exception("Failed to schedule background chat response update")

    # Optionally generate TTS audio for the response
    audio_url = None
    if generate_audio and response_text:
        try:
            response_filename = f"{uuid.uuid4()}.wav"
            audio_response_path = os.path.join(audio_dir, response_filename)
            generate_speech(response_text, audio_response_path)
            if os.path.exists(audio_response_path):
                rel_path = os.path.relpath(audio_response_path, settings.assets_dir).replace("\\", "/")
                audio_url = f"/static/{urllib.parse.quote(rel_path)}"
        except Exception:
            logger.exception("Failed to generate TTS audio")

    return {
        "response": response_text,
        "timestamp": chat.timestamp.isoformat() if getattr(chat, "timestamp", None) is not None else None,
        "audio_url": audio_url,
    }


@router.websocket("/ws")
async def websocket_stream_route(websocket: WebSocket):
    # Delegate to service module for clarity
    await websocket_stream(websocket)



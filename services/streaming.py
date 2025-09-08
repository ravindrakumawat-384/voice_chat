import os
import urllib.parse
import uuid
import json
import shutil
import subprocess
import time
from typing import Dict
from fastapi import WebSocket, WebSocketDisconnect

from config import settings
from logger_config import logger
from services.stt import transcribe_audio
from services.tts import generate_speech
from services.redis_service import retrieve_context, index_transcript
from services.llm import generate_response
from db.database import SessionLocal, get_or_create_user_by_external_id, Chat
from datetime import datetime

# Simple in-memory concurrency map: user_id -> active_stream_count
active_streams: Dict[int, int] = {}
MAX_STREAMS_PER_USER = 2
MAX_STREAM_BYTES = 50 * 1024 * 1024  # 50 MB per stream
MAX_STREAM_SECONDS = 300  # 5 minutes


async def websocket_stream(websocket: WebSocket):
    """WebSocket handler moved to a dedicated service module so it's easier to test.
    Receives binary audio chunks, writes to a .webm stream file, converts to WAV
    (ffmpeg preferred), runs transcription, LLM response, optional TTS, then
    sends a final JSON result back to the client.
    """
    await websocket.accept()
    params = dict(websocket.query_params)
    user_id_param = params.get("user_id", "0")
    token_param = params.get("token") or websocket.headers.get("authorization")
    # Validate token if configured
    if settings.ws_auth_token:
        if not token_param or (token_param != settings.ws_auth_token and token_param.replace("Bearer ", "") != settings.ws_auth_token):
            await websocket.close(code=4401)
            logger.warning("WebSocket rejected due to missing/invalid token")
            return
    # Resolve external user id to DB id using a short-lived session
    try:
        external_user_id = user_id_param
        with SessionLocal() as s:
            db_user_id = get_or_create_user_by_external_id(s, external_user_id)
    except Exception:
        db_user_id = 0

    user_id = int(db_user_id)
    logger.info(f"WebSocket stream connected for external_user={user_id_param} db_user={user_id}")

    # concurrency guard
    if user_id not in active_streams:
        active_streams[user_id] = 0
    if active_streams[user_id] >= MAX_STREAMS_PER_USER:
        await websocket.send_text(json.dumps({"error": "too_many_streams"}))
        await websocket.close(code=4409)
        return
    active_streams[user_id] += 1

    stream_dir = os.path.join(settings.assets_dir, "audio", "streams")
    os.makedirs(stream_dir, exist_ok=True)
    stream_path = os.path.join(stream_dir, f"{uuid.uuid4()}.webm")

    # Receive loop: write binary chunks and watch for stop event
    bytes_received = 0
    start_time = time.time()
    try:
        with open(stream_path, "ab") as f:
            while True:
                # safety: enforce time limit
                if time.time() - start_time > MAX_STREAM_SECONDS:
                    logger.warning("Stream exceeded max duration, closing")
                    await websocket.send_text(json.dumps({"error": "timeout"}))
                    break

                msg = await websocket.receive()

                # Binary chunk
                if msg.get("bytes") is not None:
                    chunk = msg.get("bytes")
                    if chunk:
                        f.write(chunk)
                        f.flush()
                        bytes_received += len(chunk)
                    # Optionally send an ack
                    try:
                        await websocket.send_text(json.dumps({"ack": True}))
                    except Exception:
                        pass

                    if bytes_received > MAX_STREAM_BYTES:
                        logger.warning("Stream exceeded max bytes, closing")
                        await websocket.send_text(json.dumps({"error": "too_large"}))
                        break

                # Text control message
                elif msg.get("text") is not None:
                    text = msg.get("text")
                    payload = None
                    try:
                        if text:
                            payload = json.loads(text)
                    except Exception:
                        payload = None

                    if payload and payload.get("event") == "stop":
                        logger.info("Received stop event from client - finalizing stream")
                        break

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected by client during streaming")
        # proceed to try to process what we have
    except Exception as e:
        logger.error(f"Error receiving websocket stream: {e}")
        try:
            await websocket.send_text(json.dumps({"error": "server_error", "detail": str(e)}))
        except Exception:
            pass

    # After receiving 'stop' or disconnect, perform transcription and LLM response
    try:
        if os.path.getsize(stream_path) == 0:
            await websocket.send_text(json.dumps({"error": "empty_stream"}))
            return

        # Convert to WAV if needed (ffmpeg preferred)
        converted_path = stream_path
        try:
            # If stream is webm, convert to wav for transcribe_audio which expects wav
            if stream_path.endswith('.webm') or stream_path.endswith('.ogg'):
                wav_path = stream_path + ".wav"
                ffmpeg_path = shutil.which('ffmpeg')
                if ffmpeg_path:
                    cmd = [ffmpeg_path, '-y', '-i', stream_path, '-ar', '16000', '-ac', '1', wav_path]
                    subprocess.run(cmd, check=True)
                    converted_path = wav_path
                    logger.info(f"Converted stream to WAV via ffmpeg: {wav_path}")
                else:
                    # Try pydub fallback (pydub uses ffmpeg under the hood if available)
                    from pydub import AudioSegment
                    audio = AudioSegment.from_file(stream_path)
                    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
                    wav_path = stream_path + ".wav"
                    audio.export(wav_path, format='wav')
                    converted_path = wav_path
                    logger.info(f"Converted stream to WAV via pydub: {wav_path}")

        except Exception as e:
            logger.error(f"Failed to convert stream to WAV: {e}")
            # proceed and hope transcribe_audio can handle the original

        logger.info("Running transcription on streamed audio")
        transcription = transcribe_audio(converted_path)
        logger.info(f"WebSocket transcription: {transcription}")

        # Persist the transcript as a Chat row and schedule indexing
        try:
            with SessionLocal() as s:
                chat = Chat(user_id=user_id, message=transcription, response=None, timestamp=datetime.utcnow())
                s.add(chat)
                s.commit()
                s.refresh(chat)
                # schedule background indexing
                try:
                    index_transcript(user_id, transcription, chat.id)
                except Exception:
                    logger.exception("Background indexing (websocket) failed; continuing")
        except Exception:
            logger.exception("Failed to persist chat from websocket stream")

        # Retrieve context and generate LLM response using canonical db id
        context = retrieve_context(transcription, user_id)
        logger.info("Retrieved context for websocket LLM: %s", context)
        response_text = generate_response(transcription, context)

        # Optionally generate TTS
        audio_url = None
        try:
            response_filename = f"{uuid.uuid4()}.mp3"
            audio_dir = os.path.join(settings.assets_dir, "audio")
            os.makedirs(audio_dir, exist_ok=True)
            audio_response_path = os.path.join(audio_dir, response_filename)
            generate_speech(response_text, audio_response_path)
            if os.path.exists(audio_response_path):
                rel_path = os.path.relpath(audio_response_path, settings.assets_dir).replace("\\", "/")
                # rel_path already contains the 'audio/...' segment, so don't duplicate it
                audio_url = f"/static/{urllib.parse.quote(rel_path)}"
        except Exception as e:
            logger.error(f"TTS generation failed for websocket stream: {e}")

        result = {
            "transcription": transcription,
            "response": response_text,
            "audio_url": audio_url,
        }
        await websocket.send_text(json.dumps(result))

    except Exception as e:
        logger.error(f"Error processing websocket stream: {e}")
        try:
            await websocket.send_text(json.dumps({"error": "server_error", "detail": str(e)}))
        except Exception:
            pass
    finally:
        # decrement concurrency count
        try:
            active_streams[user_id] = max(0, active_streams.get(user_id, 1) - 1)
        except Exception:
            pass

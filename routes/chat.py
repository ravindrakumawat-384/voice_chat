"""
routes/chat.py

Cleaned and re-ordered routes for chat:
- POST /send    -> accept ChatRequest, retrieve context (Redis), call generate_response, save Chat
- GET  /history/{user_id} -> return chat history for user
- GET  /index/info -> (admin) return Redis/RediSearch index info
"""

from datetime import datetime
import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from models.schemas import ChatRequest, ChatResponse, ChatHistoryResponse
from db.database import get_db, User, Chat, get_or_create_user_by_external_id
from services.llm import generate_response
from services.redis_service import retrieve_context, describe_index_stats, INDEX_NAME
from logger_config import logger as project_logger

# module logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = project_logger.getChild("routes.chat")

router = APIRouter()


# ----------------------
# Helpers
# ----------------------
def resolve_db_user(db: Session, external_user_id: str) -> User:
    """
    Resolve or create a user by the external user id coming from frontend.
    Returns the DB User object.
    Raises HTTPException if user can't be resolved.
    """
    try:
        db_id = get_or_create_user_by_external_id(db, external_user_id)
        user = db.query(User).filter(User.id == db_id).first()
        if not user:
            logger.error("Resolved db_id=%s but no User found in DB", db_id)
            raise HTTPException(status_code=404, detail="User not found after creation")
        return user
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to resolve/create user: %s", e)
        raise HTTPException(status_code=500, detail="Failed to resolve user")


# ----------------------
# Routes
# ----------------------
@router.post("/send", response_model=ChatResponse)
async def send_message(request: ChatRequest, db: Session = Depends(get_db)):
    """
    Receive a message from the frontend, retrieve user-specific context (Redis),
    call LLM generation, and persist the chat in DB.
    """
    logger.info("POST /send user=%s", request.user_id)

    # Resolve user (create if necessary)
    user = resolve_db_user(db, request.user_id)
    logger.debug("Resolved user: db_id=%s external_id=%s", user.id, request.user_id)

    # Retrieve context from Redis (vector search)
    try:
        context = retrieve_context(request.message, user.id, top_k=3)
        logger.info("Inside endpoint /send, context: %s", context)
        if context is None:
            context = ""
        # treat internal error strings as empty context but log
        if isinstance(context, str) and context.startswith("Error"):
            logger.warning("retrieve_context returned error: %s", context)
            context = ""
    except Exception as e:
        logger.exception("Error retrieving context from Redis: %s", e)
        context = ""

    # Generate LLM response
    try:
        logger.info("Generating LLM response for user=%s", user.id)
        response_text = generate_response(request.message, context)
    except Exception as e:
        logger.exception("LLM generation failed: %s", e)
        raise HTTPException(status_code=500, detail="LLM generation failed")

    # Persist chat to DB
    try:
        chat_entry = Chat(
            user_id=user.id,
            message=request.message,
            response=response_text,
            timestamp=datetime.utcnow()
        )
        db.add(chat_entry)
        db.commit()
        db.refresh(chat_entry)
        logger.info("Saved chat entry id=%s for user=%s", chat_entry.id, user.id)
    except Exception as e:
        logger.exception("Failed to save chat to DB: %s", e)
        raise HTTPException(status_code=500, detail="Failed to save chat")

    return ChatResponse(response=response_text, timestamp=chat_entry.timestamp)


@router.get("/history/{user_id}", response_model=ChatHistoryResponse)
async def get_history(user_id: str, db: Session = Depends(get_db)):
    """
    Retrieve the chat history for the user (ordered by newest first).
    """
    logger.info("GET /history user=%s", user_id)
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        logger.warning("User not found: %s", user_id)
        raise HTTPException(status_code=404, detail="User not found")

    chats = db.query(Chat).filter(Chat.user_id == user.id).order_by(Chat.timestamp.desc()).all()
    messages = [
        {"id": c.id, "message": c.message, "response": c.response, "timestamp": c.timestamp}
        for c in chats
    ]
    return ChatHistoryResponse(user_id=user_id, messages=messages)


@router.get("/index/info")
async def index_info():
    """
    Admin/debug endpoint: return Redis/RediSearch index info.
    """
    try:
        info = describe_index_stats()
        return {"index": INDEX_NAME, "info": info}
    except Exception as e:
        logger.exception("Failed to fetch index info: %s", e)
        raise HTTPException(status_code=500, detail="Failed to fetch index info")

"""
services/redis_service.py

Redis-backed vector store (replacement for Pinecone).

Key features:
- Stores chunks as RedisJSON documents under prefix `voicechat:user:{user_id}:chunk:{n}`.
- Uses a per-user atomic counter (INCR) to allocate chunk indices (grouped per user).
- Creates a RediSearch JSON index with NUMERIC fields for user_id/chat_id and a VECTOR field.
- Compatible with redis-py 6.x search API (Query(...).return_fields(...).dialect(2)).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional

import numpy as np

# OpenAI-compatible client for embeddings
from openai import OpenAI

# redis / RediSearch imports
from redis.commands.search.field import TextField, NumericField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query

# local config & helpers (must exist in your project)
from config import settings, get_redis_client
from logger_config import logger as project_logger

# module logger
logger = project_logger.getChild("redis_service")

# -------------------------
# Configuration (override via settings)
# -------------------------
INDEX_NAME = getattr(settings, "redis_vector_index_name", "voice_chat_idx")
PREFIX = getattr(settings, "redis_prefix", "voicechat:")
EMBED_DIM = int(getattr(settings, "embed_dim", 1024))
DISTANCE = getattr(settings, "distance_metric", "COSINE").upper()
EMBEDDING_MODEL = getattr(settings, "embedding_model", "intfloat/e5-large-v2")
BASE_URL = getattr(settings, "BASE_URL", None)
LLM_API_KEY = getattr(settings, "llm_api_key", None)

# OpenAI-compatible client instance (embedding generation)
openai = OpenAI(api_key=LLM_API_KEY, base_url=BASE_URL)


# -------------------------
# Embedding helper
# -------------------------
def get_embedding(text: str) -> List[float]:
    """
    Generate embedding using OpenAI-compatible client.
    Ensures returned vector length is EMBED_DIM (pad/truncate).
    """
    try:
        text = text or ""
        resp = openai.embeddings.create(input=text, model=EMBEDDING_MODEL, encoding_format="float")
        # defensive extraction
        try:
            embedding = resp.data[0].embedding
        except Exception:
            embedding = resp.get("data", [])[0].get("embedding") if isinstance(resp, dict) else None

        if not embedding:
            raise RuntimeError("Embedding API returned no vector")

        # pad/truncate
        if len(embedding) < EMBED_DIM:
            embedding = embedding + [0.0] * (EMBED_DIM - len(embedding))
        elif len(embedding) > EMBED_DIM:
            embedding = embedding[:EMBED_DIM]

        return list(map(float, embedding))
    except Exception as e:
        logger.exception("get_embedding failed: %s", e)
        # fallback: zeros
        return [0.0] * EMBED_DIM


# -------------------------
# Index helpers
# -------------------------
def ensure_index_exists(r):
    """
    Ensure RediSearch JSON index exists for the given prefix and vector field.
    Uses NUMERIC for user_id and chat_id (better numeric filtering) and TEXT for content.
    Idempotent: if index exists, returns silently.
    """
    try:
        r.ft(INDEX_NAME).info()
        logger.debug("Redis index '%s' already exists", INDEX_NAME)
        return
    except Exception:
        logger.info("Creating Redis index '%s' (prefix=%s, dim=%d, metric=%s)",
                    INDEX_NAME, PREFIX, EMBED_DIM, DISTANCE)

    fields = [
            NumericField("$.user_id", as_name="user_id"),
            NumericField("$.chat_id", as_name="chat_id"),
            TextField("$.text", as_name="text"),
            VectorField(
                "$.embedding",
                "HNSW",
                {
                    "TYPE": "FLOAT32",
                    "DIM": EMBED_DIM,
                    "DISTANCE_METRIC": DISTANCE,
                    "M": 16,
                    "EF_CONSTRUCTION": 200,
                },
                as_name="embedding",
            ),
        ]


    definition = IndexDefinition(prefix=[PREFIX], index_type=IndexType.JSON)
    # create_index will raise if index exists concurrently; that's fine in multi-instance setups
    r.ft(INDEX_NAME).create_index(fields, definition=definition)
    logger.info("Created Redis index '%s'", INDEX_NAME)


def describe_index_stats() -> dict:
    """
    Return RediSearch index info (dict).
    """
    r = get_redis_client()
    ensure_index_exists(r)
    info = r.ft(INDEX_NAME).info()
    print("info-----> ", info)
    # convert to plain dict if needed
    try:
        return dict(info)
    except Exception:
        return info


# -------------------------
# Retrieval
# -------------------------
def retrieve_context(query: str, user_id: int, top_k: int = 3) -> str:
    """
    Perform a KNN vector search restricted to the provided user_id.
    Returns joined text chunks (by similarity order) for that user.
    Compatible with redis-py 6.x Query API.
    """
    logger.info("retrieve_context user=%s query_len=%d top_k=%d", user_id, len(query or ""), top_k)
    try:
        r = get_redis_client()
    except Exception as e:
        logger.exception("Cannot obtain Redis client: %s", e)
        return "Error: cannot connect to Redis."

    try:
        ensure_index_exists(r)
    except Exception as e:
        # keep going, search may still work
        logger.warning("ensure_index_exists failed (continuing): %s", e)

    # get query embedding
    q_emb = get_embedding(query)
    vec = np.array(q_emb, dtype=np.float32)

    # numeric range filter syntax: @user_id:[<min> <max>]
    # we restrict to user_id exactly by giving same min/max
    q_str = f'(@user_id:[{int(user_id)} {int(user_id)}])=>[KNN {top_k} @embedding $vec_param AS score]'

    # Build Query object (return text/chat_id/created_at/chunk_index for richer context)
    query_obj = Query(q_str).return_fields("text", "chat_id", "created_at", "chunk_index").dialect(2)

    try:
        res = r.ft(INDEX_NAME).search(query_obj, query_params={"vec_param": vec.tobytes()})
    except Exception as e:
        logger.exception("Redis search failed: %s", e)
        return "Error retrieving context."

    docs = getattr(res, "docs", []) or []
    if not docs:
        logger.debug("No vector matches found for user=%s", user_id)
        return ""

    parts = []
    # Attempt to fetch stored JSON for each doc.id to be robust across redis-py variations
    for d in docs:
        try:
            doc_id = getattr(d, "id", None)
            if doc_id:
                stored = r.json().get(doc_id)
                if stored:
                    t = stored.get("text")
                    if t:
                        parts.append(t)
                        continue
            # fallback: check the doc attributes returned by Query.return_fields(...)
            # Some redis clients attach returned fields as attributes (e.g., d.text)
            text_field = getattr(d, "text", None) or getattr(d, "payload", {}).get("text", None) if hasattr(d, "payload") else None
            if text_field:
                parts.append(text_field)
        except Exception as e:
            logger.debug("Failed to parse search doc: %s", e)

    # join and return
    context = "\n".join(parts)
    logger.debug("retrieve_context returned %d parts (len=%d)", len(parts), len(context))
    return context


# -------------------------
# Indexing helpers
# -------------------------
def index_transcript(user_id: int, transcript: str, chat_id: Optional[int] = None):
    """
    Chunk transcript and store as JSON docs in Redis.
    Uses per-user counter key to group all chunks for same user under:
      voicechat:user:{user_id}:chunk:{n}
    Each JSON contains user_id, chat_id, chunk_index, text, created_at, embedding.
    """
    logger.info("index_transcript user=%s chat_id=%s len=%d", user_id, chat_id, len(transcript or ""))
    if not transcript:
        return

    r = get_redis_client()
    ensure_index_exists(r)

    # naive chunking ~1800 chars per chunk
    lines = [l.strip() for l in transcript.replace("\r", "\n").split("\n") if l.strip()]
    chunks = []
    cur = []
    for line in lines:
        if sum(len(x) for x in cur) + len(line) + 1 > 1800:
            chunks.append(" ".join(cur))
            cur = [line]
        else:
            cur.append(line)
    if cur:
        chunks.append(" ".join(cur))

    counter_key = f"{PREFIX}user:{user_id}:counter"

    for chunk_text in chunks:
        emb = get_embedding(chunk_text)
        try:
            chunk_index = int(r.incr(counter_key))
        except Exception:
            # fallback to timestamp-based index if INCR fails (rare)
            chunk_index = int(datetime.utcnow().timestamp() * 1000)

        key = f"{PREFIX}user:{user_id}:chunk:{chunk_index}"
        doc = {
            "user_id": int(user_id),
            "chat_id": int(chat_id) if chat_id is not None else None,
            "chunk_index": int(chunk_index),
            "text": chunk_text,
            "created_at": datetime.utcnow().isoformat(),
            "embedding": np.array(emb, dtype=np.float32).tolist(),
        }
        try:
            r.json().set(key, "$", doc)
        except Exception as e:
            logger.exception("Failed to store transcript chunk to Redis (key=%s): %s", key, e)

    logger.info("Indexed %d chunks for user=%s", len(chunks), user_id)


def index_document(file_path: str, document_id: int, user_id: int, db=None):
    """
    Index a PDF document (page-level chunks) into Redis JSON docs grouped by user counter.
    If db (SQLAlchemy session) is provided, tries to mark DB document as indexed similarly to previous behavior.
    """
    logger.info("index_document file=%s user=%s document_id=%s", file_path, user_id, document_id)
    try:
        import pdfplumber
    except Exception:
        logger.exception("pdfplumber not available; skipping document indexing")
        return

    r = get_redis_client()
    ensure_index_exists(r)

    page_chunks = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for p in pdf.pages:
                text = p.extract_text() or ""
                if text.strip():
                    page_chunks.append(text.strip())
    except Exception:
        logger.exception("Failed to extract PDF text")

    if not page_chunks:
        logger.warning("No text extracted from document: %s", file_path)
        return

    counter_key = f"{PREFIX}user:{user_id}:counter"
    for i, chunk_text in enumerate(page_chunks):
        emb = get_embedding(chunk_text)
        try:
            chunk_index = int(r.incr(counter_key))
        except Exception:
            chunk_index = int(datetime.utcnow().timestamp() * 1000)

        key = f"{PREFIX}user:{user_id}:chunk:{chunk_index}"
        doc = {
            "document_id": int(document_id),
            "user_id": int(user_id),
            "chunk_index": int(chunk_index),
            "text": chunk_text,
            "created_at": datetime.utcnow().isoformat(),
            "embedding": np.array(emb, dtype=np.float32).tolist(),
        }
        try:
            r.json().set(key, "$", doc)
        except Exception:
            logger.exception("Failed to store document chunk to Redis: %s", key)

    # optionally update DB 'indexed' flag if db session supplied
    if db is not None:
        try:
            from db.database import Document as DBDocument
            doc_obj = db.query(DBDocument).filter(DBDocument.id == document_id).first()
            if doc_obj:
                doc_obj.indexed = True
                doc_obj.indexed_at = datetime.utcnow()
                db.commit()
        except Exception:
            logger.exception("Failed to mark DB document as indexed for document_id=%s", document_id)


# -------------------------
# (Optional) Utility: small function to scan per-user chunks (not required)
# -------------------------
def list_user_chunks(user_id: int, limit: int = 100) -> List[str]:
    """
    Return list of keys for user's chunks (useful for debugging).
    """
    r = get_redis_client()
    pattern = f"{PREFIX}user:{user_id}:chunk:*"
    cursor = 0
    out = []
    while True:
        cursor, keys = r.scan(cursor=cursor, match=pattern, count=1000)
        for k in keys:
            out.append(k.decode() if isinstance(k, bytes) else k)
            if len(out) >= limit:
                return out
        if cursor == 0:
            break
    return out

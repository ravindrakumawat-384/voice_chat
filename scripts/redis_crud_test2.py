#!/usr/bin/env python3
"""Redis CRUD with RediSearch + JSON.

- Loads schema from db/user_message.yaml
- Ensures index exists
- Provides CRUD operations for user messages
- Supports vector similarity search (KNN)
"""

import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import yaml
from redis.commands.search.field import TextField, TagField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType



from config import get_redis_client

# --------------------------------------------------
# Logging
# --------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger("redis_crud")

# --------------------------------------------------
# Load YAML Config
# --------------------------------------------------
CONFIG_PATH = Path("db/user_message.yaml")
if not CONFIG_PATH.exists():
    logger.error("YAML config not found at %s", CONFIG_PATH)
    sys.exit(1)

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

INDEX_NAME = cfg["index"]["name"]
PREFIX = cfg["index"]["prefix"]

# Extract embedding dims + metric
for fdef in cfg["fields"]:
    if fdef["name"] == "embedding":
        EMBED_DIM = fdef["attrs"]["dims"]
        DISTANCE = fdef["attrs"]["distance_metric"]
        break
else:
    EMBED_DIM, DISTANCE = 128, "COSINE"

# --------------------------------------------------
# Index Setup
# --------------------------------------------------
def ensure_index_exists(r):
    """Create RediSearch index from YAML if it doesn't exist."""
    try:
        r.ft(INDEX_NAME).info()
        logger.info("Index %s already exists", INDEX_NAME)
        return
    except Exception:
        logger.info("Creating index %s...", INDEX_NAME)

    fields = []
    for fdef in cfg["fields"]:
        if fdef["type"] == "text":
            fields.append(TextField(fdef["name"]))
        elif fdef["type"] == "tag":
            fields.append(TagField(fdef["name"]))
        elif fdef["type"] == "vector":
            attrs = fdef["attrs"]
            fields.append(
                VectorField(
                    fdef["name"],
                    "HNSW",
                    {
                        "TYPE": attrs["dtype"].upper(),
                        "DIM": attrs["dims"],
                        "DISTANCE_METRIC": attrs["distance_metric"].upper(),
                        "M": attrs.get("M", 16),
                        "EF_CONSTRUCTION": attrs.get("ef_construction", 200),
                    },
                )
            )

    definition = IndexDefinition(prefix=[PREFIX], index_type=IndexType.JSON)
    r.ft(INDEX_NAME).create_index(fields, definition=definition)
    logger.info("Index %s created successfully", INDEX_NAME)

# --------------------------------------------------
# CRUD Operations
# --------------------------------------------------
def create_message(r, message: str, metadata: Dict[str, Any] = None) -> str:
    """Insert a JSON doc with embedding into Redis Search."""
    metadata = metadata or {}
    msg_id = f"{uuid.uuid4().hex[:8]}"
    emb = np.random.rand(EMBED_DIM).astype(np.float32).tobytes()  # replace with real embeddings

    doc = {
        "user_id": metadata.get("user_id", "unknown"),
        "message": message,
        "datetime": time.strftime("%Y-%m-%d %H:%M:%S"),
        "user_type": metadata.get("user_type", "user"),
        "metadata": metadata,
        "embedding": emb,
    }

    key = f"{PREFIX}{msg_id}"
    r.json().set(key, "$", doc)
    logger.info("Created message id=%s", msg_id)
    return msg_id


def fetch_by_id(r, msg_id: str) -> Dict[str, Any]:
    """Fetch a JSON doc by ID."""
    key = f"{PREFIX}{msg_id}"
    return r.json().get(key) or {}


def update_message_metadata(r, msg_id: str, new_metadata: Dict[str, Any]):
    """Update metadata for a JSON doc."""
    key = f"{PREFIX}{msg_id}"
    doc = r.json().get(key)
    if not doc:
        raise RuntimeError(f"Message {msg_id} not found")

    meta = doc.get("metadata", {})
    meta.update(new_metadata)
    doc["metadata"] = meta
    r.json().set(key, "$", doc)
    logger.info("Updated metadata for %s", msg_id)
    return meta


def delete_message(r, msg_id: str):
    """Delete a JSON doc by ID."""
    key = f"{PREFIX}{msg_id}"
    r.delete(key)
    logger.info("Deleted message %s", msg_id)


def query_similar(r, query_vec: np.ndarray, top_k: int = 3):
    """Vector similarity search using RediSearch KNN."""
    q = f'*=>[KNN {top_k} @embedding $vec_param AS score]'
    res = r.ft(INDEX_NAME).search(
        q,
        query_params={"vec_param": query_vec.tobytes()},
    )
    return [(doc.id, doc.score, doc.json) for doc in res.docs]

# --------------------------------------------------
# Demo Sequence
# --------------------------------------------------
def run_sequence():
    try:
        r = get_redis_client()
    except Exception as exc:
        logger.error("Could not connect to Redis: %s", exc)
        sys.exit(1)

    ensure_index_exists(r)

    # Create
    mid = create_message(r, "Hi there, how are you?", {"user_id": "123"})
    # Fetch
    doc = fetch_by_id(r, mid)
    logger.info("Fetched: %s", doc)

    # Update
    new_meta = update_message_metadata(r, mid, {"note": "patched"})
    logger.info("Updated metadata: %s", new_meta)

    # Query similar
    q_vec = np.random.rand(EMBED_DIM).astype(np.float32)
    results = query_similar(r, q_vec, top_k=2)
    logger.info("Similar results: %s", results)

    # Delete
    delete_message(r, mid)


if __name__ == "__main__":
    logger.info("Starting Redis CRUD test sequence")
    run_sequence()
    logger.info("Completed Redis CRUD test sequence")

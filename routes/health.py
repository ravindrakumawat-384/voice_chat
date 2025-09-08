# routes/health.py
from sqlalchemy import text
from typing import Dict, Any
from time import perf_counter
from logger_config import logger
from fastapi import APIRouter, Query
from services.redis_service import describe_index_stats, get_embedding, EMBEDDING_MODEL

router = APIRouter()


@router.get("/health")
async def health(
    include_db: bool = Query(True, description="Check database connectivity"),
    include_redis: bool = Query(True, description="Check Redis/RediSearch connectivity"),
    include_llm: bool = Query(True, description="Check LLM/embedding connectivity (may incur API call)")
) -> Dict[str, Any]:
    """
    Health endpoint that checks DB, Redis/RediSearch, and LLM connectivity.
    Query params:
      - include_db (bool): whether to check DB
      - include_redis (bool): whether to check Redis/RediSearch
      - include_llm (bool): whether to check LLM (embedding) — may call external API
    """
    logger.info("GET /health called (db=%s, redis=%s, llm=%s)", include_db, include_redis, include_llm)

    results: Dict[str, Any] = {
        "status": "ok",
        "components": {}
    }

    # -----------------------
    # 1) DB check
    # -----------------------
    if include_db:
        start = perf_counter()
        try:
            from db.database import engine  # import here to avoid startup side-effects
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            elapsed = perf_counter() - start
            results["components"]["database"] = {
                "ok": True,
                "message": "DB connected",
                "latency_s": round(elapsed, 3)
            }
        except Exception as e:
            elapsed = perf_counter() - start
            logger.exception("DB health check failed: %s", e)
            results["components"]["database"] = {
                "ok": False,
                "message": f"DB error: {e}",
                "latency_s": round(elapsed, 3)
            }
            results["status"] = "degraded"

    # -----------------------
    # 2) Redis / RediSearch check
    # -----------------------
    if include_redis:
        start = perf_counter()
        try:
            print("calling ------_> describe_index_stats------------_>>")
            stats = describe_index_stats()
            elapsed = perf_counter() - start

            # try to extract a useful total count or other summary fields if available
            total_vectors = None
            if isinstance(stats, dict):
                # RediSearch .info() output varies by redis-py version; try few common keys
                total_vectors = stats.get("total_vector_count") or stats.get("num_docs") or stats.get("num_records")
                # sometimes info nested under "index_stats" or similar
                if total_vectors is None:
                    for k in ("index_stats", "stats", "attributes"):
                        if isinstance(stats.get(k), dict):
                            total_vectors = stats.get(k).get("total_vector_count") or stats.get(k).get("num_docs")
                            if total_vectors:
                                break

            results["components"]["redis"] = {
                "ok": True,
                "message": "Redis/RediSearch reachable",
                "latency_s": round(elapsed, 3),
                "index_summary": stats,
                "total_vector_count": total_vectors,
            }
        except Exception as e:
            elapsed = perf_counter() - start
            logger.exception("Redis/RediSearch health check failed: %s", e)
            results["components"]["redis"] = {
                "ok": False,
                "message": f"Redis/RediSearch error: {e}",
                "latency_s": round(elapsed, 3)
            }
            results["status"] = "degraded"

    # -----------------------
    # 3) LLM / embedding check (optional)
    # -----------------------
    if include_llm:
        start = perf_counter()
        try:
            emb = get_embedding("health-check")
            elapsed = perf_counter() - start
            emb_len = len(emb) if emb is not None else None
            ok = bool(emb_len and emb_len > 0)
            results["components"]["llm"] = {
                "ok": ok,
                "message": "Embedding produced" if ok else "Embedding empty",
                "latency_s": round(elapsed, 3),
                "embedding_length": emb_len,
                "embedding_model": EMBEDDING_MODEL
            }
            if not ok:
                results["status"] = "degraded"
        except Exception as e:
            elapsed = perf_counter() - start
            logger.exception("LLM/embedding health check failed: %s", e)
            results["components"]["llm"] = {
                "ok": False,
                "message": f"LLM/embedding error: {e}",
                "latency_s": round(elapsed, 3)
            }
            results["status"] = "degraded"

    # Final status is carried in results["status"]
    return results

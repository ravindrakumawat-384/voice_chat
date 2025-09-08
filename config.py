# config.py
import os
from dotenv import load_dotenv
load_dotenv()
# Database settings
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/smartflow")
BASE_URL = "https://api.deepinfra.com/v1/openai"
# Pinecone settings
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "your-pinecone-api-key")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "voice-chat-index")
# LLM settings
LLM_API_KEY = os.getenv("DEEPINFRA_API_TOKEN", "your-openai-api-key")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
# STT settings (Voxtral)
STT_API_KEY = os.getenv("DEEPINFRA_API_TOKEN", "your-stt-api-key")
# WebSocket auth token (optional). If set, clients must send ?token=<value> or Authorization header.
WS_AUTH_TOKEN = os.getenv("WS_AUTH_TOKEN", "")
# Redis settings
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "") or None
# Application settings
APP_NAME = "SmartFlow Voice Chat"
DEBUG = os.getenv("DEBUG", "True").lower() == "true"
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

class Settings:
    pinecone_api_key: str = PINECONE_API_KEY
    pinecone_environment: str = PINECONE_ENVIRONMENT
    pinecone_index_name: str = PINECONE_INDEX_NAME
    DATABASE_URL: str = DATABASE_URL
    BASE_URL: str = BASE_URL
    llm_api_key: str = LLM_API_KEY
    llm_model: str = LLM_MODEL
    stt_api_key: str = STT_API_KEY
    ws_auth_token: str = WS_AUTH_TOKEN
    app_name: str = APP_NAME
    debug: bool = DEBUG 
    assets_dir: str = ASSETS_DIR
    # Redis
    redis_host: str = REDIS_HOST
    redis_port: int = REDIS_PORT
    redis_password: str | None = REDIS_PASSWORD

settings = Settings()


def get_redis_client(connect_timeout: int = 5):
    """Return a connected redis.Redis client or raise an informative error.

    Returns None if the redis package is not installed or connection fails.
    """
    try:
        import redis
    except Exception:
        raise RuntimeError("redis package not installed; install with pip install redis")

    try:
        client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            socket_connect_timeout=connect_timeout,
            decode_responses=True,
        )
        client.ping()
        return client
    except redis.AuthenticationError:
        raise RuntimeError("Redis authentication failed; check REDIS_PASSWORD")
    except redis.ConnectionError:
        raise RuntimeError("Could not connect to Redis; is the server running?")
    except Exception as exc:
        raise RuntimeError(f"Unexpected Redis error: {exc}")
    
get_redis = get_redis_client()

# services/generate_embeddings.py
import logging
from openai import OpenAI
from config import settings

logger = logging.getLogger(__name__)
BASE_URL = settings.BASE_URL
EMBEDDING_MODEL = "intfloat/e5-large-v2"  # or "intfloat/multilingual-e5-large"


# Create an OpenAI client with your deepinfra token and endpoint
openai = OpenAI(
    api_key=settings.llm_api_key,
    base_url=BASE_URL,
)


def get_embedding(text: str) -> list:
    """
    Generate an embedding for the given text using DeepInfra via OpenAI-compatible client.
    NOTE: Don't pass 'dimensions' — the model determines the vector size.
    """
    try:
        text = text or ""
        logger.info("Generating embedding for text of length %d", len(text))
        embeddings = openai.embeddings.create(
            input=text,
            model=EMBEDDING_MODEL,
            encoding_format="float",
        )
        # robust extraction: some SDKs return dict or object
        try:
            embedding = embeddings.data[0].embedding
        except Exception:
            # try alternative shapes
            if isinstance(embeddings, dict):
                embedding = embeddings.get("data", [])[0].get("embedding")
            else:
                embedding = getattr(embeddings.data[0], "embedding", None)

        if not embedding:
            raise ValueError("Embedding returned empty from the API")

        logger.info("Successfully generated embedding of length %d", len(embedding))
        return list(embedding)
    except Exception as e:
        logger.exception("Error generating embedding: %s", e)
        # Return a zero vector fallback matching your index dimension (1024)
        return [0.0] * 1024


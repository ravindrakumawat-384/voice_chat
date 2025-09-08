# services/llm.py

from openai import OpenAI
from config import settings
import ollama
from typing import Optional
from logger_config import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
import logging

# Initialize OpenAI-compatible client
openai = OpenAI(
    api_key=settings.llm_api_key,
    base_url=settings.BASE_URL,
)

# Default model names
DEFAULT_LLM_MODEL = "openai/gpt-oss-120b"
DEFAULT_OLLAMA_MODEL = "qwen3:1.7b"

# Default instruction
# DEFAULT_INSTRUCTION = "You are a helpful assistant. Use the context to answer concisely."
DEFAULT_INSTRUCTION =  "You are a helpful assistant. Use the given context strictly to answer. Store the context in the vector database." \
"Do not provide any suggestions, assumptions, or information. that is not explicitly present in the context."

# ✅ Define retry decorator using Tenacity
llm_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
    before_sleep=before_sleep_log(logger, logging.WARNING),
)


def generate_response(user_message: str, context: str, instruction: Optional[str] = None) -> str:

    """
    Attempt to generate a response using Ollama first, fall back to DeepInfra if it fails.
    """
    logger.info("Generating LLM response - llm.py")
    if instruction is None:
        instruction = DEFAULT_INSTRUCTION

    try:
        logger.info("Using Ollama for LLM response generation")
        print("\n\n\nContext:", context,"\n")
        return generate_response_ollama(
            instruction=instruction,
            user_input=user_message,
            context=context
        )
    except Exception as e:
        logger.warning("Ollama failed, falling back to DeepInfra: %s", e)

        try:
            return generate_response_deepinfra(
                instruction=instruction,
                user_message=user_message,
                context=context
            )
        except Exception as e2:
            logger.error("DeepInfra also failed: %s", e2)
            return "Sorry, I'm having trouble generating a response right now."


@llm_retry
def generate_response_ollama(instruction: str, user_input: str, context: Optional[str] = None, think: bool = False) -> str:
    """
    Generate response using Ollama (e.g., Qwen3 1.7B model).
    Retries up to 3 times on failure with exponential backoff.
    """
    messages = []
    if len(context)==0:
        logger.info("Context is empty")
    else:
        logger.info("Context is not empty")
    # System-level instruction
    if instruction:
        messages.append({"role": "system", "content": instruction})

    # User-level context and input
    if context:
        messages.append({"role": "assistant", "content": context})
    messages.append({"role": "user", "content": user_input})

    response = ollama.chat(
        model=DEFAULT_OLLAMA_MODEL,
        messages=messages,
        think=think,
    )
    return response["message"]["content"]


@llm_retry
def generate_response_deepinfra(instruction: str, user_message: str, context: Optional[str] = None, think: bool = False) -> str:
    """
    Generate response using DeepInfra (OpenAI-compatible) API.
    Retries up to 3 times on failure with exponential backoff.
    """
    prompt = f"Context:\n{context}\n\nUser: {user_message}\nAssistant:"
    logger.debug("Prepared prompt for DeepInfra (len=%d)", len(prompt))

    completion = openai.chat.completions.create(
        model=DEFAULT_LLM_MODEL,
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt},
        ],
        max_tokens=None,
    )

    # Extract response content robustly
    try:
        result = completion.choices[0].message.content
    except Exception:
        # Fallback for alternative shapes
        choices = getattr(completion, "choices", [])
        if choices:
            result = choices[0].get("message", {}).get("content") or choices[0].get("text", "")
        else:
            result = str(completion)

    result = result or "Sorry, I couldn't generate a response."
    logger.info("LLM response generated (len=%d)", len(result))
    return result

import os
import itertools
import time
import logging
from typing import List, Optional, Type, Any
from pydantic import BaseModel

from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage
from groq import RateLimitError

logger = logging.getLogger("gpa_agent.api_pool")


def load_groq_keys() -> list[str]:
    """Loads Groq API keys from environment variables."""
    raw_keys = os.getenv("GROQ_API_KEYS", "")
    keys = [k.strip() for k in raw_keys.split(",") if k.strip()]

    if not keys:
        single_key = os.getenv("GROQ_API_KEY")
        if single_key:
            keys.append(single_key.strip())

    if not keys:
        raise ValueError(
            "No Groq API keys found. Set GROQ_API_KEYS in environment variables."
        )

    return keys


# Global round-robin key iterator
_GROQ_KEYS = load_groq_keys()
_KEY_CYCLE = itertools.cycle(_GROQ_KEYS)


def get_next_groq_key() -> str:
    """Rotates to and returns the next API key in sequence."""
    key = next(_KEY_CYCLE)
    logger.info(f"[API Pool] Key selected: {key[:8]}...")
    return key


def invoke_groq_with_retry(
    messages: List[BaseMessage],
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0,
    structured_output_schema: Optional[Type[BaseModel]] = None,
    max_retries: Optional[int] = None,
) -> Any:
    """
    Invokes ChatGroq with automatic key rotation and retry on RateLimitError (HTTP 429).
    """
    # Default retries: cycle through the entire pool twice before throwing an error
    if max_retries is None:
        max_retries = len(_GROQ_KEYS) * 2

    attempts = 0

    while attempts < max_retries:
        api_key = get_next_groq_key()
        try:
            llm = ChatGroq(
                model=model,
                temperature=temperature,
                groq_api_key=api_key,
            )

            if structured_output_schema:
                runnable = llm.with_structured_output(structured_output_schema)
                return runnable.invoke(messages)

            return llm.invoke(messages)

        except RateLimitError as e:
            attempts += 1
            logger.warning(
                f"[API Pool] Key {api_key[:8]}... hit RateLimitError (429). "
                f"Attempt {attempts}/{max_retries}. Retrying with next key in pool..."
            )
            time.sleep(0.5)  # Brief pause before switching keys

        except Exception as e:
            # Fallback check for 429 string errors in raw exception wrappers
            if "429" in str(e) or "rate limit" in str(e).lower():
                attempts += 1
                logger.warning(
                    f"[API Pool] Key {api_key[:8]}... hit rate limit. "
                    f"Attempt {attempts}/{max_retries}. Retrying..."
                )
                time.sleep(0.5)
            else:
                # Immediately raise non-rate-limit errors (e.g., auth failure, bad prompt)
                raise e

    raise RuntimeError(
        f"All Groq API keys in pool failed after {max_retries} retry attempts due to rate limits."
    )

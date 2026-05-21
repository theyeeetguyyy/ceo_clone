"""
Groq API Key Pool — auto-rotates keys on rate-limit (429) or quota errors.
Keys are loaded from GROQ_API_KEYS (comma-separated) in .env.
Falls back gracefully, logs every rotation event.
"""

import asyncio
import os
import time
from typing import List, Optional
from dotenv import load_dotenv
from groq import Groq, AsyncGroq, RateLimitError, APIStatusError
from backend.utils.logger import get_logger

load_dotenv()
log = get_logger(__name__)


class GroqKeyPool:
    """
    Thread-safe, async-compatible Groq API key pool.
    Supports multiple keys with automatic rotation on 429 / rate-limit errors.
    Uses exponential backoff when all keys are exhausted.
    """

    def __init__(self):
        raw = os.getenv("GROQ_API_KEYS", os.getenv("GROQ_API_KEY", ""))
        keys = [k.strip() for k in raw.split(",") if k.strip()]
        if not keys:
            raise RuntimeError(
                "No Groq API keys found. Set GROQ_API_KEYS=key1,key2,... in .env"
            )
        self._keys: List[str] = keys
        self._current_idx: int = 0
        self._lock = asyncio.Lock()
        # cooldown tracking: key → epoch-second when it becomes available again
        self._cooldown: dict[str, float] = {}
        log.info(f"GroqKeyPool initialised with {len(self._keys)} key(s).")

    # ─── Internal helpers ─────────────────────────────────────────────────────

    def _next_available_key(self) -> Optional[str]:
        """Return the next non-cooled-down key, rotating round-robin."""
        now = time.monotonic()
        for offset in range(len(self._keys)):
            idx = (self._current_idx + offset) % len(self._keys)
            key = self._keys[idx]
            if self._cooldown.get(key, 0) <= now:
                self._current_idx = (idx + 1) % len(self._keys)
                return key
        return None  # all keys in cooldown

    def _put_key_on_cooldown(self, key: str, seconds: float = 60.0):
        self._cooldown[key] = time.monotonic() + seconds
        log.warning(f"Key ...{key[-6:]} put on cooldown for {seconds:.0f}s")

    # ─── Public sync client (for use in non-async contexts) ───────────────────

    def get_client(self, model: str = "llama-3.1-8b-instant") -> Groq:
        key = self._next_available_key() or self._keys[0]
        return Groq(api_key=key)

    # ─── Async chat completion with auto-rotation ─────────────────────────────

    async def chat(
        self,
        messages: list,
        model: str = "llama-3.1-8b-instant",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        max_retries: int = 5,
    ) -> str:
        """
        Send a chat completion request, rotating keys on rate-limit errors.
        Returns the response content string.
        """
        attempt = 0
        last_error = None

        while attempt < max_retries:
            async with self._lock:
                key = self._next_available_key()

            if key is None:
                wait = 15 * (2 ** min(attempt, 4))
                log.warning(
                    f"All keys in cooldown. Waiting {wait}s... (attempt {attempt+1})"
                )
                await asyncio.sleep(wait)
                attempt += 1
                continue

            try:
                client = AsyncGroq(api_key=key, timeout=30.0)
                log.debug(f"Using key ...{key[-6:]} | model={model} | attempt={attempt+1}")
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                content = response.choices[0].message.content
                log.debug(f"Response received ({len(content)} chars)")
                return content

            except RateLimitError as e:
                log.warning(f"Rate limit on key ...{key[-6:]}: {e}")
                self._put_key_on_cooldown(key, seconds=60.0)
                last_error = e
                attempt += 1

            except APIStatusError as e:
                if e.status_code == 429:
                    log.warning(f"429 on key ...{key[-6:]}: {e}")
                    self._put_key_on_cooldown(key, seconds=60.0)
                    last_error = e
                    attempt += 1
                else:
                    log.error(f"Groq API error (non-429): {e}")
                    raise

            except Exception as e:
                log.error(f"Unexpected error calling Groq: {e}")
                raise

        raise RuntimeError(
            f"All {max_retries} retry attempts exhausted. Last error: {last_error}"
        )

    async def stream_chat(
        self,
        messages: list,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.3,
        max_tokens: int = 1024,
        _retry_count: int = 0,
    ):
        """
        Async generator that yields tokens from a streaming Groq completion.
        Rotates key on rate-limit errors. Max 3 retries to prevent infinite recursion.
        """
        if _retry_count >= 3:
            log.error("stream_chat: max retries exhausted")
            yield "I apologise — I'm experiencing a rate limit. Please try again shortly."
            return

        key = self._next_available_key() or self._keys[0]
        try:
            client = AsyncGroq(api_key=key)
            log.debug(f"Streaming with key ...{key[-6:]} | model={model}")
            async with client.chat.completions.stream(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            ) as stream:
                async for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta
        except RateLimitError:
            log.warning(f"Rate limit during stream on key ...{key[-6:]}. Rotating (retry={_retry_count+1}).")
            self._put_key_on_cooldown(key, seconds=60.0)
            async for token in self.stream_chat(messages, model, temperature, max_tokens, _retry_count + 1):
                yield token


# ─── Singleton ────────────────────────────────────────────────────────────────
_pool: Optional[GroqKeyPool] = None


def get_pool() -> GroqKeyPool:
    global _pool
    if _pool is None:
        _pool = GroqKeyPool()
    return _pool

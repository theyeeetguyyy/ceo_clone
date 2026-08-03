"""
Gemini API Client — google-genai SDK wrapper
============================================
Drop-in replacement for GroqKeyPool.
Uses gemini-2.5-flash (latest, highest quality) for generation
and gemini-2.0-flash for fast utility tasks (routing, grading, etc.).

API key loaded from GEMINI_API_KEY in .env.
"""

import asyncio
import os
from typing import AsyncGenerator, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

from backend.utils.logger import get_logger

load_dotenv(override=True)  # override=True: .env wins over system env vars
log = get_logger(__name__)


class GeminiClient:
    """
    Async Gemini client built on the official google-genai SDK.
    Provides the same .chat() and .stream_chat() interface as the old GroqKeyPool
    so all caller code (nodes.py etc.) works without changes.
    """

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "No Gemini API key found. Set GEMINI_API_KEY=your_key in .env"
            )
        self._client = genai.Client(api_key=api_key)
        log.info("GeminiClient initialised (google-genai SDK).")

    # ─── Async chat completion ────────────────────────────────────────────────

    async def chat(
        self,
        messages: list,
        model: str = "gemini-3.5-flash-lite",     # overridden by callers
        temperature: float = 0.0,
        max_tokens: int = 1024,
        max_retries: int = 4,
    ) -> str:
        """
        Send a chat completion request to Gemini.
        Accepts OpenAI-style message list: [{"role": "system"|"user"|"assistant", "content": "..."}]
        Returns the response content string.
        """
        # Extract system instruction (Gemini separates it from history)
        system_instruction = None
        history = []
        user_content = ""

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_instruction = content
            elif role == "user":
                user_content = content   # last user message is the prompt
                # Earlier user/assistant turns go into history
                history.append(
                    types.Content(
                        role="user",
                        parts=[types.Part(text=content)]
                    )
                )
            elif role == "assistant":
                history.append(
                    types.Content(
                        role="model",
                        parts=[types.Part(text=content)]
                    )
                )

        # Build GenerateContentConfig
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system_instruction,
        )

        attempt = 0
        last_error = None
        while attempt < max_retries:
            try:
                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self._client.models.generate_content(
                        model=model,
                        contents=user_content,
                        config=config,
                    )
                )
                text = response.text
                log.debug(f"Gemini response ({len(text)} chars) | model={model}")
                return text

            except Exception as e:
                err_str = str(e).lower()
                if "quota" in err_str or "rate" in err_str or "429" in err_str:
                    wait = 10 * (2 ** min(attempt, 3))
                    log.warning(f"Gemini rate limit on attempt {attempt+1}. Waiting {wait}s... | {e}")
                    await asyncio.sleep(wait)
                    last_error = e
                    attempt += 1
                else:
                    log.error(f"Gemini API error: {e}")
                    raise

        raise RuntimeError(
            f"Gemini: all {max_retries} attempts failed. Last error: {last_error}"
        )

    # ─── Async streaming (token-by-token) ────────────────────────────────────

    async def stream_chat(
        self,
        messages: list,
        model: str = "gemini-2.0-flash",
        temperature: float = 0.3,
        max_tokens: int = 8192,
        _retry_count: int = 0,
    ) -> AsyncGenerator[str, None]:
        """
        Async generator that yields text chunks from a streaming Gemini response.
        """
        system_instruction = None
        user_content = ""

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_instruction = content
            elif role == "user":
                user_content = content

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system_instruction,
        )

        try:
            loop = asyncio.get_running_loop()

            # Use non-streaming in executor as google-genai sync streaming
            # is easier to handle than async adapter
            response = await loop.run_in_executor(
                None,
                lambda: self._client.models.generate_content(
                    model=model,
                    contents=user_content,
                    config=config,
                )
            )
            yield response.text

        except Exception as e:
            err_str = str(e).lower()
            if ("quota" in err_str or "rate" in err_str or "429" in err_str) and _retry_count < 3:
                log.warning(f"Gemini stream rate limit. Retry {_retry_count+1}... | {e}")
                await asyncio.sleep(15)
                async for token in self.stream_chat(messages, model, temperature, max_tokens, _retry_count + 1):
                    yield token
            else:
                log.error(f"Gemini stream error: {e}")
                yield "I apologise — I'm experiencing a technical issue. Please try again shortly."


# ─── Singleton ───────────────────────────────────────────────────────────────
_client: Optional[GeminiClient] = None


def get_pool() -> GeminiClient:
    """
    Returns the singleton GeminiClient.
    Named 'get_pool' for backward compatibility with all callers that do:
        from backend.utils.groq_rotator import get_pool
    Just change the import to: from backend.utils.gemini_client import get_pool
    """
    global _client
    if _client is None:
        _client = GeminiClient()
    return _client

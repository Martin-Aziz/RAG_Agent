"""
llm/client.py — Async LLM client supporting Ollama and OpenAI-compatible APIs.

Provides both streaming and non-streaming inference modes with automatic
retry logic and fallback from local Ollama to cloud OpenAI.

Design decisions:
- Ollama as primary: runs locally, no API keys needed, no data leaves the machine
- OpenAI as fallback: activates when OPENAI_API_KEY env var is set
- Async streaming via httpx for non-blocking token delivery
- Exponential backoff with 3 retries for transient failures
- Model switching via configuration for easy experimentation
"""

from __future__ import annotations

import os
from typing import AsyncGenerator, Optional

import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

# Default Ollama configuration
DEFAULT_OLLAMA_HOST = "http://ollama:11434"
DEFAULT_MODEL = "llama3.1:8b"


class LLMClient:
    """Async LLM client with Ollama primary and OpenAI fallback.

    Supports both streaming (token-by-token) and single-shot modes.
    Automatically retries on transient failures with exponential backoff.
    """

    def __init__(
        self,
        ollama_host: Optional[str] = None,
        model: Optional[str] = None,
        openai_api_key: Optional[str] = None,
    ):
        """Initialize the LLM client.

        Args:
            ollama_host: Ollama API base URL (default: from OLLAMA_HOST env var).
            model: Model name for Ollama (default: from OLLAMA_MODEL env var).
            openai_api_key: OpenAI API key for fallback (default: from env var).
        """
        self.ollama_host = ollama_host or os.getenv(
            "OLLAMA_HOST", DEFAULT_OLLAMA_HOST
        )
        self.model = model or os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

        # HTTP client with generous timeouts for LLM inference
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=10.0,
                read=120.0,    # LLM generation can take a while
                write=10.0,
                pool=10.0,
            )
        )

        self._use_openai = bool(self.openai_api_key)

        logger.info(
            f"LLMClient initialized: model={self.model}, "
            f"ollama={self.ollama_host}, "
            f"openai_fallback={'enabled' if self._use_openai else 'disabled'}"
        )

    async def stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream LLM response token by token.

        Tries Ollama first; falls back to OpenAI on failure.

        Args:
            prompt: The user prompt to send to the LLM.
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum response tokens.
            system_prompt: Optional system prompt for instruction-following.

        Yields:
            Individual text tokens as they're generated.
        """
        try:
            async for token in self._stream_ollama(
                prompt, temperature, max_tokens, system_prompt
            ):
                yield token
        except Exception as e:
            logger.warning(f"Ollama streaming failed: {e}")

            if self._use_openai:
                logger.info("Falling back to OpenAI API")
                async for token in self._stream_openai(
                    prompt, temperature, max_tokens, system_prompt
                ):
                    yield token
            else:
                logger.error("No LLM fallback available")
                yield f"[Error: LLM unavailable — {str(e)[:100]}]"

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate a complete LLM response (non-streaming).

        Args:
            prompt: The user prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum response tokens.
            system_prompt: Optional system prompt.

        Returns:
            Complete generated text.
        """
        tokens = []
        async for token in self.stream(
            prompt, temperature, max_tokens, system_prompt
        ):
            tokens.append(token)
        return "".join(tokens)

    async def _stream_ollama(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream response from Ollama API.

        Uses the /api/chat endpoint with streaming enabled.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        url = f"{self.ollama_host}/api/chat"
        logger.debug(f"Ollama request: model={self.model}, prompt_len={len(prompt)}")

        async with self._client.stream("POST", url, json=payload) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line.strip():
                    continue

                try:
                    import json
                    data = json.loads(line)

                    # Ollama streams JSON objects, each with a "message" field
                    if "message" in data and "content" in data["message"]:
                        token = data["message"]["content"]
                        if token:
                            yield token

                    # Check for stream end
                    if data.get("done", False):
                        break

                except (json.JSONDecodeError, KeyError) as e:
                    logger.trace(f"Skipping malformed Ollama response line: {e}")
                    continue

    async def _stream_openai(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream response from OpenAI-compatible API.

        Uses the /v1/chat/completions endpoint with streaming enabled.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": "gpt-4o-mini",
            "messages": messages,
            "stream": True,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json",
        }

        url = "https://api.openai.com/v1/chat/completions"

        async with self._client.stream(
            "POST", url, json=payload, headers=headers
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue

                data_str = line[6:]  # Remove "data: " prefix

                if data_str.strip() == "[DONE]":
                    break

                try:
                    import json
                    data = json.loads(data_str)
                    delta = data["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def _generate_with_retry(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Generate response with automatic retry on failure.

        Uses tenacity for exponential backoff: 1s, 2s, 4s between retries.
        """
        return await self.generate(prompt, temperature, max_tokens)

    async def close(self):
        """Close the HTTP client and release resources."""
        await self._client.aclose()
        logger.info("LLM client closed")

"""
llm/streaming.py — Token-by-token streaming handler for SSE delivery.

Manages the streaming state for a single chat response, buffering tokens
and handling stream lifecycle events (start, token, graph_update, done, error).
"""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator, Optional

from loguru import logger

from src.models.schemas import ChatToken, TokenType


class StreamingHandler:
    """Manages streaming state for a single chat response.

    Buffers tokens and provides lifecycle methods for the SSE pipeline.
    Handles cancellation and error states gracefully.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._buffer: list[str] = []
        self._is_streaming = False
        self._is_cancelled = False
        self._total_tokens = 0

    @property
    def is_streaming(self) -> bool:
        return self._is_streaming

    @property
    def full_response(self) -> str:
        """Get the complete accumulated response text."""
        return "".join(self._buffer)

    async def stream_tokens(
        self,
        token_generator: AsyncGenerator[str, None],
        include_subgraph: bool = True,
    ) -> AsyncGenerator[ChatToken, None]:
        """Wrap a raw token generator with lifecycle management.

        Yields ChatToken objects with proper type classification.
        Handles start/done/error events automatically.

        Args:
            token_generator: Async generator yielding raw text tokens.
            include_subgraph: Whether to yield a GRAPH_UPDATE at the end.

        Yields:
            ChatToken objects for SSE serialization.
        """
        self._is_streaming = True
        logger.info(f"Stream started for session {self.session_id}")

        try:
            async for token in token_generator:
                if self._is_cancelled:
                    logger.info(f"Stream cancelled for session {self.session_id}")
                    break

                self._buffer.append(token)
                self._total_tokens += 1

                # Classify token type based on content
                token_type = TokenType.TEXT

                # Check for citation markers like [NODE:abc123]
                if "[NODE:" in token:
                    token_type = TokenType.CITATION

                yield ChatToken(
                    text=token,
                    type=token_type,
                )

            # Yield done sentinel
            yield ChatToken(
                text="",
                type=TokenType.DONE,
            )

        except asyncio.CancelledError:
            logger.info(f"Stream cancelled via asyncio for {self.session_id}")
            yield ChatToken(text="", type=TokenType.DONE)

        except Exception as e:
            logger.error(f"Stream error for {self.session_id}: {e}")
            yield ChatToken(
                text=f"Error: {str(e)[:200]}",
                type=TokenType.ERROR,
            )

        finally:
            self._is_streaming = False
            logger.info(
                f"Stream ended for {self.session_id}: "
                f"{self._total_tokens} tokens, "
                f"{len(self.full_response)} chars"
            )

    def cancel(self):
        """Cancel the current stream."""
        self._is_cancelled = True
        logger.info(f"Stream cancel requested for {self.session_id}")

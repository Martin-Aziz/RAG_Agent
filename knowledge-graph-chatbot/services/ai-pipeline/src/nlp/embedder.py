"""
nlp/embedder.py — Sentence embedding generation using sentence-transformers.

Wraps the sentence-transformers library for generating 384-dimensional
embeddings using the all-MiniLM-L6-v2 model.

Design decisions:
- Lazy model loading to avoid loading the model if embedding is not needed
- Batch embedding support for efficient document processing
- Normalization to unit vectors for cosine similarity compatibility
- CPU/GPU auto-detection with explicit device override support
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
from loguru import logger

from src.models.schemas import EmbeddingResult


# Default model name — lightweight, fast, 384-dimensional output
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DIMENSION = 384


class SentenceEmbedder:
    """Generates sentence embeddings using sentence-transformers.

    Embeddings are 384-dimensional float vectors suitable for
    cosine similarity search in the HNSW index.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """Initialize the embedder.

        Args:
            model_name: HuggingFace model name or path. Defaults to MiniLM-L6-v2.
            device: "cuda", "cpu", or None for auto-detection.
        """
        self.model_name = model_name or os.getenv(
            "EMBEDDING_MODEL", DEFAULT_MODEL
        )
        self._device = device
        self._model = None  # Lazy loading
        self.dimension = DEFAULT_DIMENSION

        logger.info(f"SentenceEmbedder configured: model={self.model_name}")

    def _load_model(self):
        """Lazily load the sentence-transformer model on first use.

        This avoids loading the model during import, which speeds up
        service startup and allows configuration before first use.
        """
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer

            device = self._device
            if device is None:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info(
                f"Loading embedding model: {self.model_name} on {device}"
            )
            self._model = SentenceTransformer(self.model_name, device=device)

            # Verify output dimension matches expected
            test_embedding = self._model.encode(["test"])
            self.dimension = test_embedding.shape[1]
            logger.info(
                f"Embedding model loaded: dimension={self.dimension}, "
                f"device={device}"
            )
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise RuntimeError(f"Embedding model load failed: {e}") from e

    async def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text string.

        Args:
            text: Input text to embed.

        Returns:
            384-dimensional float vector.
        """
        self._load_model()

        # sentence-transformers is synchronous; run in threadpool would be
        # better for production, but this works for the pipeline.
        embedding = self._model.encode(
            [text],
            normalize_embeddings=True,  # L2 normalize for cosine similarity
            show_progress_bar=False,
        )[0]

        return embedding.tolist()

    async def embed_batch(
        self, texts: List[str], batch_size: int = 32
    ) -> List[EmbeddingResult]:
        """Generate embeddings for multiple texts in batches.

        Args:
            texts: List of texts to embed.
            batch_size: Number of texts per batch for GPU efficiency.

        Returns:
            List of EmbeddingResult with vectors and metadata.
        """
        self._load_model()

        if not texts:
            return []

        logger.info(f"Embedding {len(texts)} texts in batches of {batch_size}")

        # Batch encode for efficiency (important for GPU utilization)
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
        )

        results = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            results.append(EmbeddingResult(
                vector=embedding.tolist(),
                dimensions=self.dimension,
                source_text=text[:200],  # Truncate for metadata (not full text)
            ))

        logger.info(f"Generated {len(results)} embeddings")
        return results

    async def similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity between two texts.

        Convenience method for quick pairwise comparison.
        """
        self._load_model()

        embeddings = self._model.encode(
            [text_a, text_b],
            normalize_embeddings=True,
        )

        # Cosine similarity of normalized vectors = dot product
        similarity = float(np.dot(embeddings[0], embeddings[1]))
        return similarity

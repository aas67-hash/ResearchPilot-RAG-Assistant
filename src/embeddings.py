"""Embedding generation module using Sentence Transformers.

Generates dense vector representations for document chunks and user queries
using local open-source sentence transformer models (default: all-MiniLM-L6-v2).
"""

from typing import List, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"


class EmbeddingGenerator:
    """Handles dense vector embedding generation using Sentence Transformers."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: Optional[str] = None,
    ) -> None:
        """Initialize embedding generator with chosen local model.

        Args:
            model_name: HuggingFace model identifier (default: all-MiniLM-L6-v2).
            device: Computing device ('cpu', 'cuda', etc.). If None, auto-selected.
        """
        self.model_name = model_name
        self.device = device
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazily load the SentenceTransformer model on first access."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimensionality (e.g. 384 for all-MiniLM-L6-v2)."""
        if hasattr(self.model, "get_embedding_dimension"):
            dim = self.model.get_embedding_dimension()
        else:
            dim = self.model.get_sentence_embedding_dimension()
        return int(dim) if dim is not None else 384

    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        """Generate dense vector embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.
            batch_size: Number of texts to process per batch.
            normalize: If True, vectors are normalized to unit length L2 norm.

        Returns:
            np.ndarray of shape (len(texts), dimension) with float32 values.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def embed_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """Generate a dense vector embedding for a single search query.

        Args:
            query: User's search query string.
            normalize: If True, vector is normalized to unit length L2 norm.

        Returns:
            1D np.ndarray of shape (dimension,) with float32 values.
        """
        query_text = query.strip()
        if not query_text:
            return np.zeros(self.dimension, dtype=np.float32)

        embeddings = self.embed_texts([query_text], normalize=normalize)
        return embeddings[0]

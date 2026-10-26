"""Hybrid retrieval module combining dense semantic search and sparse BM25 keyword search.

Provides three retrieval modes behind a unified interface:
- 'keyword': BM25Okapi lexical retrieval
- 'semantic': Dense vector similarity retrieval via Qdrant & SentenceTransformers
- 'hybrid': Reciprocal Rank Fusion (RRF) combining dense and sparse results

All modes return identical result dictionaries containing:
`chunk_id`, `source`, `page_number`, `text`, `score`, and `rank`.
"""

import re
from typing import Any, Dict, List, Optional, Union
from rank_bm25 import BM25Okapi

from .embeddings import EmbeddingGenerator
from .vector_store import VectorStore

DEFAULT_RRF_K: int = 60
DEFAULT_TOP_K: int = 5


def tokenize_text(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric words for BM25.

    Args:
        text: Input string.

    Returns:
        List of lowercase word tokens.
    """
    if not text:
        return []
    return re.findall(r"\w+", text.lower())


class HybridRetriever:
    """Orchestrates dense vector search, BM25 keyword search, and RRF hybrid fusion."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_generator: EmbeddingGenerator,
        default_mode: str = "hybrid",
        rrf_k: int = DEFAULT_RRF_K,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ) -> None:
        """Initialize hybrid retriever.

        Args:
            vector_store: VectorStore instance managing local Qdrant collections.
            embedding_generator: EmbeddingGenerator instance for query embeddings.
            default_mode: Default search mode ('keyword', 'semantic', or 'hybrid').
            rrf_k: Reciprocal Rank Fusion smoothing parameter (default: 60).
            dense_weight: Multiplier weight for dense retrieval in RRF (default: 1.0).
            bm25_weight: Multiplier weight for BM25 retrieval in RRF (default: 1.0).
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.default_mode = default_mode.lower()
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

        # BM25 state
        self._bm25_model: Optional[BM25Okapi] = None
        self._corpus_chunks: List[Dict[str, Any]] = []

    def build_bm25_index(self, chunks: Optional[List[Dict[str, Any]]] = None) -> int:
        """Build or refresh the BM25 index over document chunks.

        Args:
            chunks: Optional list of chunk dictionaries. If None, chunks are
                    fetched automatically from the underlying VectorStore.

        Returns:
            Number of indexed chunks.
        """
        if chunks is not None:
            self._corpus_chunks = list(chunks)
        else:
            self._corpus_chunks = self.vector_store.get_all_chunks()

        if not self._corpus_chunks:
            self._bm25_model = None
            return 0

        tokenized_corpus = [tokenize_text(c.get("text", "")) for c in self._corpus_chunks]
        self._bm25_model = BM25Okapi(tokenized_corpus)
        return len(self._corpus_chunks)

    def _ensure_bm25_index(self) -> None:
        """Ensure the BM25 model is initialized if corpus data is available."""
        if self._bm25_model is None:
            self.build_bm25_index()

    def retrieve_keyword(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
        """Retrieve chunks using BM25 keyword matching.

        Args:
            query: User's search query.
            top_k: Number of highest-scoring chunks to return.

        Returns:
            List of result dictionaries sorted by BM25 score descending.
        """
        self._ensure_bm25_index()

        if self._bm25_model is None or not self._corpus_chunks:
            return []

        tokenized_query = tokenize_text(query)
        if not tokenized_query:
            return []

        scores = self._bm25_model.get_scores(tokenized_query)
        # Get sorted indices descending
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results: List[Dict[str, Any]] = []
        rank = 1

        for idx in ranked_indices:
            score = float(scores[idx])
            # Only include chunks with positive relevance (at least one matching query token)
            if score <= 0.0:
                break

            chunk = self._corpus_chunks[idx]
            results.append({
                "chunk_id": chunk.get("chunk_id", f"chunk_{idx}"),
                "source": chunk.get("source", "unknown"),
                "page_number": int(chunk.get("page_number", 1)),
                "text": chunk.get("text", ""),
                "score": score,
                "rank": rank,
                "payload": chunk,
            })
            rank += 1
            if len(results) >= top_k:
                break

        return results

    def retrieve_semantic(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
        """Retrieve chunks using dense vector similarity search.

        Args:
            query: User's search query.
            top_k: Number of most similar chunks to return.

        Returns:
            List of result dictionaries sorted by vector similarity descending.
        """
        query_vector = self.embedding_generator.embed_query(query)
        raw_results = self.vector_store.search(query_vector, top_k=top_k)

        formatted: List[Dict[str, Any]] = []
        for rank, res in enumerate(raw_results, start=1):
            formatted.append({
                "chunk_id": res.get("chunk_id", ""),
                "source": res.get("source", "unknown"),
                "page_number": int(res.get("page_number", 1)),
                "text": res.get("text", ""),
                "score": float(res.get("score", 0.0)),
                "rank": rank,
                "payload": res.get("payload", {}),
            })

        return formatted

    def retrieve_hybrid(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        rrf_k: Optional[int] = None,
        dense_weight: Optional[float] = None,
        bm25_weight: Optional[float] = None,
        candidate_multiplier: int = 3,
    ) -> List[Dict[str, Any]]:
        """Retrieve chunks by fusing dense semantic and sparse BM25 rankings with RRF.

        RRF Score Formula:
            RRF(d) = dense_weight / (k + rank_dense) + bm25_weight / (k + rank_bm25)

        Args:
            query: User's search query.
            top_k: Final number of fused chunks to return.
            rrf_k: Smoothing constant for reciprocal rank fusion (defaults to instance rrf_k).
            dense_weight: Weight applied to dense retrieval ranks (defaults to instance weight).
            bm25_weight: Weight applied to BM25 retrieval ranks (defaults to instance weight).
            candidate_multiplier: Factor to expand candidate retrieval before fusion.

        Returns:
            List of result dictionaries sorted by combined RRF score descending.
        """
        k_val = rrf_k if rrf_k is not None else self.rrf_k
        w_dense = dense_weight if dense_weight is not None else self.dense_weight
        w_bm25 = bm25_weight if bm25_weight is not None else self.bm25_weight

        # Fetch candidate pools from both retrievers
        candidate_k = max(top_k * candidate_multiplier, 20)
        dense_candidates = self.retrieve_semantic(query, top_k=candidate_k)
        bm25_candidates = self.retrieve_keyword(query, top_k=candidate_k)

        # Map chunk_id to fused score and stored item
        fused_scores: Dict[str, float] = {}
        chunk_registry: Dict[str, Dict[str, Any]] = {}

        # Accumulate dense ranks and record dense cosine similarity scores
        dense_scores: Dict[str, float] = {}
        for item in dense_candidates:
            cid = item["chunk_id"]
            chunk_registry[cid] = item
            dense_scores[cid] = float(item.get("score", 0.0))
            fused_scores[cid] = fused_scores.get(cid, 0.0) + (w_dense / (k_val + item["rank"]))

        # Accumulate BM25 ranks and record BM25 scores
        bm25_scores: Dict[str, float] = {}
        for item in bm25_candidates:
            cid = item["chunk_id"]
            if cid not in chunk_registry:
                chunk_registry[cid] = item
            bm25_scores[cid] = float(item.get("score", 0.0))
            fused_scores[cid] = fused_scores.get(cid, 0.0) + (w_bm25 / (k_val + item["rank"]))

        if not fused_scores:
            return []

        # Theoretical maximum RRF score (rank 1 in both dense and BM25)
        max_possible_rrf = (w_dense / (k_val + 1)) + (w_bm25 / (k_val + 1))

        # Sort all candidates by descending fused RRF score
        sorted_cids = sorted(fused_scores.keys(), key=lambda cid: fused_scores[cid], reverse=True)

        results: List[Dict[str, Any]] = []
        for rank, cid in enumerate(sorted_cids[:top_k], start=1):
            original = chunk_registry[cid]
            raw_score = fused_scores[cid]
            norm_rrf = float(raw_score / max_possible_rrf) if max_possible_rrf > 0 else float(raw_score)

            # Ground hybrid score in dense semantic similarity to prevent rank-only inflation
            # on small corpora or unanswerable queries
            sim = dense_scores.get(cid, 0.5 if cid in bm25_scores else 0.0)
            calibrated_score = round(norm_rrf * max(0.0, sim), 4)

            results.append({
                "chunk_id": cid,
                "source": original.get("source", "unknown"),
                "page_number": int(original.get("page_number", 1)),
                "text": original.get("text", ""),
                "score": calibrated_score,
                "rank": rank,
                "payload": original.get("payload", {}),
            })

        return results

    def retrieve(
        self,
        query: str,
        mode: Optional[str] = None,
        top_k: int = DEFAULT_TOP_K,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Unified retrieval interface supporting 'keyword', 'semantic', and 'hybrid' modes.

        All modes return a uniform list of dictionaries containing:
        'chunk_id', 'source', 'page_number', 'text', 'score', and 'rank'.

        Args:
            query: Search query string.
            mode: One of 'keyword' (or 'bm25'), 'semantic' (or 'dense'), 'hybrid'.
                  If None, uses self.default_mode.
            top_k: Number of top results to return.
            **kwargs: Additional parameters passed to retrieve_hybrid (e.g. rrf_k, weights).

        Returns:
            Ranked list of result dictionaries with identical structures.

        Raises:
            ValueError: If an unsupported mode is provided.
        """
        chosen_mode = (mode or self.default_mode).lower().strip()

        if chosen_mode in ("keyword", "bm25"):
            return self.retrieve_keyword(query, top_k=top_k)
        elif chosen_mode in ("semantic", "dense"):
            return self.retrieve_semantic(query, top_k=top_k)
        elif chosen_mode == "hybrid":
            return self.retrieve_hybrid(query, top_k=top_k, **kwargs)
        else:
            raise ValueError(
                f"Invalid retrieval mode '{chosen_mode}'. "
                f"Supported modes are: 'keyword', 'semantic', 'hybrid'."
            )

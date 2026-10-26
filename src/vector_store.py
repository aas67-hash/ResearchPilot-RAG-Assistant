"""Local vector database management using Qdrant (local mode / in-memory / disk-persisted).

Handles:
- Initializing and configuring Qdrant local collections
- Upserting document chunks with full metadata as payload
- Tracking indexed documents via file hashes to skip re-embedding unchanged files
- Detecting duplicate file uploads across different filenames
- Dense vector similarity search returning top-K chunks with similarity scores
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from .chunker import chunk_document
from .embeddings import EmbeddingGenerator
from .pdf_processor import extract_text_by_page


def compute_file_hash(file_path: Union[str, Path]) -> str:
    """Compute the SHA-256 hexadecimal digest of a local file.

    Args:
        file_path: Path to the target file.

    Returns:
        64-character SHA-256 hexadecimal hash string.
    """
    path = Path(file_path)
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class VectorStore:
    """Manages local Qdrant collection, persistence, and document registry."""

    def __init__(
        self,
        collection_name: str = "research_papers",
        storage_path: Optional[str] = "qdrant_storage",
        embedding_dimension: int = 384,
        distance: Distance = Distance.COSINE,
    ) -> None:
        """Initialize local Qdrant vector store.

        Args:
            collection_name: Name of the vector collection.
            storage_path: Filesystem path for persistence, or ':memory:'/None for in-memory mode.
            embedding_dimension: Size of dense vector embeddings (default: 384 for all-MiniLM-L6-v2).
            distance: Vector similarity metric (default: Cosine).
        """
        self.collection_name = collection_name
        self.storage_path = storage_path
        self.embedding_dimension = embedding_dimension
        self.distance = distance

        # Initialize Qdrant Client (in-memory or local disk path)
        if storage_path is None or storage_path == ":memory:":
            self.is_in_memory = True
            self.client = QdrantClient(location=":memory:")
            self._manifest_file: Optional[Path] = None
        else:
            self.is_in_memory = False
            path_obj = Path(storage_path)
            path_obj.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(path_obj))
            self._manifest_file = path_obj / "document_manifest.json"

        # Manifest tracking: maps source -> {"file_hash": str, "chunk_count": int, "indexed_at": str}
        self._manifest: Dict[str, Dict[str, Any]] = self._load_manifest()

        # Ensure vector collection exists
        self._ensure_collection()

    def _load_manifest(self) -> Dict[str, Dict[str, Any]]:
        """Load document tracking manifest from disk if available."""
        if self._manifest_file and self._manifest_file.exists():
            try:
                with open(self._manifest_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_manifest(self) -> None:
        """Persist document tracking manifest to disk if using file storage."""
        if self._manifest_file:
            with open(self._manifest_file, "w", encoding="utf-8") as f:
                json.dump(self._manifest, f, indent=2)

    def close(self) -> None:
        """Close the underlying Qdrant client connection and release file locks."""
        if hasattr(self, "client") and self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass

    def __enter__(self) -> "VectorStore":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def _ensure_collection(self) -> None:
        """Create the collection if it does not already exist."""
        existing_collections = [col.name for col in self.client.get_collections().collections]
        if self.collection_name not in existing_collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dimension,
                    distance=self.distance,
                ),
            )

    def check_document_status(self, source: str, file_hash: str) -> Dict[str, Any]:
        """Check whether a document is new, unchanged, modified, or a duplicate upload.

        Args:
            source: Filename of the target document.
            file_hash: SHA-256 hash of the target document content.

        Returns:
            Dictionary with status:
                - 'UNCHANGED': Same file name and identical hash already indexed.
                - 'DUPLICATE_CONTENT': Different filename but identical content hash to an existing file.
                - 'MODIFIED': Same filename exists with a different hash (content changed).
                - 'NEW': Neither filename nor hash exists in the index.
        """
        # Check if identical hash exists under any document name
        for existing_source, meta in self._manifest.items():
            if meta.get("file_hash") == file_hash:
                if existing_source == source:
                    return {
                        "status": "UNCHANGED",
                        "source": source,
                        "matched_source": existing_source,
                        "file_hash": file_hash,
                        "message": f"Document '{source}' is already indexed and unchanged.",
                    }
                else:
                    return {
                        "status": "DUPLICATE_CONTENT",
                        "source": source,
                        "matched_source": existing_source,
                        "file_hash": file_hash,
                        "message": (
                            f"File '{source}' has identical content to already indexed "
                            f"document '{existing_source}'. Skipping duplicate."
                        ),
                    }

        # Check if same filename exists with different hash
        if source in self._manifest:
            return {
                "status": "MODIFIED",
                "source": source,
                "matched_source": source,
                "file_hash": file_hash,
                "message": f"Document '{source}' was modified since last indexing. Re-indexing required.",
            }

        return {
            "status": "NEW",
            "source": source,
            "matched_source": None,
            "file_hash": file_hash,
            "message": f"Document '{source}' is new.",
        }

    def delete_document(self, source: str) -> None:
        """Delete all points associated with a specific document source.

        Args:
            source: Source document filename to remove from the collection.
        """
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source))]
            ),
        )
        if source in self._manifest:
            del self._manifest[source]
            self._save_manifest()

    def add_documents(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: np.ndarray,
        file_hash: Optional[str] = None,
        source: Optional[str] = None,
    ) -> int:
        """Upsert document chunks and embeddings into Qdrant with full metadata payloads.

        Args:
            chunks: List of chunk metadata dictionaries.
            embeddings: 2D numpy array of dense vector embeddings corresponding to chunks.
            file_hash: Optional SHA-256 hash of the parent document.
            source: Optional source document name.

        Returns:
            Number of points successfully upserted.
        """
        if not chunks:
            return 0

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: received {len(chunks)} chunks and {len(embeddings)} embeddings."
            )

        points: List[PointStruct] = []
        doc_source = source or chunks[0].get("source", "unknown")

        for idx, chunk in enumerate(chunks):
            chunk_id = chunk.get("chunk_id")
            if not chunk_id:
                raise ValueError(f"Chunk at index {idx} is missing required 'chunk_id'.")

            # Qdrant requires valid UUID or uint. Deterministic UUID5 preserves stability.
            point_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))

            payload = {
                "chunk_id": chunk_id,
                "source": chunk.get("source", doc_source),
                "page_number": chunk.get("page_number", 1),
                "text": chunk.get("text", ""),
                "start_page": chunk.get("start_page", chunk.get("page_number", 1)),
                "end_page": chunk.get("end_page", chunk.get("page_number", 1)),
                "spans_pages": chunk.get("spans_pages", False),
                "file_hash": file_hash,
            }

            vector_values = embeddings[idx].tolist()
            points.append(PointStruct(id=point_uuid, vector=vector_values, payload=payload))

        # Batch upsert points
        self.client.upsert(collection_name=self.collection_name, points=points)

        # Update document registry
        if file_hash:
            self._manifest[doc_source] = {
                "file_hash": file_hash,
                "chunk_count": len(chunks),
                "indexed_at": datetime.now(timezone.utc).isoformat(),
            }
            self._save_manifest()

        return len(points)

    def search(
        self,
        query_vector: Union[np.ndarray, List[float]],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Perform dense vector similarity search in Qdrant.

        Args:
            query_vector: Query embedding vector.
            top_k: Number of most similar chunks to return.
            score_threshold: Minimum similarity score threshold.

        Returns:
            List of result dictionaries sorted by similarity score (descending), each with:
                - chunk_id: Unique chunk identifier
                - source: Source filename
                - page_number: Attributed page number
                - text: Chunk content text
                - score: Similarity score
                - payload: Full metadata dictionary
        """
        if isinstance(query_vector, np.ndarray):
            vector_list = query_vector.tolist()
        else:
            vector_list = query_vector

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector_list,
            limit=top_k,
            score_threshold=score_threshold,
        )

        results: List[Dict[str, Any]] = []
        for point in response.points:
            payload = point.payload or {}
            results.append({
                "chunk_id": payload.get("chunk_id", str(point.id)),
                "source": payload.get("source", "unknown"),
                "page_number": payload.get("page_number", 1),
                "text": payload.get("text", ""),
                "score": float(point.score),
                "payload": payload,
            })

        return results

    def get_all_chunks(self, limit: int = 10000) -> List[Dict[str, Any]]:
        """Retrieve all chunk payloads stored in the collection.

        Args:
            limit: Maximum number of points to scroll.

        Returns:
            List of chunk payload dictionaries.
        """
        response, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [dict(p.payload) for p in response if p.payload]

    def index_document(
        self,
        file_path: Union[str, Path],
        embedding_generator: Optional[EmbeddingGenerator] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        cross_page: bool = True,
    ) -> Dict[str, Any]:
        """Index a PDF file: validate, check hash, chunk, embed, and store.

        Skips re-embedding if the document is unchanged or a duplicate upload.

        Args:
            file_path: Path to the target PDF file.
            embedding_generator: EmbeddingGenerator instance (created if None).
            chunk_size: Character length per chunk.
            chunk_overlap: Character overlap between consecutive chunks.
            cross_page: Whether chunks can span page boundaries.

        Returns:
            Dictionary describing indexing outcome:
                - 'status': 'INDEXED', 'SKIPPED_UNCHANGED', or 'SKIPPED_DUPLICATE'
                - 'source': Document filename
                - 'chunks_indexed': Number of chunks added
                - 'message': Description of action taken
        """
        path = Path(file_path)
        source = path.name
        file_hash = compute_file_hash(path)

        # Check existing index status
        status_info = self.check_document_status(source, file_hash)
        if status_info["status"] == "UNCHANGED":
            return {
                "status": "SKIPPED_UNCHANGED",
                "source": source,
                "chunks_indexed": 0,
                "message": status_info["message"],
            }
        elif status_info["status"] == "DUPLICATE_CONTENT":
            return {
                "status": "SKIPPED_DUPLICATE",
                "source": source,
                "chunks_indexed": 0,
                "message": status_info["message"],
            }
        elif status_info["status"] == "MODIFIED":
            # Remove previous points for this modified document before re-indexing
            self.delete_document(source)

        # Extract text by page
        pages = extract_text_by_page(path)

        # Chunk pages
        chunks = chunk_document(
            pages,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            cross_page=cross_page,
        )

        if not chunks:
            self._manifest[source] = {
                "file_hash": file_hash,
                "chunk_count": 0,
                "indexed_at": datetime.now(timezone.utc).isoformat(),
            }
            self._save_manifest()
            return {
                "status": "INDEXED",
                "source": source,
                "chunks_indexed": 0,
                "message": f"Document '{source}' contained no non-empty chunks to index.",
            }

        # Initialize embedding generator if not provided
        embedder = embedding_generator or EmbeddingGenerator()

        # Embed chunk texts
        texts = [c["text"] for c in chunks]
        embeddings = embedder.embed_texts(texts)

        # Upsert into Qdrant
        count = self.add_documents(chunks, embeddings, file_hash=file_hash, source=source)

        return {
            "status": "INDEXED",
            "source": source,
            "chunks_indexed": count,
            "message": f"Successfully indexed {count} chunks from '{source}'.",
        }

"""Document chunking module for ResearchPilot.

Splits page-level research text into discrete semantic chunks while
preserving mandatory metadata:
- source: source filename
- page_number: attributed page number
- chunk_id: stable and deterministic hash

Page Boundary Attribution:
    When chunks span across page boundaries (e.g. text begins on Page 1 and continues
    onto Page 2):
    1. The chunk's primary `page_number` is attributed to the starting page (`start_page`)
       where the chunk originates.
    2. Rationale: In academic research and citation validation, researchers inspecting
       cited evidence need to begin reading at the origin of the passage.
    3. For full provenance transparency, each chunk dictionary also provides `start_page`,
       `end_page`, and a boolean flag `spans_pages`.
    4. If page-isolated chunking is required, setting `cross_page=False` strictly limits
       chunks within their respective individual page boundaries.
"""

from collections import defaultdict
import hashlib
from typing import Any, Dict, List, Optional, Tuple


def generate_chunk_id(source: str, page_number: int, chunk_index: int) -> str:
    """Generate a stable, deterministic chunk identifier from source, page, and index.

    Args:
        source: Source document filename (e.g. 'paper.pdf').
        page_number: Attributed page number for the chunk.
        chunk_index: Deterministic zero-based index of the chunk.

    Returns:
        A deterministic 16-character SHA-256 hexadecimal string.
    """
    raw_key = f"{source}:{page_number}:{chunk_index}"
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]


def _find_page_number_for_offset(offset: int, page_spans: List[Tuple[int, int, int]]) -> int:
    """Map a character offset back to its origin page number.

    Args:
        offset: Character index in the concatenated text stream.
        page_spans: List of (start_offset, end_offset, page_number) tuples.

    Returns:
        The page number containing or immediately preceding the offset.
    """
    for start, end, page_num in page_spans:
        if start <= offset < end:
            return page_num

    # If offset falls into a delimiter boundary between pages, attribute to preceding page
    for i, (start, _, page_num) in enumerate(page_spans):
        if offset < start:
            return page_spans[max(0, i - 1)][2]

    return page_spans[-1][2]


def chunk_text_sliding_window(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Tuple[int, int, str]]:
    """Produce sliding window slices over a single text string.

    Args:
        text: Input text string.
        chunk_size: Maximum character length per chunk.
        chunk_overlap: Overlapping character count between consecutive chunks.

    Returns:
        List of tuples: (start_offset, end_offset, stripped_chunk_text).
    """
    if not text or not text.strip():
        return []

    text_length = len(text)
    step = chunk_size - chunk_overlap
    chunks: List[Tuple[int, int, str]] = []
    start = 0

    while start < text_length:
        end = min(start + chunk_size, text_length)
        raw_slice = text[start:end]
        cleaned_slice = raw_slice.strip()

        if cleaned_slice:
            chunks.append((start, end, cleaned_slice))

        if end >= text_length:
            break

        start += step

    return chunks


def chunk_document(
    pages: List[Dict[str, Any]],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    cross_page: bool = True,
) -> List[Dict[str, Any]]:
    """Split page records into discrete chunks with preserved metadata.

    Page Boundary Attribution:
        When a chunk spans across page boundaries (e.g. text starting on Page 1 and
        continuing onto Page 2), `page_number` is attributed to the starting page
        (`start_page`) where the chunk originates.
        For granular multi-page tracing, `start_page`, `end_page`, and `spans_pages`
        are also included in each chunk record.

    Args:
        pages: List of page dictionaries, each containing 'source', 'page_number', and 'text'.
        chunk_size: Target size of each chunk in characters. Must be > 0.
        chunk_overlap: Overlapping size between consecutive chunks. Must be >= 0 and < chunk_size.
        cross_page: If True, text flows across page boundaries within each document.
                    If False, chunks are strictly confined to individual pages.

    Returns:
        List of chunk dictionaries with keys:
            - chunk_id: Stable deterministic hash of source, page, and chunk index
            - source: Source filename
            - page_number: Attributed page number (starting page if spanning)
            - text: Non-empty chunk text
            - start_page: Page where chunk text begins
            - end_page: Page where chunk text ends
            - spans_pages: Boolean indicating whether chunk crossed a page boundary

    Raises:
        ValueError: If chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size.
    """
    if pages is None:
        raise ValueError("Pages list cannot be None.")

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}.")

    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap cannot be negative, got {chunk_overlap}.")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})."
        )

    if not pages:
        return []

    # Group pages by source document so chunks never mix different files
    docs_by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for p in pages:
        src = p.get("source", "unknown")
        docs_by_source[src].append(p)

    all_chunks: List[Dict[str, Any]] = []

    for source, doc_pages in docs_by_source.items():
        # Sort by page number to maintain sequential flow
        sorted_pages = sorted(doc_pages, key=lambda x: x.get("page_number", 0))

        if not cross_page:
            # Per-page isolated chunking
            doc_chunk_idx = 0
            for page in sorted_pages:
                page_num = page.get("page_number", 1)
                text = page.get("text", "")
                page_chunks = chunk_text_sliding_window(text, chunk_size, chunk_overlap)

                for _, _, chunk_text in page_chunks:
                    chunk_id = generate_chunk_id(source, page_num, doc_chunk_idx)
                    all_chunks.append({
                        "chunk_id": chunk_id,
                        "source": source,
                        "page_number": page_num,
                        "text": chunk_text,
                        "start_page": page_num,
                        "end_page": page_num,
                        "spans_pages": False,
                    })
                    doc_chunk_idx += 1
        else:
            # Continuous streaming chunking across page boundaries
            page_spans: List[Tuple[int, int, int]] = []
            text_segments: List[str] = []
            current_offset = 0

            for page in sorted_pages:
                p_num = page.get("page_number", 1)
                p_text = page.get("text", "")
                if not p_text.strip():
                    continue

                p_len = len(p_text)
                page_spans.append((current_offset, current_offset + p_len, p_num))
                text_segments.append(p_text)
                # Join pages with clean paragraph separator
                current_offset += p_len + 2

            if not text_segments:
                continue

            full_doc_text = "\n\n".join(text_segments)
            doc_windows = chunk_text_sliding_window(full_doc_text, chunk_size, chunk_overlap)

            for doc_chunk_idx, (start_off, end_off, chunk_text) in enumerate(doc_windows):
                start_page = _find_page_number_for_offset(start_off, page_spans)
                end_page = _find_page_number_for_offset(max(start_off, end_off - 1), page_spans)
                spans_pages = start_page != end_page

                # Attributed page number is start_page (where chunk text originates)
                attributed_page = start_page
                chunk_id = generate_chunk_id(source, attributed_page, doc_chunk_idx)

                all_chunks.append({
                    "chunk_id": chunk_id,
                    "source": source,
                    "page_number": attributed_page,
                    "text": chunk_text,
                    "start_page": start_page,
                    "end_page": end_page,
                    "spans_pages": spans_pages,
                })

    return all_chunks

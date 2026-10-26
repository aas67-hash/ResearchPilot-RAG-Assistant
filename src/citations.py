"""Citation extraction and evidence grounding verification module.

Parses answer citations (e.g. [1], [Passage 2]), maps them back to the
exact source chunks, rejects citations that do not exist in the context,
and returns structured attribution metadata:
- filename (source document)
- page_number
- excerpt
- chunk_id
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple


def extract_citation_tags(text: str) -> List[Tuple[str, List[int]]]:
    """Extract citation tags and their referenced passage numbers from text.

    Matches:
    - [1], [2], [10]
    - [Passage 1], [passage 2]
    - [1, 2], [1, 3, 5]

    Args:
        text: Input response text.

    Returns:
        List of tuples: (raw_tag_string, [passage_number, ...]).
    """
    if not text:
        return []

    pattern = r"\[(?:Passage\s*)?([0-9]+(?:,\s*[0-9]+)*)\]"
    matches = re.finditer(pattern, text, re.IGNORECASE)

    extracted: List[Tuple[str, List[int]]] = []
    for m in matches:
        raw_tag = m.group(0)
        num_str = m.group(1)
        nums = [int(n.strip()) for n in num_str.split(",") if n.strip().isdigit()]
        if nums:
            extracted.append((raw_tag, nums))

    return extracted


def extract_chunk_id_citations(text: str) -> List[str]:
    """Extract raw 16-hex chunk IDs cited in brackets (e.g. [37111946df8f237f]).

    Args:
        text: Input response text.

    Returns:
        List of 16-character hexadecimal chunk IDs.
    """
    if not text:
        return []
    pattern = r"\[([a-f0-9]{16})\]"
    return re.findall(pattern, text, re.IGNORECASE)


def map_citations(
    answer: str,
    context_chunks: List[Dict[str, Any]],
    excerpt_max_chars: int = 250,
) -> Dict[str, Any]:
    """Map answer citations back to stored chunks, rejecting non-existent citations.

    Args:
        answer: Model generated answer text containing inline citations.
        context_chunks: List of context chunks provided to the generator.
        excerpt_max_chars: Maximum character length for citation excerpt.

    Returns:
        Dictionary containing:
            - 'valid_citations': List of mapped citation dictionaries with:
                * 'filename': Source filename
                * 'source': Same as filename
                * 'page_number': Source page number
                * 'excerpt': Text excerpt from the cited chunk
                * 'chunk_id': Unique chunk identifier
                * 'citation_tag': Original bracketed citation tag (e.g. '[1]')
            - 'rejected_citations': List of cited tags/passage numbers that do not exist
            - 'cited_chunk_ids': Set of unique chunk IDs successfully cited
    """
    if not answer or not context_chunks:
        raw_tags = extract_citation_tags(answer)
        rejected = [t[0] for t in raw_tags]
        return {
            "valid_citations": [],
            "rejected_citations": rejected,
            "cited_chunk_ids": [],
        }

    total_passages = len(context_chunks)
    passage_citations = extract_citation_tags(answer)
    chunk_id_cites = extract_chunk_id_citations(answer)

    # Build chunk_id lookup
    chunk_id_map: Dict[str, Tuple[int, Dict[str, Any]]] = {}
    for idx, c in enumerate(context_chunks, start=1):
        cid = c.get("chunk_id")
        if cid:
            chunk_id_map[cid.lower()] = (idx, c)

    valid_citations: List[Dict[str, Any]] = []
    rejected_citations: List[str] = []
    seen_keys: Set[Tuple[str, int]] = set()
    cited_chunk_ids: List[str] = []

    # 1. Process passage number citations (e.g. [1], [Passage 2])
    for raw_tag, passage_nums in passage_citations:
        for p_num in passage_nums:
            if 1 <= p_num <= total_passages:
                target_chunk = context_chunks[p_num - 1]
                chunk_id = target_chunk.get("chunk_id", f"passage_{p_num}")
                filename = target_chunk.get("source", "unknown")
                page_number = int(target_chunk.get("page_number", 1))

                dedup_key = (chunk_id, p_num)
                if dedup_key not in seen_keys:
                    seen_keys.add(dedup_key)
                    if chunk_id not in cited_chunk_ids:
                        cited_chunk_ids.append(chunk_id)

                    text = target_chunk.get("text", "").strip()
                    excerpt = text[:excerpt_max_chars] + ("..." if len(text) > excerpt_max_chars else "")

                    valid_citations.append({
                        "filename": filename,
                        "source": filename,
                        "page_number": page_number,
                        "excerpt": excerpt,
                        "chunk_id": chunk_id,
                        "citation_tag": f"[{p_num}]",
                    })
            else:
                # Passage number out of bounds
                invalid_tag = f"[{p_num}]"
                if invalid_tag not in rejected_citations:
                    rejected_citations.append(invalid_tag)

    # 2. Process direct chunk_id citations if present
    for cid in chunk_id_cites:
        cid_lower = cid.lower()
        if cid_lower in chunk_id_map:
            p_num, target_chunk = chunk_id_map[cid_lower]
            dedup_key = (cid, p_num)
            if dedup_key not in seen_keys:
                seen_keys.add(dedup_key)
                if cid not in cited_chunk_ids:
                    cited_chunk_ids.append(cid)

                text = target_chunk.get("text", "").strip()
                excerpt = text[:excerpt_max_chars] + ("..." if len(text) > excerpt_max_chars else "")

                valid_citations.append({
                    "filename": target_chunk.get("source", "unknown"),
                    "source": target_chunk.get("source", "unknown"),
                    "page_number": int(target_chunk.get("page_number", 1)),
                    "excerpt": excerpt,
                    "chunk_id": cid,
                    "citation_tag": f"[{cid}]",
                })
        else:
            invalid_tag = f"[{cid}]"
            if invalid_tag not in rejected_citations:
                rejected_citations.append(invalid_tag)

    return {
        "valid_citations": valid_citations,
        "rejected_citations": rejected_citations,
        "cited_chunk_ids": cited_chunk_ids,
    }


def verify_grounding(
    answer: str,
    context_chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Verify that an answer is evidence-grounded and has no hallucinated citations.

    Args:
        answer: Model generated response string.
        context_chunks: List of context chunks provided to the model.

    Returns:
        Dictionary with:
            - 'is_grounded': True if answer has valid citations and zero rejected citations
            - 'has_citations': True if at least one citation was present
            - 'has_hallucinated_citations': True if any citation failed mapping
            - 'valid_citations': List of mapped citation dictionaries
            - 'rejected_citations': List of invalid citation strings
    """
    mapping = map_citations(answer, context_chunks)
    valid_cites = mapping["valid_citations"]
    rejected_cites = mapping["rejected_citations"]

    has_citations = len(valid_cites) > 0
    has_hallucinated = len(rejected_cites) > 0

    # Grounded if citations exist, none were rejected, and not an abstention
    is_abstention = "insufficient evidence" in answer.lower()
    is_grounded = has_citations and not has_hallucinated and not is_abstention

    return {
        "is_grounded": is_grounded,
        "has_citations": has_citations,
        "has_hallucinated_citations": has_hallucinated,
        "is_abstention": is_abstention,
        "valid_citations": valid_cites,
        "rejected_citations": rejected_cites,
    }

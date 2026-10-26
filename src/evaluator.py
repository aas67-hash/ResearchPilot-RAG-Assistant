"""Evaluation module for retrieval performance, citation accuracy, and abstention.

Computes empirical benchmark metrics:
- Retrieval: Recall@K and MRR across keyword, semantic, and hybrid modes
- Abstention: Accuracy on unanswerable questions (pre-LLM & model abstention)
- Citations: Citation precision and provenance matching accuracy
- Reproducibility: Saves timestamped JSON and CSV reports with execution configuration
"""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .generator import GroundedGenerator
from .retriever import HybridRetriever

RESULTS_DIR = Path("evaluation") / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def is_chunk_relevant(
    chunk: Dict[str, Any],
    relevant_papers: List[str],
    expected_pages: Optional[List[int]] = None,
) -> bool:
    """Determine whether a retrieved chunk matches ground truth papers and pages.

    Args:
        chunk: Retrieved chunk dictionary with 'source' and 'page_number'.
        relevant_papers: List of ground-truth relevant document filenames.
        expected_pages: Optional list of expected page numbers.

    Returns:
        True if the chunk matches relevant papers (and expected pages if provided).
    """
    if not relevant_papers:
        return False

    chunk_source = chunk.get("source", "").lower()
    paper_matched = any(p.lower() == chunk_source for p in relevant_papers)

    if not paper_matched:
        return False

    if expected_pages:
        chunk_page = int(chunk.get("page_number", 0))
        return chunk_page in expected_pages

    return True


def calculate_recall_at_k(
    retrieved_chunks: List[Dict[str, Any]],
    relevant_papers: List[str],
    expected_pages: Optional[List[int]] = None,
) -> float:
    """Calculate binary Recall@K for a single query.

    Args:
        retrieved_chunks: Top-K retrieved chunks.
        relevant_papers: Ground-truth relevant document filenames.
        expected_pages: Ground-truth expected page numbers.

    Returns:
        1.0 if at least one relevant chunk was retrieved in top-K, else 0.0.
    """
    for chunk in retrieved_chunks:
        if is_chunk_relevant(chunk, relevant_papers, expected_pages):
            return 1.0
    return 0.0


def calculate_reciprocal_rank(
    retrieved_chunks: List[Dict[str, Any]],
    relevant_papers: List[str],
    expected_pages: Optional[List[int]] = None,
) -> float:
    """Calculate Reciprocal Rank (1/rank) for a single query.

    Args:
        retrieved_chunks: Top-K retrieved chunks.
        relevant_papers: Ground-truth relevant document filenames.
        expected_pages: Ground-truth expected page numbers.

    Returns:
        1/rank of the first relevant chunk (1-indexed), or 0.0 if none found.
    """
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        if is_chunk_relevant(chunk, relevant_papers, expected_pages):
            return 1.0 / float(idx)
    return 0.0


def evaluate_retrieval_modes(
    questions: List[Dict[str, Any]],
    retriever: HybridRetriever,
    top_k: int = 3,
    modes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Evaluate and compare Recall@K and MRR across keyword, semantic, and hybrid modes.

    Args:
        questions: Benchmark questions loaded from questions.json.
        retriever: HybridRetriever instance.
        top_k: Cut-off rank for evaluation.
        modes: List of modes to evaluate (defaults to ['keyword', 'semantic', 'hybrid']).

    Returns:
        Dictionary containing aggregate metrics and per-question evaluations.
    """
    eval_modes = modes or ["keyword", "semantic", "hybrid"]
    answerable_questions = [q for q in questions if q.get("answerable", True)]

    mode_metrics: Dict[str, Dict[str, float]] = {}
    per_question_retrieval: List[Dict[str, Any]] = []

    for mode in eval_modes:
        recalls: List[float] = []
        reciprocal_ranks: List[float] = []

        for q in answerable_questions:
            q_id = q.get("id", "")
            query_text = q.get("question", "")
            rel_papers = q.get("relevant_papers", [])
            exp_pages = q.get("expected_pages", [])

            # Run retrieval
            retrieved = retriever.retrieve(query_text, mode=mode, top_k=top_k)

            recall = calculate_recall_at_k(retrieved, rel_papers, exp_pages)
            rr = calculate_reciprocal_rank(retrieved, rel_papers, exp_pages)

            recalls.append(recall)
            reciprocal_ranks.append(rr)

            top_source = retrieved[0].get("source", "none") if retrieved else "none"
            top_page = retrieved[0].get("page_number", 0) if retrieved else 0

            per_question_retrieval.append({
                "question_id": q_id,
                "question": query_text,
                "mode": mode,
                "top_k": top_k,
                "recall_at_k": recall,
                "reciprocal_rank": rr,
                "top_hit_source": top_source,
                "top_hit_page": top_page,
            })

        mean_recall = sum(recalls) / len(recalls) if recalls else 0.0
        mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

        mode_metrics[mode] = {
            f"recall_at_{top_k}": round(mean_recall, 4),
            "mrr": round(mrr, 4),
            "evaluated_queries": len(answerable_questions),
        }

    return {
        "modes": mode_metrics,
        "per_question": per_question_retrieval,
        "total_answerable_questions": len(answerable_questions),
    }


def evaluate_abstention_and_citations(
    questions: List[Dict[str, Any]],
    retriever: HybridRetriever,
    generator: GroundedGenerator,
    top_k: int = 3,
    retrieval_mode: str = "hybrid",
    score_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Evaluate abstention accuracy on unanswerable questions and citation accuracy.

    Args:
        questions: Benchmark questions from questions.json.
        retriever: HybridRetriever instance.
        generator: GroundedGenerator instance.
        top_k: Number of passages retrieved.
        retrieval_mode: Mode used for generation evaluation.
        score_threshold: Abstention threshold override.

    Returns:
        Dictionary reporting abstention accuracy and citation precision.
    """
    unanswerable = [q for q in questions if not q.get("answerable", True)]
    answerable = [q for q in questions if q.get("answerable", True)]

    # 1. Evaluate Abstention on Unanswerable Questions
    correct_abstentions = 0
    abstention_details: List[Dict[str, Any]] = []

    for q in unanswerable:
        query_text = q.get("question", "")
        passages = retriever.retrieve(query_text, mode=retrieval_mode, top_k=top_k)
        result = generator.generate_answer(query_text, passages, score_threshold=score_threshold)

        is_abstained = result.get("abstained", False)
        if is_abstained:
            correct_abstentions += 1

        abstention_details.append({
            "question_id": q.get("id", ""),
            "question": query_text,
            "abstained": is_abstained,
            "abstain_reason": result.get("abstain_reason", ""),
            "top_score": passages[0].get("score", 0.0) if passages else 0.0,
        })

    abstention_accuracy = (
        correct_abstentions / len(unanswerable) if unanswerable else 1.0
    )

    # 2. Evaluate Citations on Answerable Questions
    valid_citation_counts: List[int] = []
    rejected_citation_counts: List[int] = []
    grounded_flags: List[bool] = []
    citation_details: List[Dict[str, Any]] = []

    for q in answerable:
        query_text = q.get("question", "")
        passages = retriever.retrieve(query_text, mode=retrieval_mode, top_k=top_k)
        result = generator.generate_answer(query_text, passages, score_threshold=score_threshold)

        valid_cites = result.get("citations", [])
        rejected_cites = result.get("rejected_citations", [])
        is_grounded = result.get("grounding", {}).get("is_grounded", False)

        valid_citation_counts.append(len(valid_cites))
        rejected_citation_counts.append(len(rejected_cites))
        grounded_flags.append(is_grounded)

        citation_details.append({
            "question_id": q.get("id", ""),
            "question": query_text,
            "valid_citations_count": len(valid_cites),
            "rejected_citations_count": len(rejected_cites),
            "is_grounded": is_grounded,
            "abstained": result.get("abstained", False),
        })

    total_valid = sum(valid_citation_counts)
    total_rejected = sum(rejected_citation_counts)
    total_generated_cites = total_valid + total_rejected

    citation_precision = (
        total_valid / total_generated_cites if total_generated_cites > 0 else 1.0
    )
    grounded_answer_rate = (
        sum(grounded_flags) / len(grounded_flags) if grounded_flags else 0.0
    )

    return {
        "abstention_accuracy": round(abstention_accuracy, 4),
        "total_unanswerable": len(unanswerable),
        "correct_abstentions": correct_abstentions,
        "citation_precision": round(citation_precision, 4),
        "total_valid_citations": total_valid,
        "total_rejected_citations": total_rejected,
        "grounded_answer_rate": round(grounded_answer_rate, 4),
        "abstention_details": abstention_details,
        "citation_details": citation_details,
    }


def save_evaluation_results(
    evaluation_payload: Dict[str, Any],
    output_dir: Optional[Union[str, Path]] = None,
) -> Tuple[Path, Path]:
    """Save evaluation results to timestamped JSON and CSV files.

    Args:
        evaluation_payload: Complete evaluation report dictionary.
        output_dir: Directory where results will be saved.

    Returns:
        Tuple of (json_filepath, csv_filepath).
    """
    target_dir = Path(output_dir or RESULTS_DIR)
    target_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = target_dir / f"evaluation_{timestamp_str}.json"
    csv_path = target_dir / f"evaluation_{timestamp_str}.csv"

    # Save complete JSON report
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_payload, f, indent=2)

    # Save CSV tabular breakdown
    per_question = evaluation_payload.get("retrieval", {}).get("per_question", [])
    if per_question:
        keys = list(per_question[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(per_question)
    else:
        # Fallback empty CSV
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            f.write("timestamp,status\n")
            f.write(f"{timestamp_str},completed\n")

    return json_path, csv_path


def run_full_evaluation(
    questions_path: Union[str, Path],
    retriever: HybridRetriever,
    generator: Optional[GroundedGenerator] = None,
    top_k: int = 3,
    output_dir: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run full benchmark evaluation: retrieval modes, abstention, citations, and persistence.

    Args:
        questions_path: Path to questions.json benchmark file.
        retriever: HybridRetriever instance.
        generator: Optional GroundedGenerator instance.
        top_k: Number of passages to retrieve for evaluation.
        output_dir: Directory to store timestamped result files.
        config: Optional configuration dictionary for reproducibility logging.

    Returns:
        Consolidated evaluation report dictionary.
    """
    path = Path(questions_path)
    if not path.exists():
        raise FileNotFoundError(f"Questions benchmark file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    # Ensure BM25 index is synchronized
    retriever.build_bm25_index()

    # 1. Run Retrieval Benchmark across Keyword, Semantic, and Hybrid modes
    retrieval_results = evaluate_retrieval_modes(
        questions=questions,
        retriever=retriever,
        top_k=top_k,
    )

    # 2. Run Generation & Citation Evaluation (if generator provided)
    generation_results = {}
    if generator is not None:
        generation_results = evaluate_abstention_and_citations(
            questions=questions,
            retriever=retriever,
            generator=generator,
            top_k=top_k,
        )

    # Construct comprehensive evaluation payload with configuration
    now_iso = datetime.now(timezone.utc).isoformat()
    execution_config = {
        "embedding_model": retriever.embedding_generator.model_name,
        "llm_model": generator.model_name if generator else "none",
        "top_k": top_k,
        "rrf_k": retriever.rrf_k,
        "min_score_threshold": generator.min_score_threshold if generator else 0.0,
        "questions_file": str(path.name),
        "total_questions": len(questions),
    }
    if config:
        execution_config.update(config)

    report: Dict[str, Any] = {
        "timestamp": now_iso,
        "configuration": execution_config,
        "retrieval": retrieval_results,
        "generation": generation_results,
    }

    # Save timestamped JSON and CSV
    json_file, csv_file = save_evaluation_results(report, output_dir=output_dir)
    report["saved_files"] = {
        "json": str(json_file),
        "csv": str(csv_file),
    }

    return report

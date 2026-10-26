"""Streamlit User Interface for ResearchPilot.

Evidence-grounded research paper assistant using local Retrieval-Augmented Generation.
All business logic is strictly encapsulated in src/ modules; app.py coordinates UI presentation.
"""
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional
import streamlit as st

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.citations import map_citations, verify_grounding
from src.embeddings import EmbeddingGenerator
from src.generator import GroundedGenerator
from src.pdf_processor import (
    CorruptPDFError,
    EmptyPDFError,
    InvalidPDFError,
    PDFProcessingError,
    PDFSizeLimitError,
    SCANNED_PAGE_WARNING,
    extract_text_by_page,
)
from src.evaluator import RESULTS_DIR, run_full_evaluation
from src.retriever import HybridRetriever
from src.vector_store import VectorStore

UPLOAD_DIR = REPO_ROOT / "uploaded_documents"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@st.cache_resource
def get_generator() -> GroundedGenerator:
    """Initialize and cache the local Ollama language model."""
    return GroundedGenerator(
        model_name="qwen2.5:3b",
        host="http://localhost:11434",
        min_score_threshold=0.30,
    )


@st.cache_resource
def get_embedding_generator() -> EmbeddingGenerator:
    """Initialize and cache the local embedding generator."""
    return EmbeddingGenerator()

@st.cache_resource
def get_vector_store() -> VectorStore:
    """Initialize and cache the persistent vector database."""
    return VectorStore(
        collection_name="research_papers",
        storage_path="qdrant_storage",
        embedding_dimension=384,
    )
def handle_file_upload(
    uploaded_file: st.runtime.uploaded_file_manager.UploadedFile,
    vector_store: VectorStore,
    embedder: EmbeddingGenerator,
) -> Dict[str, str]:
    """Save and index an uploaded PDF, reporting status, OCR warnings, or errors.

    Args:
        uploaded_file: File object from st.file_uploader.
        vector_store: VectorStore instance.
        embedder: EmbeddingGenerator instance.

    Returns:
        Dictionary reporting outcome status and messages.
    """
    dest_path = UPLOAD_DIR / uploaded_file.name

    # Save uploaded bytes to local disk
    try:
        with open(dest_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    except Exception as e:
        return {"status": "error", "message": f"Failed to save file '{uploaded_file.name}': {e}"}

    # Inspect pages for low-text / scanned warnings
    try:
        pages = extract_text_by_page(dest_path)
        ocr_warnings = [
            f"Page {p['page_number']}"
            for p in pages
            if p.get("warning") == SCANNED_PAGE_WARNING
        ]
    except EmptyPDFError as e:
        return {"status": "error", "message": f"Empty PDF rejected: {e}"}
    except InvalidPDFError as e:
        return {"status": "error", "message": f"Invalid file format: {e}"}
    except CorruptPDFError as e:
        return {"status": "error", "message": f"Corrupt or damaged PDF rejected: {e}"}
    except PDFSizeLimitError as e:
        return {"status": "error", "message": f"File size limit exceeded: {e}"}
    except PDFProcessingError as e:
        return {"status": "error", "message": f"PDF processing error: {e}"}
    except Exception as e:
        return {"status": "error", "message": f"Unexpected error processing '{uploaded_file.name}': {e}"}

    # Index into VectorStore (with duplicate and unchanged skip detection)
    index_res = vector_store.index_document(dest_path, embedding_generator=embedder)

    return {
        "status": index_res["status"],
        "message": index_res["message"],
        "chunks": str(index_res["chunks_indexed"]),
        "ocr_warnings": ocr_warnings,
    }


def main() -> None:
    """Render and orchestrate the Streamlit web application."""
    st.set_page_config(
        page_title="ResearchPilot",
        page_icon="🧭",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🧭 ResearchPilot: Research Paper Assistant")
    st.caption("Evidence-grounded local RAG assistant for academic research. 100% open-source.")

    # Prominent Human Verification Notice
    st.info(
        "⚠️ **Human Verification Required:** AI-generated answers and citations are probabilistic "
        "and must be verified against original publications before citing in academic work."
    )

    # Initialize shared components
    embedder = get_embedding_generator()
    vector_store = get_vector_store()
    generator = get_generator()
    retriever = HybridRetriever(vector_store=vector_store, embedding_generator=embedder)

    # ==========================
    # SIDEBAR: INGESTION & CONFIG
    # ==========================
    st.sidebar.header("📁 Document Ingestion")

    uploaded_files = st.sidebar.file_uploader(
        "Upload Research Papers (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload academic papers. Duplicate uploads and unchanged files are automatically detected.",
    )

    if uploaded_files:
        if st.sidebar.button("Index Uploaded Files", type="primary"):
            with st.sidebar.status("Processing and indexing papers...", expanded=True) as status_box:
                for uf in uploaded_files:
                    st.write(f"Processing: **{uf.name}**")
                    result = handle_file_upload(uf, vector_store, embedder)

                    if result["status"] == "error":
                        st.sidebar.error(f"❌ {result['message']}")
                    elif result["status"] == "SKIPPED_UNCHANGED":
                        st.sidebar.info(f"ℹ️ {uf.name}: Already indexed (unchanged).")
                    elif result["status"] == "SKIPPED_DUPLICATE":
                        st.sidebar.info(f"ℹ️ {uf.name}: Duplicate content detected. Skipped.")
                    elif result["status"] == "INDEXED":
                        st.sidebar.success(f"✅ {uf.name}: Indexed {result['chunks']} chunks.")

                    # Display OCR warnings if any pages were flagged
                    for ocr_warn in result.get("ocr_warnings", []):
                        st.sidebar.warning(
                            f"⚠️ {uf.name} ({ocr_warn}): Little or no text found (possibly scanned, OCR needed)."
                        )

                # Refresh BM25 keyword index with newly added documents
                retriever.build_bm25_index()
                status_box.update(label="Document processing complete!", state="complete", expanded=False)

    # Document Index Status
    indexed_docs = vector_store._manifest
    st.sidebar.subheader("Indexed Documents")
    if indexed_docs:
        for doc_name, meta in indexed_docs.items():
            st.sidebar.text(f"• {doc_name} ({meta.get('chunk_count', 0)} chunks)")
    else:
        st.sidebar.caption("No papers currently indexed. Upload PDFs above.")

    st.sidebar.divider()

    # Search Configuration
    st.sidebar.header("⚙️ Retrieval Settings")
    retrieval_mode = st.sidebar.selectbox(
        "Retrieval Mode",
        options=["hybrid", "semantic", "keyword"],
        format_func=lambda x: {
            "hybrid": "Hybrid (RRF: Dense + BM25)",
            "semantic": "Semantic (Dense Vector Search)",
            "keyword": "Keyword (BM25 Lexical Match)",
        }[x],
        help="Choose how relevant evidence is identified.",
    )

    top_k = st.sidebar.slider(
        "Passages to Retrieve (Top-K)",
        min_value=1,
        max_value=10,
        value=3,
        help="Number of evidence chunks provided to answer synthesis.",
    )

    score_threshold = st.sidebar.slider(
        "Relevance Threshold (Abstention)",
        min_value=0.0,
        max_value=1.0,
        value=0.30,
        step=0.05,
        help="Minimum retrieval score required before invoking the generator.",
    )

    # Ollama Local LLM Status Check
    ollama_online = generator.is_ollama_available()
    if ollama_online:
        st.sidebar.success(f"🟢 Local LLM Online: {generator.model_name}")
    else:
        st.sidebar.warning("🟡 Local LLM Offline (Ollama)\nRetrieval-only mode active.")

    # ==========================
    # WORKSPACE TABS
    # ==========================
    tab_qa, tab_eval = st.tabs(["💬 Evidence Search & QA", "📊 Evaluation & Benchmarks"])

    # --------------------------
    # TAB 1: EVIDENCE SEARCH & QA
    # --------------------------
    with tab_qa:
        query = st.text_input(
            "Enter your research question:",
            placeholder="e.g. How does self-attention mechanism compute representations across tokens?",
        )

        col1, col2 = st.columns([1, 5])
        with col1:
            search_clicked = st.button("Search & Answer", type="primary", use_container_width=True)

        if search_clicked and query.strip():
            # Step 1: Retrieve Evidence Chunks
            with st.spinner(f"Retrieving top evidence passages using {retrieval_mode.upper()} mode..."):
                retriever.build_bm25_index()  # Ensure index sync
                passages = retriever.retrieve(query.strip(), mode=retrieval_mode, top_k=top_k)

            if not passages:
                st.warning("No relevant evidence passages found in the indexed documents.")
            else:
                # Step 2: Generate Grounded Answer (with Abstention & Offline Fallback)
                with st.spinner("Synthesizing grounded response..."):
                    answer_payload = generator.generate_answer(
                        query=query.strip(),
                        context_chunks=passages,
                        score_threshold=score_threshold,
                    )

                answer_text = answer_payload["answer"]
                is_abstained = answer_payload.get("abstained", False)
                citations = answer_payload.get("citations", [])
                rejected_citations = answer_payload.get("rejected_citations", [])

                # Display Answer
                st.markdown("### Answer")
                if is_abstained:
                    st.warning(f"🛑 **Abstention:** {answer_text}")
                elif not answer_payload.get("ollama_available", True):
                    st.info(f"ℹ️ {answer_text}")
                else:
                    st.markdown(answer_text)

                # Warning for rejected/hallucinated citations
                if rejected_citations:
                    st.error(
                        f"⚠️ **Rejected Unmapped Citations:** The model referenced citations "
                        f"that do not exist in the retrieved evidence: {', '.join(rejected_citations)}"
                    )

                st.divider()

                # ==========================
                # EXPANDABLE CITATIONS PANEL
                # ==========================
                st.markdown("### 📚 Evidence Citations & Provenance")
                if citations:
                    for cite in citations:
                        tag = cite.get("citation_tag", "")
                        filename = cite.get("filename", "unknown")
                        page = cite.get("page_number", 1)
                        chunk_id = cite.get("chunk_id", "")
                        excerpt = cite.get("excerpt", "")

                        with st.expander(f"Citation {tag} — {filename} (Page {page})", expanded=True):
                            st.markdown(f"**Document:** `{filename}` | **Page:** `{page}` | **Chunk ID:** `{chunk_id}`")
                            st.markdown(f"**Passage Excerpt:**\n> {excerpt}")
                else:
                    if not is_abstained and answer_payload.get("ollama_available", True):
                        st.caption("No specific citation tags detected in the synthesized text.")

                # ==========================
                # RAW CONTEXT INSPECTION
                # ==========================
                with st.expander("🔍 Inspect All Retrieved Chunks (Raw Context)", expanded=False):
                    for p in passages:
                        st.markdown(
                            f"**[Rank {p['rank']}] Score: `{p['score']:.4f}` | "
                            f"Source: `{p['source']}` (Page {p['page_number']}) | ID: `{p['chunk_id']}`**"
                        )
                        st.text(p["text"])
                        st.markdown("---")

    # --------------------------
    # TAB 2: SYSTEM EVALUATION & BENCHMARKS
    # --------------------------
    with tab_eval:
        st.markdown("### 📊 System Evaluation & Empirical Benchmarks")
        st.caption(
            "Evaluate Recall@K, Mean Reciprocal Rank (MRR), Abstention Accuracy, "
            "and Citation Precision on the evaluation question suite."
        )

        questions_file = REPO_ROOT / "evaluation" / "questions.json"

        if not questions_file.exists():
            st.error(f"Questions benchmark file not found at '{questions_file}'.")
            return

        with open(questions_file, "r", encoding="utf-8") as f:
            q_data = json.load(f)

        st.markdown(f"**Current Benchmark Suite:** `{questions_file.name}` ({len(q_data)} questions)")

        eval_col1, eval_col2 = st.columns([1, 2])
        with eval_col1:
            eval_top_k = st.slider("Evaluation Top-K", min_value=1, max_value=5, value=3)
        with eval_col2:
            st.write("")
            st.write("")
            run_eval_clicked = st.button("Run Full Benchmark Evaluation", type="primary")

        if run_eval_clicked:
            with st.spinner("Running retrieval & generation benchmark across all modes..."):
                retriever.build_bm25_index()
                report = run_full_evaluation(
                    questions_path=questions_file,
                    retriever=retriever,
                    generator=generator,
                    top_k=eval_top_k,
                    output_dir=RESULTS_DIR,
                )
                st.session_state["latest_eval_report"] = report

        # Render Latest Report
        report = st.session_state.get("latest_eval_report")
        if report:
            st.divider()
            st.markdown(f"#### 📈 Measured Benchmark Results ({report.get('timestamp', '')[:19]})")

            # Retrieval Comparison Cards
            modes_data = report.get("retrieval", {}).get("modes", {})
            st.markdown("##### Retrieval Performance Across Modes")

            k_col, s_col, h_col = st.columns(3)
            with k_col:
                k_info = modes_data.get("keyword", {})
                st.metric(
                    label=f"Keyword (BM25) Recall@{eval_top_k}",
                    value=f"{k_info.get(f'recall_at_{eval_top_k}', 0.0) * 100:.1f}%",
                    help="Proportion of queries with relevant chunk retrieved in top-K",
                )
                st.caption(f"MRR: **{k_info.get('mrr', 0.0):.4f}**")

            with s_col:
                s_info = modes_data.get("semantic", {})
                st.metric(
                    label=f"Semantic (Dense) Recall@{eval_top_k}",
                    value=f"{s_info.get(f'recall_at_{eval_top_k}', 0.0) * 100:.1f}%",
                    help="Proportion of queries with relevant chunk retrieved in top-K",
                )
                st.caption(f"MRR: **{s_info.get('mrr', 0.0):.4f}**")

            with h_col:
                h_info = modes_data.get("hybrid", {})
                st.metric(
                    label=f"Hybrid (RRF) Recall@{eval_top_k}",
                    value=f"{h_info.get(f'recall_at_{eval_top_k}', 0.0) * 100:.1f}%",
                    help="Proportion of queries with relevant chunk retrieved in top-K",
                )
                st.caption(f"MRR: **{h_info.get('mrr', 0.0):.4f}**")

            # Abstention & Citations Metrics
            gen_data = report.get("generation", {})
            if gen_data:
                st.markdown("##### Generation & Grounding Quality")
                abst_col, cite_col = st.columns(2)
                with abst_col:
                    abst_acc = gen_data.get("abstention_accuracy", 0.0) * 100
                    st.metric(
                        label="Abstention Accuracy",
                        value=f"{abst_acc:.1f}%",
                        help="Accuracy on unanswerable questions",
                    )
                    st.caption(
                        f"Correctly Abstained: {gen_data.get('correct_abstentions', 0)} / "
                        f"{gen_data.get('total_unanswerable', 0)}"
                    )

                with cite_col:
                    cite_prec = gen_data.get("citation_precision", 0.0) * 100
                    st.metric(
                        label="Citation Precision",
                        value=f"{cite_prec:.1f}%",
                        help="Proportion of generated citations that accurately map to context",
                    )
                    st.caption(
                        f"Valid: {gen_data.get('total_valid_citations', 0)} | "
                        f"Rejected: {gen_data.get('total_rejected_citations', 0)}"
                    )

            # Tabular breakdown
            per_q = report.get("retrieval", {}).get("per_question", [])
            if per_q:
                with st.expander("📋 Per-Question Retrieval Breakdown", expanded=False):
                    st.dataframe(per_q, use_container_width=True)

            # Export Files Download
            saved = report.get("saved_files", {})
            if saved.get("csv"):
                csv_path = Path(saved["csv"])
                if csv_path.exists():
                    with open(csv_path, "r", encoding="utf-8") as f:
                        csv_data = f.read()
                    st.download_button(
                        label="📥 Download Benchmark Results (CSV)",
                        data=csv_data,
                        file_name=csv_path.name,
                        mime="text/csv",
                    )
                st.caption(f"Results archived to `{saved.get('json', '')}`")

        # Questions preview
        with st.expander("📝 View Evaluation Questions Template", expanded=False):
            st.json(q_data)


if __name__ == "__main__":
    main()

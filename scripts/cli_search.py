import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path for direct script execution
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.embeddings import EmbeddingGenerator
from src.retriever import HybridRetriever
from src.vector_store import VectorStore


def main() -> None:
    """Run CLI retrieval pipeline."""
    parser = argparse.ArgumentParser(
        description="ResearchPilot CLI: Index PDFs and retrieve grounded evidence."
    )
    parser.add_argument(
        "--folder",
        "--docs",
        "-d",
        dest="folder",
        type=str,
        default="sample_documents",
        help="Path to folder containing PDF research papers (default: sample_documents).",
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Search query to run against the indexed papers.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["keyword", "semantic", "hybrid"],
        default="hybrid",
        help="Retrieval mode: keyword (BM25), semantic (Qdrant), or hybrid (RRF). (default: hybrid).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of relevant chunks to retrieve (default: 3).",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="qdrant_storage",
        help="Directory path for local Qdrant persistence (default: qdrant_storage).",
    )

    args = parser.parse_args()
    folder_path = Path(args.folder)

    if not folder_path.exists():
        print(f"[Error] Specified folder '{folder_path}' does not exist.")
        sys.exit(1)

    pdf_files = list(folder_path.glob("*.pdf"))
    if not pdf_files:
        print(f"[Warning] No PDF files found in '{folder_path}'.")

    print(f"=== ResearchPilot Evidence Retrieval ===")
    print(f"Retrieval Mode : {args.mode.upper()}")
    print(f"Embedding Model: all-MiniLM-L6-v2 (local)")
    print(f"Vector Database: Qdrant (local mode: {args.storage})")
    print(f"Document Folder: {folder_path.resolve()}")
    print("-" * 50)

    # Initialize components
    embedder = EmbeddingGenerator()
    vector_store = VectorStore(
        collection_name="research_papers",
        storage_path=args.storage,
        embedding_dimension=embedder.dimension,
    )

    # Index all PDFs found in folder
    if pdf_files:
        print(f"Scanning {len(pdf_files)} PDF(s)...")
        for pdf_file in pdf_files:
            result = vector_store.index_document(pdf_file, embedding_generator=embedder)
            status = result["status"]
            if status == "INDEXED":
                print(f"  [Indexed] {pdf_file.name} ({result['chunks_indexed']} chunks)")
            elif status == "SKIPPED_UNCHANGED":
                print(f"  [Skipped - Unchanged] {pdf_file.name}")
            elif status == "SKIPPED_DUPLICATE":
                print(f"  [Skipped - Duplicate] {pdf_file.name}")
        print("-" * 50)

    # Initialize retriever and sync BM25 corpus
    retriever = HybridRetriever(vector_store=vector_store, embedding_generator=embedder)
    retriever.build_bm25_index()

    # Run query retrieval
    print(f"Query: \"{args.query}\"")
    print(f"Retrieving top {args.top_k} evidence passages via {args.mode.upper()} search...\n")

    results = retriever.retrieve(args.query, mode=args.mode, top_k=args.top_k)

    if not results:
        print("No matching evidence found. Make sure PDFs have been indexed.")
        return

    for res in results:
        rank = res["rank"]
        score = res["score"]
        source = res["source"]
        page = res["page_number"]
        chunk_id = res["chunk_id"]
        text = res["text"].replace("\n", " ")

        print(f"[{rank}] Score: {score:.4f} | Source: {source} (Page {page}) | Chunk ID: {chunk_id}")
        print(f"    Excerpt: {text[:280]}..." if len(text) > 280 else f"    Excerpt: {text}")
        print()

    print("=" * 50)
    print(f"Search completed using {args.mode.upper()} retrieval. No external API used.")


if __name__ == "__main__":
    main()

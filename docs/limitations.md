# ResearchPilot System Limitations & Failure Modes

An honest, transparent appraisal of system limitations is vital when building tools for scientific and technical literature. This document details the architectural boundaries, edge cases, and known failure modes of **ResearchPilot**, spanning document ingestion, retrieval, generation, citation verification, and empirical evaluation.

---

## 1. PDF Extraction & Parsing Limitations

ResearchPilot uses **PyMuPDF** (`fitz`) for local PDF parsing. While fast, memory-efficient, and dependency-free, standard programmatic PDF extraction suffers from inherent structural limitations:

### 1.1 Multi-Column Layouts & Reading Order Confusion
- **Issue**: Academic papers frequently use two-column layouts (e.g. IEEE, ACM, NeurIPS formats). Raw text stream extraction can inadvertently read horizontally across both columns rather than vertically down column 1 followed by column 2.
- **Consequence**: Sentences from adjacent columns are interleaved, producing garbled syntax that degrades embedding quality and confuses the LLM.
- **Mitigation in ResearchPilot**: PyMuPDF's block-based extraction order is utilized, which partially respects bounding-box hierarchy, but complex floating figures or full-width equations can still disrupt the linear reading flow.

### 1.2 Table Structure Flattening
- **Issue**: Standard PDF text extraction extracts cell text as a flattened sequence of strings without retaining tabular grid coordinates, row/column spans, or header associations.
- **Consequence**: Queries asking for specific numerical comparisons from experimental tables (e.g. "What was the BLEU score on WMT 2014 in Table 2?") may retrieve the table's text, but the numbers become disconnected from their column headers.

### 1.3 Mathematical Formulas & Scientific Notation
- **Issue**: Embedded LaTeX formulas, subscripts, superscripts, summation symbols ($\sum$), and fraction bars are often rendered as vector glyphs or non-standard Unicode fonts rather than plain ASCII.
- **Consequence**: Mathematical expressions like $W_Q \in \mathbb{R}^{d_{model} \times d_k}$ may extract as disjointed characters (e.g. `"W Q R d model d k"`), losing mathematical semantics.

### 1.4 Scanned Documents & Rasterized Text
- **Issue**: Historical papers, laboratory reports, or scanned archives lack a digital text layer (consisting only of bitmap images).
- **Current Behavior**: ResearchPilot does not bundle heavy external OCR runtimes (such as Tesseract or OCRmyPDF) by default to maintain a lightweight, cross-platform installation. Instead, `src/pdf_processor.py` computes character density and whitespace ratios, flagging pages with $< 50$ characters with a `SCANNED_PAGE_WARNING`.
- **Limitation**: The system alerts the user that OCR is required, but cannot extract text from raster-only pages without external pre-processing.

### 1.5 Header, Footer, and Footnote Pollution
- **Issue**: Running headers (e.g., "JOURNAL OF MACHINE LEARNING RESEARCH, VOL 15"), conference footers, and page numbers repeat on every page.
- **Consequence**: These recurring boilerplate strings are ingested into chunks, occasionally matching keyword queries or polluting LLM context.

---

## 2. Retrieval & Embedding Limitations

### 2.1 BM25 Keyword Search Failure Modes
- **Vocabulary Mismatch**: BM25 relies strictly on exact term frequencies. It fails when a user queries synonyms or paraphrases (e.g., querying `"gradient descent step size"` will fail to match a passage that exclusively uses `"learning rate $\eta$"`).
- **Zero-Hit Inflexibility**: In `src/retriever.py`, we explicitly filter out BM25 candidates with scores $\le 0.0$ to prevent false-positive noise. However, this means keyword mode returns an empty result set if none of the query tokens appear in the corpus.

### 2.2 Dense Embedding Dilution (`all-MiniLM-L6-v2`)
- **Dimension Bottleneck**: The default embedding model compresses an entire 500-character chunk into a fixed 384-dimensional vector.
- **Representation Dilution**: Highly specific technical identifiers, exact model version strings (e.g. `"v2.1.4-beta"`), or fine-grained numerical hyperparameters can be smoothed out during mean-pooling, leading dense search to retrieve passages that are *thematically* related but lack the exact requested fact.
- **Context Length**: The model has a maximum sequence length of 256 WordPiece tokens. Any chunk exceeding this limit is silently truncated during embedding.

### 2.3 Small-Corpus Ranking Artifacts in Hybrid Fusion (RRF)
- **Rank vs. Confidence Gap**: Standard Reciprocal Rank Fusion (RRF) is purely ordinal:
  $$\text{RRF}(d) = \sum \frac{w}{k + \text{rank}(d)}$$
  In a small corpus (e.g., 2–5 chunks), *some* chunk will always be assigned rank #1. If an unanswerable query (e.g., "What are the quantum gravity equations?") is run against a small database of biology papers, the top biology chunk is rank #1 by default. Without calibration, normalized RRF score $\frac{\text{RRF}}{\text{RRF}_{\max}}$ would yield $1.0$, falsely signaling high confidence.
- **Mitigation & Remaining Boundary**: ResearchPilot mitigates this by scaling the normalized RRF score by the dense cosine similarity:
  $$\text{Score} = \text{Norm\_RRF} \times \max(0.0, \text{Sim}_{\text{dense}})$$
  While this successfully drops unrelated queries below the $0.30$ threshold, queries that have moderate semantic overlap with an unrelated topic (e.g. similarity $\approx 0.35$) may still occasionally pass the threshold.

### 2.4 Arbitrary Chunk Boundary Splitting
- **Context Severance**: Sliding-window chunking (`chunk_size=500`, `chunk_overlap=50`) splits text based on character/word counts rather than syntactic boundaries. A crucial definition, multi-sentence argument, or theorem can be split right across two chunks, leaving both halves incomplete.

---

## 3. Generation & Hallucination Risks

Even with strict system prompts, local Large Language Models (e.g., Mistral 7B, Llama 3 8B) remain probabilistic models susceptible to several generation failures:

### 3.1 Parametric Knowledge Bleed-Through
- **Issue**: The LLM possesses extensive pre-trained knowledge. When asked about famous papers (e.g. the original 2017 Transformer paper "Attention Is All You Need"), the model may answer based on its pre-training weights rather than solely reading the provided context chunks.
- **Risk**: If the user provides a modified version of the paper or a different paper that uses different notation, the model might report the standard textbook conventions rather than what is actually printed in the uploaded document.

### 3.2 Correlation-Causation and Synthesis Hallucinations
- **Issue**: When synthesizing answers across multiple passages, the model may accurately extract Fact A from Passage 1 and Fact B from Passage 2, but invent a false causal or correlative link between them that neither passage claimed.

### 3.3 Sycophantic Answering on Leading Questions
- **Issue**: If a user asks a leading question containing an incorrect premise (e.g., "Why did the authors conclude that attention layers are inferior to RNNs?"), an LLM may attempt to justify the user's premise rather than declaring that the premise contradicts the text.

---

## 4. Citation & Grounding Errors

ResearchPilot incorporates automated citation parsing (`src/citations.py`), which exposes specific error patterns:

### 4.1 Citation Hallucination
- **Issue**: The model outputs citation tags referencing passages that do not exist in the context window (e.g., generating `[Passage 7]` when only 3 passages were retrieved).
- **Handling in ResearchPilot**: The citation verification engine identifies and rejects these out-of-bounds citations, marking the answer as `is_grounded = False` and reporting the rejected tags. However, the accompanying prose written by the LLM may still be displayed.

### 4.2 Citation Misattribution
- **Issue**: When two retrieved passages discuss similar topics (e.g. Passage 1 describes scaled dot-product attention and Passage 2 describes multi-head attention), the model may attribute a claim about multi-head attention to `[Passage 1]`.
- **Limitation**: The regex citation engine verifies that Passage 1 *exists*, but semantic AST verification of whether Passage 1 *actually entailed* the specific sentence requires an NLI (Natural Language Inference) model, which is outside the current lightweight scope.

### 4.3 Broad Attribution / Under-Citation
- **Issue**: The model writes a detailed four-sentence technical summary but only places a single citation `[1]` at the very end of the paragraph. It is ambiguous whether `[1]` supports all four sentences or only the final clause.

---

## 5. Small Test-Set & Evaluation Limitations

The starter evaluation benchmark provided in `evaluation/questions.json` contains 5 sample questions. While valuable for unit and smoke testing, evaluating RAG systems on small datasets has severe limitations:

### 5.1 High Statistical Variance
- On a 5-question test set, a single failure shifts the accuracy or recall metric by **20.0 percentage points**. No statistically significant claim can be made regarding system performance with fewer than 30–50 carefully annotated queries.

### 5.2 Annotator & Domain Bias
- The gold-standard labels (`relevant_papers`, `expected_pages`) reflect the specific annotations of the creator.
- Systems evaluated exclusively on computer science or machine learning papers may perform dramatically differently on biomedical literature, legal briefs, or historical economics papers due to vocabulary, format, and layout divergence.

### 5.3 Risk of Hyperparameter Overfitting
- Tuning `chunk_size`, `chunk_overlap`, `top_k`, and `min_score_threshold` against a small batch of 5 questions risks overfitting these constants to the specific sentence lengths and similarity distributions of those specific questions.

---

## 6. Summary of Architectural Mitigations & Future Roadmap

| Limitation Domain | Current Mitigation in ResearchPilot | Future Architectural Roadmap |
| :--- | :--- | :--- |
| **Scanned PDFs** | `SCANNED_PAGE_WARNING` heuristic surfaces warning banners to the user. | Integrate optional local OCR plugin (Tesseract / Surya OCR). |
| **Multi-Column Flow** | Block-based bounding box sorting in PyMuPDF. | Adopt layout-aware vision parsers (e.g. Grobid, Marker). |
| **Small-Corpus RRF Inflation** | Calibrated score: $\text{Norm\_RRF} \times \text{Sim}_{\text{dense}}$. | Transition to learned cross-encoder rerankers (`ms-marco-MiniLM-L-6-v2`). |
| **Arbitrary Chunks** | Configurable overlap (`50` chars) + page boundary attribution. | Implement semantic boundary / proposition chunking. |
| **Citation Misattribution** | Out-of-bounds rejection + regex tag extraction. | Implement token-level NLI entailment scoring for each cited sentence. |
| **Small Benchmark** | Standardized JSON schema supporting 30–50+ questions with automated CSV/JSON export. | Curate domain-specific public evaluation benchmarks (e.g. PubMed, ArXiv-QA). |

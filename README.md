<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.135-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/PyTorch-2.11-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/FAISS-1.13-4285F4?logo=meta&logoColor=white" alt="FAISS">
  <img src="https://img.shields.io/badge/License-Academic-green" alt="License">
</p>

# LMDIS — Lossless Multimodal Document Intelligence System

A **21-module deterministic document intelligence pipeline** that transforms uploaded PDF documents into queryable knowledge graphs with evidence-grounded, citation-accurate answers.

LMDIS ingests a PDF, performs lossless content extraction (text, tables, images, OCR), builds a semantically enriched knowledge graph, and serves natural-language queries with dual-path answer generation — one grounded in the document graph, the other via LLM reasoning — automatically arbitrated for accuracy.

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────────────┐
│                        DOCUMENT UPLOAD                                 │
│  PDF ──► M1 Ingest ──► M2 Page Type ──► M3 Extract ──► M4 Normalize   │
│           │               │                │               │           │
│           ▼               ▼                ▼               ▼           │
│       M5 Structure ──► M6 Normalize ──► M7 Graph Build                 │
│                                             │                          │
│                           ┌─────────────────┤                          │
│                           ▼                 ▼                          │
│                   M8 Confidence    M9 Embeddings (BGE-large 1024d)     │
│                           │         M9b Semantic Enrichment            │
│                           ▼                 │                          │
│                     M10 Claims ──► M11 Evidence Graph ──► LLM-K Store  │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│                        QUERY PIPELINE                                  │
│                                                                        │
│  Query ──► M12 Interpret ──► M13-15 Retrieval                          │
│                                  │                                     │
│                    ┌─────────────┴─────────────┐                       │
│                    ▼                           ▼                       │
│           Path A: M16-19              Path B: LLM-K                    │
│        Evidence-Graph Answer       Direct LLM Answer                   │
│                    │                           │                       │
│                    └─────────┬─────────────────┘                       │
│                              ▼                                         │
│                    Answer Selection Layer (ASL)                         │
│                    Fact extraction, scoring, merge                      │
│                              │                                         │
│                              ▼                                         │
│                    M20 Formatter + Citation Engine                      │
│                    M21 Suggested Follow-ups                             │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Module Inventory

| Module | Name | Purpose |
|--------|------|---------|
| **M1** | Ingestion | Immutable source registration, SHA-256 fingerprint, UUID assignment |
| **M2** | Page Type | Classifies pages as digital, scanned, or mixed (text-density + image-region heuristics) |
| **M3** | Extraction | Lossless text (with font metadata), table cells (pdfplumber + PaddleOCR), image extraction |
| **M4** | Line Memory | Line-level normalization, paragraph detection, table element grouping |
| **M5** | Structure | Hierarchical section detection, heading recognition, image–caption linking |
| **M6** | Normalization | Component-level normalization across paragraphs, tables, figures |
| **M7** | Graph Build | NetworkX DiGraph construction with 15+ edge types |
| **M8** | Confidence | Per-node OCR confidence annotation, low-confidence flagging |
| **M9** | Embeddings | Deep contextual embeddings via BAAI/bge-large-en-v1.5 (1024-dim) + FAISS index |
| **M9b** | Semantic Enrichment | Information-type classification (14 types), density scoring, section summaries, cross-links |
| **M10** | Claims | Rule-based extraction: dates, money, percentages, definitions, specifications |
| **M11** | Evidence Graph | Claim nodes, inter-claim relationships, section evidence summaries |
| **LLM-K** | Knowledge Store | Direct LLM document comprehension stored in graph as backup knowledge |
| **M12** | Query Interpretation | Intent detection, sub-claim decomposition, keyword extraction |
| **M13-15** | Retrieval | Bi-encoder search → graph traversal (10+ edge types) → cross-encoder re-ranking |
| **M16-19** | Generation | Evidence-bound LLM answer with multi-signal citation attribution |
| **ASL** | Answer Selection | Dual-path fact extraction, scoring, merge, LLM-based reframing |
| **M20** | Formatter | Citation linking, composite confidence scoring, full analysis payload |
| **M21** | Suggestions | Context-aware follow-up question generation |

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.13, FastAPI, Uvicorn |
| **Embedding Model** | BAAI/bge-large-en-v1.5 (1024-dim) via sentence-transformers |
| **Cross-Encoder** | ms-marco-MiniLM-L-12-v2 for re-ranking |
| **Vector Store** | FAISS (IndexFlatIP, cosine similarity) |
| **Graph Engine** | NetworkX DiGraph, JSON serialization |
| **LLM** | OpenRouter API (Gemini 2.0 Flash, Llama 3.3 70B, DeepSeek R1, Mistral 7B) |
| **OCR** | PaddleOCR 3.4 + PyMuPDF |
| **Table Extraction** | pdfplumber (digital) + PaddleOCR structure (scanned) |
| **Frontend** | Single-page HTML/CSS/JS, glassmorphic dark UI |
| **Storage** | Per-document file-based persistence (JSON + Pickle) |

---

## Project Structure

```
LMDIS/
├── backend/
│   ├── app/
│   │   ├── main.py                    # FastAPI application, all endpoints
│   │   ├── config.py                  # API keys, model config
│   │   ├── modules/
│   │   │   ├── module1_ingestion.py
│   │   │   ├── module2_page_type.py
│   │   │   ├── module3_extraction.py
│   │   │   ├── module4_line_memory.py
│   │   │   ├── module5_structure.py
│   │   │   ├── module6_normalization.py
│   │   │   ├── module7_graph.py
│   │   │   ├── module8_confidence.py
│   │   │   ├── module9_embedding.py
│   │   │   ├── module9b_semantic_enrichment.py
│   │   │   ├── module10_claims.py
│   │   │   ├── module11_evidence_graph.py
│   │   │   ├── module12_query.py
│   │   │   ├── module13_14_15_retrieval.py
│   │   │   ├── module16_19_generation.py
│   │   │   ├── module20_formatter.py
│   │   │   ├── module21_suggestions.py
│   │   │   ├── module_llm_knowledge.py
│   │   │   └── module_answer_selection.py
│   │   ├── utils/
│   │   │   ├── persistent_storage.py  # Per-document file storage
│   │   │   └── hash_utils.py
│   │   └── storage/                   # (empty at runtime)
│   ├── requirements.txt               # Python dependencies
│   └── requirements.lock.txt          # Pinned versions
├── frontend/
│   └── index.html                     # Single-page application
├── app/
│   └── storage/
│       └── documents/                 # Per-document data store
│           └── {doc_id}/
│               ├── graph.json
│               ├── embeddings.pkl
│               ├── metadata.json
│               ├── raw/               # Original uploaded file
│               └── images/            # Extracted images
├── test_pipeline.py                   # Integration tests
├── test_transformer_pipeline.py       # Embedding & enrichment tests
├── start.bat                          # One-click application launcher
└── README.md
```

---

## Per-Document Storage

Every uploaded document is isolated into its own directory:

```
app/storage/documents/
├── a33a6c35-db7b-4aed-afe7-ba92a8b3f5ef/
│   ├── graph.json         # Full knowledge graph (nodes + edges)
│   ├── embeddings.pkl     # FAISS index + ID mappings
│   ├── metadata.json      # Filename, checksum, page count, title
│   ├── raw/
│   │   └── a33a6c35....pdf
│   └── images/
│       ├── doc_p1.png
│       └── doc_p2.png
└── b7f1e2c4-...
    └── ...
```

This ensures clean separation — no cross-contamination between documents. Documents can be individually deleted without affecting others.

---

## Quick Start

### One-Click Launch

Double-click **`start.bat`** in the project root. It will:
1. Verify Python and virtual environment
2. Check all dependencies
3. Launch the backend server
4. Open the browser when ready

### Manual Setup

```bash
# 1. Create virtual environment
python -m venv .venv

# 2. Activate
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r backend/requirements.txt

# 4. Set API key (optional — has free-tier default)
set OPENROUTER_API_KEY=sk-or-v1-your-key-here

# 5. Start backend
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000

# 6. Open frontend
# Navigate to: http://127.0.0.1:8000/frontend/index.html
# Or open frontend/index.html directly
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | System status and active module count |
| `GET` | `/documents` | List all uploaded documents |
| `POST` | `/documents/upload` | Upload and process a PDF document |
| `GET` | `/documents/{id}/status` | Check document processing status |
| `POST` | `/documents/{id}/query?query=...` | Query a document with natural language |
| `GET` | `/documents/{id}/suggestions` | Get AI-generated follow-up questions |
| `GET` | `/documents/{id}/structure` | Retrieve logical document structure |

---

## Citation Engine

Citations are built from the **cross-encoder re-ranked evidence pipeline** — not from simple keyword matching. Each citation carries:

- **Composite confidence score** — blends semantic alignment (40%), OCR quality (30%), cross-encoder relevance (20%), and contribution verification (10%)
- **Alignment score** — from the retrieval pipeline's bi-encoder + graph traversal + cross-encoder re-ranking
- **Information type** — fact, definition, statistic, methodology, result, etc.
- **Contribution flag** — whether M16-19 independently identified this evidence as contributing to the answer

The citation selection in M16-19 uses multi-signal attribution:
- Unigram overlap (25%) — content word matching
- Bigram overlap (30%) — phrase-level attribution
- High-value token overlap (25%) — numbers, proper nouns, abbreviations
- Retrieval alignment prior (20%) — score from the evidence pipeline

---

## Frontend

The single-page application provides three views:

1. **Upload** — Drag-and-drop PDF upload with real-time processing status
2. **Chat** — Natural language Q&A with citation-backed answers
3. **Analysis** — Per-query analysis dashboard with:
   - **Fact Breakdown** — Individual facts with source attribution and grounding status
   - **Evidence Graph** — Interactive visualization of retrieved evidence relationships
   - **Processing Pipeline** — Which modules contributed (M12 → M13-15 → M16-19 → ASL → M20)
   - **Confidence Scoring** — Multi-factor breakdown (OCR, semantic alignment, cross-encoder, source grounding)

---

## Testing

```bash
# Integration tests (graph build, evidence, answer selection)
python test_pipeline.py

# Transformer pipeline tests (embeddings, semantic enrichment, retrieval)
python test_transformer_pipeline.py
```

---

## Configuration

Environment variables (all optional — have working defaults):

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | Free-tier key | OpenRouter API key for LLM calls |
| `OPENROUTER_MODEL` | `openrouter/free` | Primary LLM model |

---

## Requirements

- **Python** 3.13+
- **OS** Windows 10/11 (tested), Linux/macOS (compatible)
- **RAM** 4 GB minimum (embedding model loads ~1.3 GB)
- **Disk** 2 GB for dependencies + model weights
- **Network** Required for LLM calls (OpenRouter API); embedding models download on first run

---

## License

Academic use. Capstone project — Lossless Multimodal Document Intelligence System.

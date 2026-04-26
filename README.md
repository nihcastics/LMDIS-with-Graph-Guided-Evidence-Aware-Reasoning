<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13%2B-blue?logo=python&logoColor=white" alt="Python Version">
  <img src="https://img.shields.io/badge/FastAPI-0.135-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/PyTorch-2.11-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/FAISS-1.13-4285F4?logo=meta&logoColor=white" alt="FAISS">
  <img src="https://img.shields.io/badge/License-GPL--3.0-green" alt="License">
  <img src="https://img.shields.io/badge/Status-Production-orange" alt="Status">
</p>

<h1 align="center">LMDIS: Lossless Multimodal Document Intelligence System</h1>

<p align="center">
  <strong>Graph Guided Evidence Aware Reasoning for Enterprise Document Intelligence</strong>
</p>

<p align="center">
  A production ready 21 module deterministic pipeline that transforms complex multimodal documents into queryable knowledge graphs with evidence grounded, citation accurate answers.
</p>

---

## Table of Contents

- [Overview](#overview)
- [Key Capabilities](#key-capabilities)
- [System Architecture](#system-architecture)
- [Module Inventory](#module-inventory)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Running the System](#running-the-system)
- [API Reference](#api-reference)
- [Document Processing Pipeline](#document-processing-pipeline)
- [Query and Retrieval System](#query-and-retrieval-system)
- [Citation Engine](#citation-engine)
- [Storage Architecture](#storage-architecture)
- [Frontend Interface](#frontend-interface)
- [Testing and Validation](#testing-and-validation)
- [Performance Characteristics](#performance-characteristics)
- [Use Cases](#use-cases)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License and Copyright](#license-and-copyright)
- [Contact](#contact)

---

## Overview

LMDIS is an enterprise grade document intelligence platform that addresses the critical challenge of extracting, structuring, and retrieving information from complex multimodal documents without data loss. The system combines lossless parsing techniques, graph based knowledge representation, semantic retrieval, and evidence grounded response generation to deliver traceable and accurate document analytics.

The platform ingests PDF documents in any format (digital, scanned, or hybrid), performs comprehensive content extraction preserving layout hierarchy and visual elements, constructs semantically enriched knowledge graphs, and serves natural language queries through a dual path reasoning architecture that ensures factual accuracy through citation backed responses.

---

## Key Capabilities

**Lossless Structural Parsing**
Preserves document layout, hierarchy, text formatting, table structures, and visual elements exactly as they appear in the source document. Adaptive extraction handles digital PDFs through PyMuPDF and pdfplumber, while scanned documents are processed through PaddleOCR with layout preservation.

**Knowledge Graph Construction**
Documents are transformed into directed graphs using NetworkX with fifteen plus edge types capturing semantic relationships, structural hierarchies, and cross references. This graph representation enables sophisticated reasoning, traceability, and evidence based query resolution.

**Semantic Retrieval Engine**
Replaces brittle keyword matching with embedding based similarity search using BAAI bge large en v1.5 (1024 dimensional vectors) indexed through FAISS. The retrieval pipeline combines bi encoder search, graph traversal, and cross encoder re ranking for high precision evidence extraction.

**Evidence Grounded Responses**
Every answer is backed by verifiable source evidence with clear citations. The dual path architecture generates answers from both the document knowledge graph and direct LLM reasoning, then arbitrates between them through a multi signal Answer Selection Layer to ensure accuracy and prevent hallucinations.

**Composite Confidence Scoring**
Each citation carries a multi factor confidence score blending semantic alignment (40 percent), OCR quality (30 percent), cross encoder relevance (20 percent), and contribution verification (10 percent) for transparent reliability assessment.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         DOCUMENT INGESTION AND PROCESSING                       │
│                                                                                 │
│  PDF Upload                                                                     │
│       │                                                                         │
│       ▼                                                                         │
│  ┌─────────┐    ┌────────────┐    ┌──────────┐    ┌──────────────┐            │
│  │ M1      │───▶│ M2         │───▶│ M3       │───▶│ M4           │            │
│  │Ingestion│    │Page Type   │    │Extraction│    │Line Memory   │            │
│  └─────────┘    └────────────┘    └──────────┘    └──────────────┘            │
│       │                                                                         │
│       ▼                                                                         │
│  ┌──────────┐    ┌────────────┐    ┌──────────────┐                            │
│  │M5        │───▶│ M6         │───▶│ M7           │                            │
│  │Structure │    │Normalization│   │Graph Build   │                            │
│  └──────────┘    └────────────┘    └──────┬───────┘                            │
│                                           │                                     │
│                          ┌────────────────┼────────────────┐                   │
│                          ▼                ▼                ▼                   │
│                   ┌──────────┐    ┌──────────┐    ┌──────────────┐            │
│                   │M8        │    │M9        │    │M9b           │            │
│                   │Confidence│    │Embeddings│    │Semantic      │            │
│                   └──────────┘    └────┬─────┘    │Enrichment    │            │
│                                        │          └──────┬───────┘            │
│                                        ▼                 │                     │
│                                 ┌──────────┐             │                     │
│                                 │M10       │             │                     │
│                                 │Claims    │─────────────┘                     │
│                                 └────┬─────┘                                   │
│                                      ▼                                         │
│                                 ┌──────────────┐    ┌────────────┐            │
│                                 │M11           │───▶│LLM K       │            │
│                                 │Evidence Graph│    │Store       │            │
│                                 └──────────────┘    └────────────┘            │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                              QUERY AND RESPONSE PIPELINE                        │
│                                                                                 │
│  Natural Language Query                                                         │
│       │                                                                         │
│       ▼                                                                         │
│  ┌────────────────┐                                                             │
│  │M12             │                                                             │
│  │Query           │                                                             │
│  │Interpretation  │                                                             │
│  └───────┬────────┘                                                             │
│          │                                                                      │
│          ▼                                                                      │
│  ┌───────────────────┐                                                          │
│  │M13 M14 M15        │                                                          │
│  │Retrieval Pipeline │                                                          │
│  │Bi Encoder Search  │                                                          │
│  │Graph Traversal    │                                                          │
│  │Cross Encoder Rank │                                                          │
│  └───────┬───────────┘                                                          │
│          │                                                                      │
│          ├──────────────────────────┬──────────────────────────┐               │
│          ▼                          ▼                          ▼               │
│  ┌──────────────┐          ┌──────────────┐          ┌──────────────┐         │
│  │Path A        │          │Path B        │          │LLM K         │         │
│  │M16 M19       │          │Evidence      │          │Direct LLM    │         │
│  │Generation    │          │Bound Answer  │          │Answer        │         │
│  └──────┬───────┘          └──────┬───────┘          └──────┬───────┘         │
│         │                         │                         │                 │
│         └─────────────────────────┼─────────────────────────┘                 │
│                                   ▼                                           │
│                          ┌────────────────────┐                               │
│                          │Answer Selection    │                               │
│                          │Layer (ASL)         │                               │
│                          │Fact Extraction     │                               │
│                          │Multi Signal Scoring│                               │
│                          │Merge and Reframe   │                               │
│                          └────────┬───────────┘                               │
│                                   │                                           │
│                                   ▼                                           │
│                          ┌────────────────────┐                               │
│                          │M20                 │                               │
│                          │Formatter and       │                               │
│                          │Citation Engine     │                               │
│                          └────────┬───────────┘                               │
│                                   │                                           │
│                                   ▼                                           │
│                          ┌────────────────────┐                               │
│                          │M21                 │                               │
│                          │Suggested Follow    │                               │
│                          │Up Questions        │                               │
│                          └────────────────────┘                               │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Module Inventory

| Module | Name | Function | Key Outputs |
|--------|------|----------|-------------|
| **M1** | Ingestion | Immutable source registration with SHA 256 fingerprinting and UUID assignment | Document ID, checksum, metadata |
| **M2** | Page Type Classifier | Classifies pages as digital, scanned, or mixed using text density and image region heuristics | Page type labels |
| **M3** | Content Extraction | Lossless text extraction with font metadata, table cell extraction via pdfplumber and PaddleOCR, image extraction | Raw text, tables, images |
| **M4** | Line Memory | Line level normalization, paragraph detection, table element grouping | Normalized text segments |
| **M5** | Structure Detection | Hierarchical section detection, heading recognition, image caption linking | Document hierarchy |
| **M6** | Normalization | Component level normalization across paragraphs, tables, and figures | Normalized components |
| **M7** | Graph Construction | NetworkX DiGraph construction with fifteen plus edge types capturing semantic and structural relationships | Knowledge graph |
| **M8** | Confidence Annotation | Per node OCR confidence scoring and low confidence flagging | Confidence metadata |
| **M9** | Embedding Generation | Deep contextual embeddings via BAAI bge large en v1.5 (1024 dimensional) with FAISS index construction | Vector embeddings, FAISS index |
| **M9b** | Semantic Enrichment | Information type classification across fourteen types, density scoring, section summaries, cross link generation | Semantic metadata |
| **M10** | Claim Extraction | Rule based extraction of dates, monetary values, percentages, definitions, and specifications | Claim nodes |
| **M11** | Evidence Graph | Claim node integration, inter claim relationship mapping, section evidence summary generation | Evidence graph |
| **LLM K** | Knowledge Store | Direct LLM document comprehension stored in graph as backup knowledge source | LLM derived knowledge |
| **M12** | Query Interpretation | Intent detection, sub claim decomposition, keyword extraction | Query representation |
| **M13 to 15** | Retrieval Pipeline | Bi encoder search followed by graph traversal across ten plus edge types, then cross encoder re ranking | Ranked evidence set |
| **M16 to 19** | Answer Generation | Evidence bound LLM answer generation with multi signal citation attribution | Generated answer with citations |
| **ASL** | Answer Selection | Dual path fact extraction, multi signal scoring, merge logic, LLM based reframing | Final selected answer |
| **M20** | Formatter | Citation linking, composite confidence scoring, full analysis payload construction | Formatted response |
| **M21** | Suggestions | Context aware follow up question generation based on query context and document content | Suggested questions |

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Backend Framework** | Python 3.13, FastAPI, Uvicorn | High performance async API server |
| **Embedding Model** | BAAI bge large en v1.5 via sentence transformers | 1024 dimensional semantic embeddings |
| **Cross Encoder** | ms marco MiniLM L 12 v2 | Evidence re ranking for precision retrieval |
| **Vector Store** | FAISS IndexFlatIP with cosine similarity | Efficient similarity search over embeddings |
| **Graph Engine** | NetworkX DiGraph with JSON serialization | Knowledge graph construction and traversal |
| **LLM Integration** | OpenRouter API, Google Gemini API, Anthropic Claude API | Multi model LLM orchestration |
| **Supported Models** | Gemini 2.5 Flash, Gemini 2.0 Flash, Llama 3.3 70B, Claude Sonnet 4.6, Claude Haiku 4.5, DeepSeek R1, Mistral 7B | Flexible model selection |
| **OCR Engine** | PaddleOCR 3.4 with PyMuPDF | Scanned document text extraction |
| **PDF Parsing** | pdfplumber for digital PDFs, PyMuPDF for hybrid documents | Lossless content extraction |
| **Table Extraction** | pdfplumber structural analysis for digital, PaddleOCR structure for scanned | Accurate table parsing |
| **Frontend** | Single page HTML CSS JavaScript with glassmorphic dark UI | Modern responsive interface |
| **Storage** | Per document file based persistence with JSON and Pickle serialization | Isolated document storage |

---

## Project Structure

```
LMDIS/
├── backend/                              # FastAPI backend server
│   ├── app/
│   │   ├── main.py                       # FastAPI application with all API endpoints
│   │   ├── config.py                     # Configuration management and API key handling
│   │   ├── train_model.py                # Model training utilities
│   │   ├── modules/                      # Core processing pipeline modules
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
│   │   │   ├── module_graph_semantic_alignment.py
│   │   │   └── module_answer_selection.py
│   │   ├── utils/                        # Utility modules
│   │   │   ├── persistent_storage.py     # Per document file storage management
│   │   │   └── hash_utils.py             # Cryptographic hashing utilities
│   │   ├── model/                        # Trained model artifacts
│   │   │   └── document_ai_model.pkl
│   │   └── storage/                      # Runtime document storage (excluded from version control)
│   ├── requirements.txt                  # Python package dependencies
│   └── requirements.lock.txt             # Pinned dependency versions
├── frontend/                             # Frontend user interface
│   ├── index.html                        # Single page application
│   ├── index.html.bak                    # Backup version
│   └── index.html.old                    # Previous version
├── app/                                  # Application runtime data
│   └── storage/
│       └── documents/                    # Per document data stores (excluded from version control)
│           └── {document_id}/
│               ├── graph.json            # Complete knowledge graph
│               ├── embeddings.pkl        # FAISS index and ID mappings
│               ├── metadata.json         # Document metadata and checksums
│               ├── raw/                  # Original uploaded files
│               └── images/               # Extracted images and page renders
├── test_pipeline.py                      # Integration test suite
├── test_transformer_pipeline.py          # Transformer pipeline validation tests
├── launcher.py                           # Alternative application launcher
├── start.bat                             # Windows one click application launcher
├── README.md                             # Project documentation
├── LICENSE                               # GNU GPL v3 license text
└── .vscode/                              # VS Code workspace configuration
    └── settings.json
```

---

## Getting Started

### Prerequisites

Before deploying LMDIS, ensure your environment meets the following requirements:

- **Python**: Version 3.13 or higher
- **Operating System**: Windows 10 or 11 (fully tested), Linux or macOS (compatible)
- **Memory**: Minimum 4 GB RAM (embedding model requires approximately 1.3 GB)
- **Storage**: 2 GB available disk space for dependencies and model weights
- **Network**: Internet connection required for LLM API calls and initial model downloads
- **API Keys**: At least one LLM API key (OpenRouter, Google Gemini, or Anthropic Claude)

### Installation

**Step One: Clone the Repository**

```bash
git clone https://github.com/nihcastics/LMDIS-with-Graph-Guided-Evidence-Aware-Reasoning.git
cd LMDIS-with-Graph-Guided-Evidence-Aware-Reasoning
```

**Step Two: Create and Activate Virtual Environment**

Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS or Linux:
```bash
python -m venv .venv
source .venv/bin/activate
```

**Step Three: Install Dependencies**

```bash
pip install -r backend/requirements.txt
```

The installation process will download and install all required packages including PyTorch, sentence transformers, FAISS, PaddleOCR, and other dependencies. First time installation may take several minutes depending on network speed.

### Configuration

LMDIS requires at least one LLM API key for operation. Configure your API keys through environment variables before starting the system.

**Windows (Command Prompt):**
```bash
set OPENROUTER_API_KEY=your_openrouter_api_key_here
set GOOGLE_API_KEY=your_google_api_key_here
set ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

**Windows (PowerShell):**
```powershell
$env:OPENROUTER_API_KEY="your_openrouter_api_key_here"
$env:GOOGLE_API_KEY="your_google_api_key_here"
$env:ANTHROPIC_API_KEY="your_anthropic_api_key_here"
```

**macOS or Linux:**
```bash
export OPENROUTER_API_KEY="your_openrouter_api_key_here"
export GOOGLE_API_KEY="your_google_api_key_here"
export ANTHROPIC_API_KEY="your_anthropic_api_key_here"
```

**Optional Configuration Variables:**

| Variable | Default Value | Description |
|----------|---------------|-------------|
| `OPENROUTER_API_KEY` | Empty string | OpenRouter API key for primary LLM access |
| `OPENROUTER_MODEL` | `openrouter/free` | Primary LLM model identifier |
| `OPENROUTER_STRONG_MODEL` | `google/gemini-2.5-flash` | Strong model for critical reasoning tasks |
| `GOOGLE_API_KEY` | Empty string | Google Gemini API key for direct endpoint access |
| `GOOGLE_STRONG_MODEL` | `gemini-2.5-flash` | Gemini model for strong reasoning path |
| `ANTHROPIC_API_KEY` | Empty string | Anthropic Claude API key for crisp answer synthesis |
| `ANTHROPIC_CRISP_MODEL` | `claude-haiku-4-5-20251001` | Claude model for one line answers |
| `ANTHROPIC_BEST_MODEL` | `claude-opus-4-6` | Highest capability Claude model |

### Running the System

**Option One: One Click Launch (Windows)**

Double click `start.bat` in the project root directory. The launcher will:
1. Verify Python installation and virtual environment activation
2. Check all required dependencies
3. Launch the FastAPI backend server on port 8000
4. Automatically open the browser when the system is ready

**Option Two: Manual Launch**

Start the backend server:
```bash
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Access the frontend interface by navigating to:
```
http://127.0.0.1:8000/frontend/index.html
```

Alternatively, open `frontend/index.html` directly in your web browser.

**Option Three: Using Launcher Script**

```bash
python launcher.py
```

The system is now running and ready to process documents.

---

## API Reference

LMDIS exposes a RESTful API for programmatic access to all system capabilities.

| Method | Endpoint | Description | Request Parameters | Response |
|--------|----------|-------------|-------------------|----------|
| `GET` | `/` | System health check and status | None | System name, active module count, version |
| `GET` | `/documents` | List all uploaded documents | None | Array of document metadata objects |
| `POST` | `/documents/upload` | Upload and process a PDF document | Multipart form data with PDF file | Document ID, processing status |
| `GET` | `/documents/{id}/status` | Check document processing status | Document ID in path | Processing stage, completion percentage |
| `POST` | `/documents/{id}/query` | Query a document with natural language | Document ID in path, query string parameter `query` | Answer with citations, confidence scores, analysis |
| `GET` | `/documents/{id}/suggestions` | Get AI generated follow up questions | Document ID in path | Array of suggested questions |
| `GET` | `/documents/{id}/structure` | Retrieve logical document structure | Document ID in path | Hierarchical structure tree |

**Example Query Request:**

```bash
curl -X POST "http://127.0.0.1:8000/documents/{document_id}/query?query=What%20is%20the%20revenue%20for%20Q4%202025?"
```

**Example Query Response:**

```json
{
  "answer": "The Q4 2025 revenue was $4.2 million, representing a 15 percent increase year over year.",
  "citations": [
    {
      "text": "Q4 2025 revenue reached $4.2M, up 15% YoY",
      "page": 12,
      "confidence": 0.94,
      "information_type": "statistic"
    }
  ],
  "composite_confidence": 0.92,
  "fact_count": 3,
  "evidence_sources": 5,
  "suggested_followups": [
    "What were the main revenue drivers in Q4 2025?",
    "How does Q4 2025 compare to Q3 2025?"
  ]
}
```

---

## Document Processing Pipeline

The document processing pipeline executes through four distinct stages:

**Stage One: Document Ingestion and Parsing**

The pipeline begins when a PDF document is uploaded. Module M1 registers the document with an immutable SHA 256 fingerprint and assigns a unique UUID. Module M2 classifies each page as digital, scanned, or mixed using text density analysis and image region detection heuristics. Module M3 performs lossless content extraction, preserving text with font metadata for digital pages, extracting table structures through pdfplumber for digital documents or PaddleOCR for scanned content, and extracting embedded images. Module M4 normalizes extracted content at the line level, detects paragraph boundaries, and groups table elements into coherent structures.

**Stage Two: Semantic Processing and Graph Construction**

Module M5 detects hierarchical document structure including sections, subsections, headings, and establishes image caption linkages. Module M6 performs component level normalization across paragraphs, tables, and figures to ensure consistent representation. Module M7 constructs the knowledge graph as a NetworkX directed graph with fifteen plus edge types capturing semantic relationships (references, definitions, elaborations), structural hierarchies (contains, part of), and cross references. Module M8 annotates each graph node with OCR confidence scores and flags low confidence extractions. Module M9 generates 1024 dimensional contextual embeddings using BAAI bge large en v1.5 and constructs a FAISS index for efficient similarity search. Module M9b enriches the graph with semantic metadata including information type classification across fourteen categories (facts, definitions, statistics, methodologies, results), density scoring, section summaries, and cross link generation.

**Stage Three: Knowledge Extraction and Evidence Graph**

Module M10 performs rule based claim extraction identifying dates, monetary values, percentages, definitions, and specifications from the document content. Module M11 integrates claim nodes into the evidence graph, maps inter claim relationships (supports, contradicts, elaborates), and generates section level evidence summaries. The LLM Knowledge Store module generates direct LLM comprehension of the document and stores it in the graph as a backup knowledge source for query resolution.

**Stage Four: Indexing and Storage**

All processed data is serialized and stored in the per document storage directory. The knowledge graph is saved as JSON, embeddings and FAISS indices are serialized as Pickle files, and document metadata including filename, checksum, page count, and title are stored in JSON format. Original uploaded files and extracted images are preserved in their respective subdirectories.

---

## Query and Retrieval System

The query pipeline implements a sophisticated multi stage retrieval and dual path answer generation architecture:

**Query Interpretation (M12)**

Natural language queries are analyzed for intent detection, decomposed into sub claims when necessary, and key terms are extracted for retrieval. The query representation captures semantic intent, entity mentions, and temporal constraints.

**Multi Strategy Retrieval (M13 to M15)**

The retrieval pipeline operates through three sequential stages. First, bi encoder search identifies candidate evidence through embedding similarity against the FAISS index. Second, graph traversal explores the knowledge graph across ten plus edge types to discover related evidence and contextual information. Third, cross encoder re ranking using ms marco MiniLM L 12 v2 precisely scores and ranks the retrieved evidence set for maximum relevance.

**Dual Path Answer Generation**

The system generates answers through two independent paths. Path A (M16 to M19) produces evidence bound answers grounded in the retrieved document evidence with multi signal citation attribution. Path B queries the LLM Knowledge Store for direct LLM reasoning about the document content. This dual path architecture ensures both factual grounding and comprehensive reasoning.

**Answer Selection Layer (ASL)**

The Answer Selection Layer performs fact extraction from both paths, applies multi signal scoring based on semantic alignment, evidence support, and factual consistency, merges complementary facts, and uses LLM based reframing to produce the final answer. This arbitration mechanism prevents hallucinations while maximizing answer completeness.

---

## Citation Engine

The citation engine provides transparent and verifiable source attribution for every claim in the generated answer. Citations are constructed from the cross encoder re ranked evidence pipeline rather than simple keyword matching, ensuring high precision source linking.

**Composite Confidence Score Calculation**

Each citation carries a composite confidence score computed as a weighted combination of four signals:

- Semantic Alignment (40 percent): Cosine similarity between the citation text and the query intent vector
- OCR Quality (30 percent): Confidence score from the OCR extraction process, reflecting text recognition accuracy
- Cross Encoder Relevance (20 percent): Relevance score from the ms marco MiniLM cross encoder re ranking stage
- Contribution Verification (10 percent): Whether the M16 to M19 generation module independently identified this evidence as contributing to the answer

**Multi Signal Citation Attribution**

The citation selection in M16 to M19 employs four complementary attribution signals:

- Unigram Overlap (25 percent): Content word matching between generated answer and evidence text
- Bigram Overlap (30 percent): Phrase level attribution capturing multi word concepts
- High Value Token Overlap (25 percent): Matching of numbers, proper nouns, and abbreviations that carry high information content
- Retrieval Alignment Prior (20 percent): Score inherited from the evidence retrieval pipeline

**Citation Metadata**

Every citation includes:
- Source text excerpt with page number reference
- Composite confidence score (0.0 to 1.0)
- Information type classification (fact, definition, statistic, methodology, result, etc.)
- Alignment score from the retrieval pipeline
- Contribution flag indicating independent attribution by the generation module

---

## Storage Architecture

LMDIS implements per document isolated storage ensuring complete separation between documents and enabling individual document management without cross contamination.

**Storage Structure**

```
app/storage/documents/
├── a33a6c35-db7b-4aed-afe7-ba92a8b3f5ef/
│   ├── graph.json              # Complete knowledge graph with nodes and edges
│   ├── embeddings.pkl          # FAISS index and vector ID mappings
│   ├── metadata.json           # Filename, SHA-256 checksum, page count, title
│   ├── raw/
│   │   └── a33a6c35-db7b-4aed-afe7-ba92a8b3f5ef.pdf  # Original uploaded file
│   └── images/
│       ├── a33a6c35...pdf_p1.png    # Page render images
│       ├── a33a6c35...pdf_p2.png
│       ├── img_a33a6c35...pdf_p1_0.jpeg  # Extracted image regions
│       └── img_a33a6c35...pdf_p2_0.jpeg
└── b7f1e2c4-8a91-4d3f-b2e6-c5d8f1a3e7b9/
    └── ...
```

**Storage Components**

- `graph.json`: Complete NetworkX graph serialized to JSON format containing all nodes (paragraphs, tables, figures, claims, sections) and edges (fifteen plus relationship types)
- `embeddings.pkl`: Pickle serialized FAISS IndexFlatIP index with cosine similarity metric, plus node ID to vector index mappings
- `metadata.json`: Document metadata including original filename, SHA 256 cryptographic checksum, page count, detected title, processing timestamps, and module completion status
- `raw/`: Immutable copy of the original uploaded PDF file for reference and reprocessing
- `images/`: Extracted images including full page renders (PNG format) and embedded image regions extracted from the document (JPEG or PNG format)

**Storage Benefits**

- Complete document isolation prevents cross contamination
- Individual documents can be deleted without affecting others
- Enables parallel processing of multiple documents
- Simplifies backup and archival strategies
- Supports document level access control in enterprise deployments

---

## Frontend Interface

The LMDIS frontend is a single page application featuring a modern glassmorphic dark UI design with three primary views:

**Upload View**

Provides drag and drop PDF upload functionality with real time processing status visualization. Users can monitor processing progress through each module stage, view completion percentages, and access processing logs. Multiple documents can be uploaded and processed concurrently.

**Chat View**

Natural language question and answer interface with citation backed responses. The chat interface displays generated answers with inline citation markers, expandable citation details showing source text and page numbers, composite confidence scores for each response, and suggested follow up questions for exploratory analysis. The conversation history is maintained within the session for contextual queries.

**Analysis View**

Comprehensive per query analysis dashboard providing transparent insight into the answer generation process:

- **Fact Breakdown**: Individual facts extracted from the document with source attribution, grounding status (evidence supported or LLM inferred), and confidence scores
- **Evidence Graph**: Interactive visualization of retrieved evidence relationships showing how different document components connect and support the answer
- **Processing Pipeline**: Visual representation of which modules contributed to the answer (M12 Query Interpretation through M13 to M15 Retrieval, M16 to M19 Generation, ASL Answer Selection, and M20 Formatting)
- **Confidence Scoring**: Multi factor confidence breakdown displaying semantic alignment score, OCR quality assessment, cross encoder relevance rating, and source grounding verification

---

## Testing and Validation

LMDIS includes comprehensive test suites for validating pipeline integrity and component functionality:

**Integration Tests**

```bash
python test_pipeline.py
```

Validates end to end pipeline execution including graph construction, evidence extraction, and answer selection logic. Tests verify that documents are correctly processed through all twenty one modules and that query responses are properly generated with citations.

**Transformer Pipeline Tests**

```bash
python test_transformer_pipeline.py
```

Validates embedding generation, semantic enrichment, and retrieval pipeline components. Tests verify that embeddings are correctly generated, FAISS indices are properly constructed, semantic enrichment classifies information types accurately, and the retrieval pipeline returns relevant evidence.

**Running Tests**

Execute tests after installation to verify system integrity:
```bash
# Activate virtual environment first
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # macOS/Linux

# Run integration tests
python test_pipeline.py

# Run transformer tests
python test_transformer_pipeline.py
```

All tests should pass with successful status messages. If any tests fail, verify that all dependencies are correctly installed and that API keys are properly configured.

---

## Performance Characteristics

**Processing Time**

Document processing time varies based on document complexity and length:
- Simple digital PDFs (10 to 50 pages): 30 to 90 seconds
- Complex digital PDFs with tables and figures (50 to 200 pages): 2 to 5 minutes
- Scanned documents requiring OCR (10 to 50 pages): 2 to 4 minutes
- Large scanned documents (50 to 200 pages): 5 to 15 minutes

**Query Response Time**

Query response times depend on document size and query complexity:
- Simple factual queries: 3 to 8 seconds
- Complex analytical queries requiring extensive retrieval: 8 to 15 seconds
- Multi aspect queries with graph traversal: 10 to 20 seconds

**Resource Utilization**

- Memory: Embedding model loading requires approximately 1.3 GB RAM. Additional memory scales with document complexity (approximately 50 to 200 MB per 100 pages)
- Storage: Processed documents require approximately 10 to 50 MB per 100 pages depending on image content and graph complexity
- CPU: Graph construction and embedding generation are CPU intensive. Multi core systems see significant performance improvements
- GPU: Optional GPU acceleration for embedding generation and cross encoder re ranking (automatic if CUDA capable GPU detected)

**Scalability**

The per document storage architecture enables horizontal scaling through document level sharding. Multiple documents can be processed in parallel, and the FAISS index supports efficient retrieval even with large document collections. For enterprise deployments, consider implementing document level access control and distributed storage backends.

---

## Use Cases

**Legal and Compliance Document Analysis**

Analyze contracts, regulatory filings, and compliance documents with precise citation tracking. Extract obligations, deadlines, monetary terms, and compliance requirements with verifiable source attribution.

**Financial Auditing and Reporting**

Process financial statements, audit reports, and annual filings to extract financial metrics, performance indicators, and comparative analysis. Generate evidence backed answers for audit queries with complete traceability.

**Healthcare Record Processing**

Analyze medical records, clinical trial reports, and research publications while maintaining data integrity. Extract patient information, treatment protocols, outcomes, and statistical findings with accurate citations.

**Research and Academic Document Analysis**

Process research papers, dissertations, and academic publications for literature review and knowledge synthesis. Extract methodologies, results, conclusions, and cross references with academic rigor.

**Enterprise Knowledge Management**

Build searchable knowledge bases from corporate documentation, standard operating procedures, technical manuals, and policy documents. Enable employees to query organizational knowledge with source verified answers.

---

## Limitations

The current implementation has the following known limitations:

**Computational Requirements**

Graph construction and semantic indexing are computationally intensive operations. Large documents (over 200 pages) or documents with complex layouts may require significant processing time and memory resources. Systems with limited RAM (below 4 GB) may experience performance degradation during embedding model loading.

**OCR Quality Dependency**

Processing quality for scanned documents depends on input document quality. Documents with poor scan quality, handwritten content, unusual fonts, or degraded text may produce lower OCR confidence scores. The system flags low confidence extractions, but accuracy cannot be guaranteed for severely degraded documents.

**Single Document Scope**

The current implementation is optimized for single document workflows. Each document is processed and queried independently. Multi document reasoning, cross document linking, and comparative analysis across multiple documents are planned for future releases.

**Model Download Requirements**

The embedding model (BAAI bge large en v1.5) and cross encoder model (ms marco MiniLM L 12 v2) are downloaded on first run and require internet connectivity. Model files occupy approximately 1.5 GB of disk space. Offline deployment requires pre downloading model weights.

---

## Roadmap

The following features and improvements are planned for future releases:

**Multi Document Reasoning**

Implement cross document linking and reasoning capabilities enabling comparative analysis, aggregation of information across multiple documents, and knowledge synthesis from document collections.

**Performance Optimization**

Optimize graph construction algorithms for large scale documents, implement incremental processing for document updates, and reduce memory footprint for deployment on resource constrained systems.

**Enhanced OCR Capabilities**

Integrate advanced OCR models for improved handling of noisy documents, support for handwritten text recognition, and enhanced table structure detection for complex layouts.

**Real Time Processing**

Implement streaming document processing for real time analysis, enable progressive query answering during document processing, and support live document collaboration features.

**Enterprise Features**

Add role based access control, audit logging, integration with enterprise identity providers, API rate limiting, and deployment automation for cloud environments.

**Model Flexibility**

Support pluggable embedding models, enable custom cross encoder fine tuning, and provide model selection interfaces for different use case requirements.

---

## Contributing

Contributions to LMDIS are welcome. Please follow these guidelines:

1. Fork the repository and create a feature branch from the `main` branch
2. Ensure all existing tests pass before submitting changes
3. Add tests for new functionality
4. Follow the existing code style and documentation standards
5. Submit a pull request with a clear description of changes and motivation

For major changes, please open an issue first to discuss the proposed modification.

---

## License and Copyright

This project is licensed under the **GNU General Public License Version 3 (GPL 3.0)**.

You are free to use, modify, and distribute this software under the terms of the GNU GPL 3.0 license. The license ensures that derivative works remain open source and that users retain the freedom to run, study, share, and modify the software.

For complete license text, refer to the `LICENSE` file included with this distribution or visit https://www.gnu.org/licenses/gpl-3.0.html

**Copyright Notice**

Copyright (C) 2026 Sachin S and Ayush Raj

This program is free software: you can redistribute it and or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see https://www.gnu.org/licenses/.

**Trademarks and Attributions**

All trademarks, service marks, and product names referenced in this project are the property of their respective owners. References to third party products, services, or APIs are for informational purposes only and do not imply endorsement or affiliation.

**Academic Use**

LMDIS was developed as a capstone project demonstrating advanced document intelligence capabilities. Academic use, research applications, and educational deployments are encouraged. If you use this software in academic research, please cite the repository.

---

## Contact

For questions, feature requests, bug reports, or collaboration inquiries, please reach out through the following channels:

**Primary Contacts**

- **Sachin S**: sachin.shiva1612@gmail.com
- **Ayush Raj**: ayushraj0901@gmail.com

**GitHub Repository**

- Repository: https://github.com/nihcastics/LMDIS-with-Graph-Guided-Evidence-Aware-Reasoning
- Issues: https://github.com/nihcastics/LMDIS-with-Graph-Guided-Evidence-Aware-Reasoning/issues
- Discussions: https://github.com/nihcastics/LMDIS-with-Graph-Guided-Evidence-Aware-Reasoning/discussions

---

<p align="center">
  <strong>LMDIS: Graph Guided Evidence Aware Reasoning for Enterprise Document Intelligence</strong>
</p>

<p align="center">
  Built with precision. Grounded in evidence. Designed for trust.
</p>

<p align="center">
  Copyright (C) 2026 Sachin S and Ayush Raj. Licensed under GPL 3.0.
</p>

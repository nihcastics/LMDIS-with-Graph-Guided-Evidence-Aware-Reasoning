from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
import asyncio
import shutil
import os
import networkx as nx
import pickle
from concurrent.futures import ThreadPoolExecutor

# --- Module Imports ---
from backend.app.modules.module1_ingestion import ingest_document
from backend.app.modules.module2_page_type import analyze_page_types
from backend.app.modules.module3_extraction import extract_content
from backend.app.modules.module4_line_memory import normalize_lines
from backend.app.modules.module5_structure import detect_structure
from backend.app.modules.module6_normalization import normalize_components
from backend.app.modules.module7_graph import build_graph
from backend.app.modules.module8_confidence import annotate_reliability
from backend.app.modules.module9_embedding import EmbeddingManager
from backend.app.modules.module9b_semantic_enrichment import enrich_semantics
from backend.app.modules.module10_claims import extract_claims
from backend.app.modules.module11_evidence_graph import build_evidence_graph
from backend.app.modules.module12_query import interpret_query
from backend.app.modules.module13_14_15_retrieval import retrieve_evidence
from backend.app.modules.module16_19_generation import generate_answer
from backend.app.modules.module20_formatter import format_response
from backend.app.modules.module21_suggestions import generate_questions
from backend.app.modules.module_llm_knowledge import store_llm_knowledge, query_llm_directly
from backend.app.modules.module_answer_selection import select_and_merge
from backend.app.modules.module_graph_semantic_alignment import align_graph_semantics
from backend.app.utils.persistent_storage import PersistentStorage

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Persistent Storage — per-document directory structure
storage = PersistentStorage("app/storage")

# In-memory cache for active sessions
GRAPH_CACHE = {}
EMBED_CACHE = {}

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

@app.get("/app", response_class=HTMLResponse, include_in_schema=False)
async def serve_frontend():
    """Serve the frontend single-page application."""
    frontend_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "frontend", "index.html",
    )
    if not os.path.exists(frontend_path):
        raise HTTPException(status_code=404, detail="Frontend not found")
    with open(frontend_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/")
async def root():
    return {
        "status": "System is running - v1.0 Completed", 
        "endpoints": ["/upload", "/query", "/docs", "/documents"],
        "modules_active": 21
    }

@app.get("/documents")
async def list_documents():
    """List all uploaded documents"""
    doc_ids = storage.list_documents()
    documents = []
    for doc_id in doc_ids:
        meta = storage.load_metadata(doc_id)
        if meta:
            documents.append(meta)
    return {"documents": documents, "count": len(documents)}

def _process_upload_sync(content, filename):
    """Heavy sync processing — runs in a thread to keep the event loop free."""
    import time as _time
    _t_start = _time.perf_counter()

    # --- PHASE 1: Ingestion & Core Processing (Modules 1-7) ---

    # M1: Ingestion — save raw file into per-document directory
    ingest_meta = ingest_document(content, filename)
    doc_id = ingest_meta["doc_id"]
    local_path = ingest_meta["local_path"]

    # Ensure the raw file lives inside the per-document storage dir
    doc_raw_dir = storage.raw_dir(doc_id)
    expected_path = os.path.join(doc_raw_dir, os.path.basename(local_path))
    if os.path.abspath(local_path) != os.path.abspath(expected_path):
        os.makedirs(doc_raw_dir, exist_ok=True)
        shutil.move(local_path, expected_path)
        local_path = expected_path
        ingest_meta["local_path"] = local_path

    # Per-document image directory
    doc_image_dir = storage.images_dir(doc_id)

    # M2: Page Analysis
    page_types = analyze_page_types(local_path)

    # M3: Extraction — images saved into per-document images/ dir
    raw_extraction = extract_content(local_path, page_types, output_image_dir=doc_image_dir)

    # --- Document Title Detection ---
    first_page_data = raw_extraction.get(1, {})
    text_blocks = first_page_data.get("text_blocks", [])
    if text_blocks:
        top_blocks = [b for b in text_blocks if b["bbox"][1] < 250]
        if top_blocks:
            max_font_size = max(b.get("font_size", 0) for b in top_blocks)
            title_parts = []
            sorted_top = sorted(top_blocks, key=lambda b: b["bbox"][1])
            for b in sorted_top:
                if b.get("font_size", 0) >= max_font_size - 1:
                    text = b["text"].strip()
                    if len(text) > 2 and text not in title_parts:
                        title_parts.append(text)
                elif title_parts:
                    break
            if title_parts:
                full_title = " ".join(title_parts)
                ingest_meta["detected_title"] = full_title[:250]

    # M4: Line Normalization (returns lines + document elements for paragraphs/tables)
    lines, document_elements = normalize_lines(raw_extraction, doc_id)

    # M5: Structure
    all_images = []
    for p in raw_extraction.values():
        all_images.extend(p["images"])
    sections, sorted_lines, linked_images = detect_structure(lines, all_images, doc_id)

    # M6: Normalization (with paragraph + cell-level table components)
    normalized_data = normalize_components(sorted_lines, linked_images, sections, document_elements)

    # M7: Graph Construction
    G = build_graph(normalized_data, doc_meta=ingest_meta)

    # --- PHASE 2: Semantics & Logic (Modules 8-11) ---

    # M8: Confidence
    G, low_conf_nodes = annotate_reliability(G)

    # M9: Embeddings
    embed_mgr = EmbeddingManager()
    G = embed_mgr.generate_embeddings(G)

    # M9b: Semantic Enrichment (information types, density, summaries, cross-links)
    G = enrich_semantics(G, embedding_manager=embed_mgr)

    # M10: Claims
    claims = extract_claims(G)

    # M11: Evidence Graph
    G = build_evidence_graph(G, claims)

    # GSA: LLM-Based Graph Semantic Alignment (arranges graph logically)
    G = align_graph_semantics(G, doc_meta=ingest_meta)

    # LLM-K: Direct LLM Document Knowledge Store (backup path)
    G = store_llm_knowledge(G, doc_meta=ingest_meta)

    # Save to Persistent Storage
    storage.save_graph(doc_id, G)
    storage.save_embeddings(doc_id, embed_mgr)

    # Record processing time
    _t_end = _time.perf_counter()
    processing_time_sec = round(_t_end - _t_start, 2)
    ingest_meta["processing_time_sec"] = processing_time_sec

    storage.save_metadata(doc_id, ingest_meta)

    # Cache in memory for fast access
    GRAPH_CACHE[doc_id] = G
    EMBED_CACHE[doc_id] = embed_mgr

    return {
        "status": "success",
        "doc_id": doc_id,
        "metadata": ingest_meta,
        "stats": {
            "pages": len(page_types),
            "lines": len(lines),
            "claims": len(claims)
        },
        "processing_time_sec": processing_time_sec,
    }


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        content = await file.read()
        # Run heavy ML processing in a thread so the event loop stays alive
        result = await asyncio.to_thread(_process_upload_sync, content, file.filename)
        return result
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/{doc_id}/status")
async def get_document_status(doc_id: str):
    """Check processing status of a document"""
    meta = storage.load_metadata(doc_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"doc_id": doc_id, "status": "processed", "metadata": meta}

def _process_query_sync(doc_id, query):
    """Heavy sync query processing — runs in a thread."""
    import time as _time
    _t_start = _time.perf_counter()

    # Try cache first
    G = GRAPH_CACHE.get(doc_id)
    embed_mgr = EMBED_CACHE.get(doc_id)

    # If not in cache, load from storage
    if G is None:
        G = storage.load_graph(doc_id)
        if G is None:
            return None  # signal 404
        GRAPH_CACHE[doc_id] = G

    if embed_mgr is None:
        from backend.app.modules.module9_embedding import EmbeddingManager
        embed_mgr = EmbeddingManager()
        embed_mgr = storage.load_embeddings(doc_id, embed_mgr)
        if embed_mgr is None:
            return "embeddings_missing"  # signal 404
        EMBED_CACHE[doc_id] = embed_mgr

    # --- PHASE 3: Query & Retrieval (Modules 12-15) ---

    # M12: Interpretation
    structured_query = interpret_query(query)

    # M13-15: Retrieval
    aligned_evidence = retrieve_evidence(G, embed_mgr, structured_query)

    # --- PHASE 4: Dual-Path Answer Generation & Selection ---
    doc_meta = storage.load_metadata(doc_id)

    # Path A and Path B run concurrently — they are fully independent
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_ev = executor.submit(generate_answer, structured_query, aligned_evidence, doc_meta)
        future_llm = executor.submit(query_llm_directly, G, query, doc_meta)
        ev_answer = future_ev.result()
        llm_direct = future_llm.result()

    # Answer Selection & Compensation Layer: merge facts, pick best answer
    merged_result = select_and_merge(
        query,
        ev_answer,
        llm_direct,
        doc_meta,
        aligned_evidence=aligned_evidence,
        graph=G,
    )

    # M20: Formatting (with full analysis payload)
    final_response = format_response(
        merged_result,
        G,
        aligned_evidence=aligned_evidence,
        ev_answer=ev_answer,
        llm_direct=llm_direct,
        structured_query=structured_query,
    )

    _t_end = _time.perf_counter()
    final_response["query_time_sec"] = round(_t_end - _t_start, 2)

    return final_response


@app.post("/documents/{doc_id}/query")
async def query_document(doc_id: str, query: str):
    result = await asyncio.to_thread(_process_query_sync, doc_id, query)
    if result is None:
        raise HTTPException(status_code=404, detail="Document not found")
    if result == "embeddings_missing":
        raise HTTPException(status_code=404, detail="Document embeddings not found")
    return result

@app.get("/documents/{doc_id}/suggestions")
async def get_suggestions(doc_id: str):
    """Generate suggested questions based on the document"""
    G = storage.load_graph(doc_id)
    if not G:
        raise HTTPException(status_code=404, detail="Document not found")
    
    doc_meta = storage.load_metadata(doc_id)
    
    # Get some text components for summary
    text_samples = []
    count = 0
    for node, data in G.nodes(data=True):
        if data.get("type") == "text_component" and count < 15:
            text_samples.append(data.get("text", ""))
            count += 1
    
    summary_text = "\n".join(text_samples)
    questions = generate_questions(summary_text, doc_meta)
    
    return {"suggestions": questions}

@app.get("/documents/{doc_id}/structure")
async def get_document_structure(doc_id: str):
    """Retrieve the logical structure and memory of the document for debugging/viewing"""
    G = storage.load_graph(doc_id)
    if not G:
        raise HTTPException(status_code=404, detail="Document not found")
    
    sections = []
    # Find all section nodes
    section_nodes = [(n, d) for n, d in G.nodes(data=True) if d.get("type") == "section"]
    # Sort by page
    section_nodes.sort(key=lambda x: x[1].get("page", 0))
    
    for s_id, s_data in section_nodes:
        # Find all lines belonging to this section
        lines = []
        for n, d in G.nodes(data=True):
            if d.get("type") == "text_component" and G.has_edge(n, s_id):
                lines.append({
                    "text": d.get("text", ""),
                    "page": d.get("page", 0),
                    "confidence": d.get("confidence", 1.0)
                })
        
        # Sort lines by page and then by their own internal order (NEXT_LINE edges)
        # For simplicity in this view, we'll just sort by page and Y
        # In a real graph we'd traverse NEXT_LINE.
        lines.sort(key=lambda l: l["page"])
        
        sections.append({
            "id": s_id,
            "title": s_data.get("label", "Untitled Section"),
            "page": s_data.get("page", 0),
            "line_count": len(lines),
            "lines": lines[:50] # Limit for UI performance
        })
        
    return {"doc_id": doc_id, "sections": sections}

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document and all its associated data (graph, embeddings, raw files)."""
    meta = storage.load_metadata(doc_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Document not found")
    # Clear from in-memory caches first
    GRAPH_CACHE.pop(doc_id, None)
    EMBED_CACHE.pop(doc_id, None)
    # Remove entire document directory from disk
    storage.delete_document(doc_id)
    return {"status": "deleted", "doc_id": doc_id}

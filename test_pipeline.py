"""Quick integration test for the new graph-based pipeline."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import networkx as nx
from backend.app.modules.module7_graph import build_graph
from backend.app.modules.module11_evidence_graph import build_evidence_graph
from backend.app.modules.module_answer_selection import select_and_merge

# ── Test 1: build_graph ──
norm = {
    "sections": [{"section_id": "s1", "title": "Introduction", "page_start": 1, "lines": ["l1", "l2"]}],
    "text_components": [
        {"line_id": "l1", "text": "Machine learning is defined as a method of data analysis.",
         "section_id": "s1", "page": 1, "reading_order": 0, "type": "text_component"},
        {"line_id": "l2", "text": "Deep learning uses neural networks for feature extraction.",
         "section_id": "s1", "page": 1, "reading_order": 1, "type": "text_component"},
    ],
    "image_components": [], "table_components": [], "paragraphs": [], "paragraph_components": [],
}
meta = {"doc_id": "test123", "detected_title": "ML Paper"}
G = build_graph(norm, doc_meta=meta)
print(f"[OK] Graph nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")
edge_types = set(d.get("relationship") for _, _, d in G.edges(data=True))
print(f"[OK] Edge types: {edge_types}")

# ── Test 2: evidence graph ──
claims = [
    {"claim_id": "c1", "value": "machine learning = method of data analysis",
     "type": "fact_definition", "source_lines": ["l1"]},
    {"claim_id": "c2", "value": "deep learning uses neural networks",
     "type": "fact", "source_lines": ["l2"]},
]
G = build_evidence_graph(G, claims)
ev_nodes = [n for n, d in G.nodes(data=True)
            if d.get("type") in ("claim", "evidence_summary")]
print(f"[OK] Evidence nodes added: {len(ev_nodes)}")
assert len(ev_nodes) >= 2, "Expected at least 2 claim nodes"

# ── Test 3: answer selection (both paths available) ──
ev_ans = {"answer_text": "Machine learning is a subfield of AI.",
          "contributing_lines": ["l1"]}
llm_ans = {"answer": "ML is an AI subfield that uses data.",
           "facts": ["ML is AI subfield", "Uses data for learning"]}
result = select_and_merge("What is machine learning?", ev_ans, llm_ans, meta)
print(f"[OK] Selection source: {result['source_path']}")
print(f"[OK] Answer length: {len(result['answer_text'])}")
assert result["source_path"] in ("merged", "evidence_graph", "llm_direct")

# ── Test 4: answer selection (only evidence path) ──
result2 = select_and_merge("What is ML?", ev_ans, None, meta)
assert result2["source_path"] == "evidence_graph"
print("[OK] Evidence-only fallback works")

# ── Test 5: answer selection (only LLM path) ──
result3 = select_and_merge("What is ML?", None, llm_ans, meta)
assert result3["source_path"] == "llm_direct"
print("[OK] LLM-only fallback works")

# ── Test 6: both paths empty ──
result4 = select_and_merge("What is ML?", None, None, meta)
assert result4["source_path"] == "none"
print("[OK] Empty fallback works")

print("\n=== ALL INTEGRATION TESTS PASSED ===")

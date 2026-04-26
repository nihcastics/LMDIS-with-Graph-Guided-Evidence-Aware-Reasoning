"""
Test: Transformer Semantic Enrichment Pipeline
Verifies Module 9 (deep contextual embeddings) and Module 9b (semantic enrichment)
work correctly with a synthetic document graph.
"""

import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

import networkx as nx

# ── Build a realistic synthetic document graph ──
def build_test_graph():
    G = nx.DiGraph()

    # Document node
    G.add_node("doc_1", type="document", filename="test.pdf")

    # Sections
    G.add_node("sec_intro", type="section", label="Introduction", page=1)
    G.add_node("sec_method", type="section", label="Methodology", page=2)
    G.add_node("sec_results", type="section", label="Results", page=3)

    # Lines in Introduction
    intro_lines = [
        ("line_1", "Machine learning is a subset of artificial intelligence.", 1, [72, 100, 500, 115]),
        ("line_2", "Deep learning refers to neural networks with multiple hidden layers.", 1, [72, 120, 500, 135]),
        ("line_3", "Transfer learning leverages pre-trained models for downstream tasks.", 1, [72, 140, 500, 155]),
        ("line_4", "Recent advances have shown 95% accuracy on benchmark datasets.", 1, [72, 160, 500, 175]),
    ]
    for lid, text, page, bbox in intro_lines:
        G.add_node(lid, type="text_component", text=text, page=page, bbox=bbox)
        G.add_edge(lid, "sec_intro", relationship="BELONGS_TO")

    # NEXT_LINE edges
    for i in range(len(intro_lines) - 1):
        G.add_edge(intro_lines[i][0], intro_lines[i+1][0], relationship="NEXT_LINE")

    # Lines in Methodology
    method_lines = [
        ("line_5", "We propose a transformer-based approach for document understanding.", 2, [72, 100, 500, 115]),
        ("line_6", "The model is trained using cross-entropy loss with Adam optimizer.", 2, [72, 120, 500, 135]),
        ("line_7", "Hyperparameters include learning rate of 0.001 and batch size of 32.", 2, [72, 140, 500, 155]),
    ]
    for lid, text, page, bbox in method_lines:
        G.add_node(lid, type="text_component", text=text, page=page, bbox=bbox)
        G.add_edge(lid, "sec_method", relationship="BELONGS_TO")

    for i in range(len(method_lines) - 1):
        G.add_edge(method_lines[i][0], method_lines[i+1][0], relationship="NEXT_LINE")

    # Lines in Results
    result_lines = [
        ("line_8", "The proposed method achieved 97.3% accuracy on the test set.", 3, [72, 100, 500, 115]),
        ("line_9", "Compared to baselines, our approach outperforms by 4.2 percentage points.", 3, [72, 120, 500, 135]),
        ("line_10", "Therefore, we conclude that transformer models are effective for this task.", 3, [72, 140, 500, 155]),
    ]
    for lid, text, page, bbox in result_lines:
        G.add_node(lid, type="text_component", text=text, page=page, bbox=bbox)
        G.add_edge(lid, "sec_results", relationship="BELONGS_TO")

    for i in range(len(result_lines) - 1):
        G.add_edge(result_lines[i][0], result_lines[i+1][0], relationship="NEXT_LINE")

    # Paragraphs
    G.add_node("para_1", type="paragraph", category="body", page=1)
    for lid, _, _, _ in intro_lines:
        G.add_edge("para_1", lid, relationship="PARAGRAPH_CONTAINS")

    G.add_node("para_2", type="paragraph", category="body", page=2)
    for lid, _, _, _ in method_lines:
        G.add_edge("para_2", lid, relationship="PARAGRAPH_CONTAINS")

    # Image node
    G.add_node("img_1", type="image_component", page=3,
               llm_description="Bar chart showing accuracy comparison across five models",
               bbox=[72, 200, 500, 400])

    # Table node with cells
    G.add_node("table_1", type="table", page=3, bbox=[72, 420, 500, 520])
    G.add_node("cell_1_0_0", type="text_component", text="Model", row=0, col=0, is_header=True, page=3, bbox=[72,420,200,440])
    G.add_node("cell_1_0_1", type="text_component", text="Accuracy", row=0, col=1, is_header=True, page=3, bbox=[200,420,350,440])
    G.add_node("cell_1_1_0", type="text_component", text="BERT", row=1, col=0, page=3, bbox=[72,440,200,460])
    G.add_node("cell_1_1_1", type="text_component", text="93.1%", row=1, col=1, page=3, bbox=[200,440,350,460])
    G.add_edge("table_1", "cell_1_0_0", relationship="TABLE_CELL_OF")
    G.add_edge("table_1", "cell_1_0_1", relationship="TABLE_CELL_OF")
    G.add_edge("table_1", "cell_1_1_0", relationship="TABLE_CELL_OF")
    G.add_edge("table_1", "cell_1_1_1", relationship="TABLE_CELL_OF")

    return G


def test_module9_embeddings():
    """Test that Module 9 generates contextual embeddings for all element types."""
    print("=" * 60)
    print("TEST: Module 9 — Deep Contextual Embedding Pipeline")
    print("=" * 60)

    from backend.app.modules.module9_embedding import EmbeddingManager, EMBEDDING_MODEL_NAME, EMBEDDING_DIM

    G = build_test_graph()
    mgr = EmbeddingManager()

    print(f"  Model: {EMBEDDING_MODEL_NAME}")
    print(f"  Dimension: {EMBEDDING_DIM}")
    assert EMBEDDING_DIM == 1024, f"Expected 1024-dim, got {EMBEDDING_DIM}"
    assert mgr.dimension == 1024

    G = mgr.generate_embeddings(G)

    # Verify embeddings were created
    embedded_nodes = [n for n, d in G.nodes(data=True) if "embedding_id" in d]
    print(f"  Embedded nodes: {len(embedded_nodes)}")
    assert len(embedded_nodes) >= 10, f"Expected ≥10 embedded nodes, got {len(embedded_nodes)}"

    # Verify FAISS index size matches
    assert mgr.index.ntotal == len(embedded_nodes), \
        f"FAISS ntotal={mgr.index.ntotal} != embedded nodes={len(embedded_nodes)}"

    # Test semantic search
    results = mgr.search("transformer approach for document understanding", k=5)
    assert len(results) > 0, "Search returned no results"
    print(f"  Search test: {len(results)} results returned")
    for node_id, dist in results[:3]:
        text = G.nodes.get(node_id, {}).get("text", G.nodes.get(node_id, {}).get("label", "???"))
        print(f"    → {node_id}: dist={dist:.4f} | {text[:60]}...")

    # Verify section and paragraph embeddings exist
    section_embedded = [n for n, d in G.nodes(data=True) if d.get("type") == "section" and "embedding_id" in d]
    para_embedded = [n for n, d in G.nodes(data=True) if d.get("type") == "paragraph" and "embedding_id" in d]
    img_embedded = [n for n, d in G.nodes(data=True) if d.get("type") == "image_component" and "embedding_id" in d]
    table_embedded = [n for n, d in G.nodes(data=True) if d.get("type") == "table" and "embedding_id" in d]

    print(f"  Sections embedded: {len(section_embedded)}")
    print(f"  Paragraphs embedded: {len(para_embedded)}")
    print(f"  Images embedded: {len(img_embedded)}")
    print(f"  Tables embedded: {len(table_embedded)}")

    assert len(section_embedded) >= 1, "No section embeddings generated"
    assert len(para_embedded) >= 1, "No paragraph embeddings generated"
    assert len(img_embedded) >= 1, "No image description embeddings generated"

    print("  ✓ Module 9 — ALL CHECKS PASSED\n")
    return G, mgr


def test_module9b_enrichment(G, mgr):
    """Test that Module 9b enriches the graph with semantic metadata."""
    print("=" * 60)
    print("TEST: Module 9b — Semantic Enrichment Pipeline")
    print("=" * 60)

    from backend.app.modules.module9b_semantic_enrichment import enrich_semantics, INFORMATION_TYPES

    G = enrich_semantics(G, embedding_manager=mgr)

    # 1. Check information_type assigned to text components
    text_nodes = [(n, d) for n, d in G.nodes(data=True) if d.get("type") == "text_component"]
    classified = [(n, d) for n, d in text_nodes if "information_type" in d]
    print(f"  Text nodes classified: {len(classified)}/{len(text_nodes)}")
    assert len(classified) == len(text_nodes), "Not all text nodes got information_type"

    for n, d in classified:
        itype = d["information_type"]
        assert itype in INFORMATION_TYPES, f"Invalid type '{itype}' on {n}"
        print(f"    {n}: {itype} | {d.get('text', '')[:50]}")

    # 2. Check semantic_density on text nodes
    density_nodes = [(n, d) for n, d in text_nodes if "semantic_density" in d]
    print(f"\n  Nodes with semantic_density: {len(density_nodes)}/{len(text_nodes)}")
    assert len(density_nodes) == len(text_nodes), "Not all text nodes got semantic_density"
    for n, d in density_nodes:
        sd = d["semantic_density"]
        assert 0.0 <= sd <= 1.0, f"Density {sd} out of range on {n}"

    # 3. Check paragraph aggregation
    para_nodes = [(n, d) for n, d in G.nodes(data=True) if d.get("type") == "paragraph"]
    for n, d in para_nodes:
        if "dominant_info_type" in d:
            print(f"  Paragraph {n}: dominant={d['dominant_info_type']}, avg_density={d.get('avg_semantic_density', '?')}")

    # 4. Check section aggregation
    sec_nodes = [(n, d) for n, d in G.nodes(data=True) if d.get("type") == "section"]
    for n, d in sec_nodes:
        if "dominant_info_type" in d:
            print(f"  Section {n}: dominant={d['dominant_info_type']}, avg_density={d.get('avg_semantic_density', '?')}")
        if "key_concepts" in d:
            print(f"    Key concepts: {d['key_concepts']}")
        if "semantic_summary" in d:
            print(f"    Summary: {d['semantic_summary'][:80]}...")

    # 5. Check table/image classification
    for n, d in G.nodes(data=True):
        if d.get("type") == "table" and "table_type" in d:
            print(f"  Table {n}: table_type={d['table_type']}")
        if d.get("type") == "image_component" and "image_type" in d:
            print(f"  Image {n}: image_type={d['image_type']}")

    # 6. Check SEMANTICALLY_RELATED edges
    sem_edges = [(u, v, d) for u, v, d in G.edges(data=True) if d.get("relationship") == "SEMANTICALLY_RELATED"]
    print(f"\n  SEMANTICALLY_RELATED edges: {len(sem_edges)}")
    for u, v, d in sem_edges[:5]:
        print(f"    {u} → {v} (similarity={d.get('similarity', '?')})")

    print("  ✓ Module 9b — ALL CHECKS PASSED\n")
    return G


def test_retrieval_integration(G, mgr):
    """Test that the enhanced retrieval module works with enriched data."""
    print("=" * 60)
    print("TEST: Enhanced Retrieval (M13-15) with semantic enrichment")
    print("=" * 60)

    from backend.app.modules.module13_14_15_retrieval import retrieve_evidence

    query = {
        "intent": "factual_lookup",
        "original_text": "What accuracy did the transformer model achieve?",
        "claims": ["transformer model accuracy", "test set performance results"],
        "requires_reasoning": False,
    }

    evidence = retrieve_evidence(G, mgr, query)
    print(f"  Evidence items returned: {len(evidence)}")
    assert len(evidence) > 0, "No evidence returned!"

    for ev in evidence[:5]:
        meta = ev.get("metadata", {})
        print(f"    score={ev['alignment_score']:.3f} | page={meta.get('page')} | "
              f"type={meta.get('information_type', '?')} | {ev['text'][:60]}...")

    # Check that metadata now includes information_type
    has_info_type = any(ev.get("metadata", {}).get("information_type") for ev in evidence)
    print(f"  Evidence includes information_type: {has_info_type}")

    print("  ✓ Retrieval integration — ALL CHECKS PASSED\n")


if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("  LMDIS Transformer Pipeline Test Suite")
    print("═" * 60 + "\n")

    G, mgr = test_module9_embeddings()
    G = test_module9b_enrichment(G, mgr)
    test_retrieval_integration(G, mgr)

    print("═" * 60)
    print("  ALL TESTS PASSED ✓")
    print("═" * 60)

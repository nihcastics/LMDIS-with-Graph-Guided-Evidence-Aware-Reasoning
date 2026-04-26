"""
Module 9: Deep Contextual Transformer Embedding Pipeline
=========================================================
Maximum-accuracy semantic embedding using BAAI/bge-large-en-v1.5 (1024-dim).

Every element is embedded with rich surrounding context:
- Lines: sliding window of prev/next lines + section/paragraph context
- Sections: full hierarchical child text
- Paragraphs: coherent multi-line with section context
- Images: LLM descriptions with spatial context
- Tables: cell-level structured content

Uses normalized cosine similarity (IndexFlatIP) for precise retrieval.
"""

import faiss
import numpy as np
import pickle
import os

os.environ["HF_HUB_READ_TIMEOUT"] = "120"

# ── Strongest available bi-encoder for maximum semantic accuracy ──
# bge-large-en-v1.5: 335M params, 1024-dim, top-tier MTEB ranking
EMBEDDING_MODEL_NAME = 'BAAI/bge-large-en-v1.5'
EMBEDDING_DIM = 1024

# Lazy-loaded singleton — avoids slow import-time model load
_model = None


def _get_model():
    global _model
    if _model is None:
        import logging
        logging.getLogger("transformers.utils.loading_report").setLevel(logging.ERROR)
        from sentence_transformers import SentenceTransformer
        print(f"[M9] Loading embedding model: {EMBEDDING_MODEL_NAME} ...")
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print(f"[M9] Model loaded ({EMBEDDING_DIM}-dim).")
    return _model

# Context window size: each line sees N lines before and after it
CONTEXT_WINDOW = 2


def _build_reading_order(G):
    """Build document reading order from NEXT_LINE edges, page by page."""
    page_lines = {}
    for node, data in G.nodes(data=True):
        if data.get("type") == "text_component":
            page = data.get("page", 1)
            page_lines.setdefault(page, []).append(node)

    ordered = []
    for page in sorted(page_lines.keys()):
        nodes_on_page = set(page_lines[page])
        succ_map = {}
        pred_set = set()
        for node in nodes_on_page:
            for follower in G.successors(node):
                if G.edges[node, follower].get("relationship") == "NEXT_LINE" and follower in nodes_on_page:
                    succ_map[node] = follower
                    pred_set.add(follower)

        heads = [n for n in nodes_on_page if n not in pred_set]
        visited = set()
        for head in sorted(heads, key=lambda n: G.nodes[n].get("bbox", [0, 0, 0, 0])[1]):
            curr = head
            while curr and curr not in visited:
                visited.add(curr)
                ordered.append(curr)
                curr = succ_map.get(curr)

        for node in nodes_on_page:
            if node not in visited:
                ordered.append(node)

    return ordered


def _get_section_title(G, node):
    """Get the section title for a text component node."""
    for target in G.successors(node):
        if G.nodes.get(target, {}).get("type") == "section":
            return G.nodes[target].get("label", "")
    return ""


def _build_contextual_line_text(G, node, ordered_nodes, idx_map):
    """
    Build a context-rich embedding text for a single line.
    Includes: section context + prev lines + CURRENT LINE + next lines.
    Uses BGE instruction prefix for optimal retrieval performance.
    """
    data = G.nodes[node]
    text = data.get("text", "").strip()
    if not text:
        return ""

    idx = idx_map.get(node, -1)

    # Node not in reading order (e.g. table cell embedded separately) — embed bare text
    if idx == -1:
        section_title = _get_section_title(G, node)
        prefix = f"Section: {section_title}. " if section_title else ""
        return f"Represent this document passage for retrieval: {prefix}{text}"

    # ── Gather surrounding lines ──
    prev_parts = []
    for i in range(max(0, idx - CONTEXT_WINDOW), idx):
        prev_text = G.nodes[ordered_nodes[i]].get("text", "").strip()
        if prev_text:
            prev_parts.append(prev_text)

    next_parts = []
    for i in range(idx + 1, min(len(ordered_nodes), idx + 1 + CONTEXT_WINDOW)):
        next_text = G.nodes[ordered_nodes[i]].get("text", "").strip()
        if next_text:
            next_parts.append(next_text)

    # ── Section context ──
    section_title = _get_section_title(G, node)

    # ── Assemble contextual representation ──
    parts = []
    if section_title:
        parts.append(f"Section: {section_title}.")
    if prev_parts:
        parts.append("Previous: " + " ".join(prev_parts) + ".")
    parts.append(text)
    if next_parts:
        parts.append("Next: " + " ".join(next_parts) + ".")

    full_text = " ".join(parts)
    # BGE instruction prefix maximizes retrieval quality
    return f"Represent this document passage for retrieval: {full_text}"


class EmbeddingManager:
    def __init__(self):
        self.dimension = EMBEDDING_DIM
        # IndexFlatIP with normalized vectors = exact cosine similarity
        self.index = faiss.IndexFlatIP(self.dimension)
        self.id_map = {}  # int_id -> str_node_id
        self.counter = 0

    def generate_embeddings(self, G):
        """
        Contextual Embedding Pipeline — fast 3-level semantic encoding.

        Level 1: Context-aware line embeddings (prev/next window + section)
        Level 2: Image description embeddings (LLM-generated context)
        Level 3: Table embeddings (cell-level structured text)

        Section and paragraph embeddings are skipped — line embeddings
        already capture all text content and are sufficient for retrieval.
        """
        nodes_to_embed = []
        texts = []

        # Build reading order for contextual windowing
        ordered = _build_reading_order(G)
        idx_map = {n: i for i, n in enumerate(ordered)}

        # ═══ Level 1: Context-Aware Line Embeddings ═══
        for node in ordered:
            ctx_text = _build_contextual_line_text(G, node, ordered, idx_map)
            if ctx_text:
                nodes_to_embed.append(node)
                texts.append(ctx_text)

        # ═══ Level 2: Image Description Embeddings ═══
        for node, data in G.nodes(data=True):
            if data.get("type") != "image_component":
                continue
            desc = data.get("llm_description", "")
            if not desc:
                continue
            page = data.get("page", 1)
            img_text = f"Represent this image description for retrieval: " \
                       f"[Image on page {page}] {desc}"
            if len(img_text) > 2000:
                img_text = img_text[:2000]
            nodes_to_embed.append(node)
            texts.append(img_text)

        # ═══ Level 3: Table Embeddings ═══
        for node, data in G.nodes(data=True):
            if data.get("type") != "table":
                continue
            page = data.get("page", 1)
            cell_parts = []
            for neighbor in G.successors(node):
                if G.edges[node, neighbor].get("relationship") == "TABLE_CELL_OF":
                    cell_data = G.nodes[neighbor]
                    cell_text = cell_data.get("text", "").strip()
                    if cell_text:
                        row = cell_data.get("row", 0)
                        col = cell_data.get("col", 0)
                        cell_parts.append(f"R{row}C{col}: {cell_text}")
            if cell_parts:
                table_text = f"Represent this table for retrieval: " \
                             f"[Table on page {page}] {'; '.join(cell_parts)}"
                if len(table_text) > 4000:
                    table_text = table_text[:4000]
                nodes_to_embed.append(node)
                texts.append(table_text)

        if not texts:
            return G

        # ── Encode with normalized embeddings for cosine similarity ──
        embeddings = _get_model().encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=128,
        )

        self.index.add(np.array(embeddings).astype('float32'))

        for i, node_id in enumerate(nodes_to_embed):
            self.id_map[self.counter + i] = node_id
            G.nodes[node_id]["embedding_id"] = self.counter + i

        self.counter += len(texts)
        print(f"[M9] Embedded {len(texts)} elements with {EMBEDDING_MODEL_NAME} ({EMBEDDING_DIM}-dim)")
        return G

    def search(self, query_text, k=5):
        """Semantic search with BGE instruction-augmented query."""
        augmented = f"Represent this query for retrieving relevant document content: {query_text}"
        query_vec = _get_model().encode([augmented], normalize_embeddings=True)
        D, I = self.index.search(np.array(query_vec).astype('float32'), k)

        results = []
        for i, idx in enumerate(I[0]):
            if idx == -1:
                continue
            node_id = self.id_map.get(int(idx))
            if node_id:
                # Convert cosine similarity → pseudo-distance for backward compatibility
                similarity = float(D[0][i])
                distance = max(0.0, 1.0 - similarity)
                results.append((node_id, distance))
        return results

    def save(self, path="app/storage/embeddings"):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, f"{path}/index.faiss")
        with open(f"{path}/id_map.pkl", "wb") as f:
            pickle.dump({
                "id_map": self.id_map,
                "counter": self.counter,
                "dimension": self.dimension,
            }, f)

    def load(self, path="app/storage/embeddings"):
        idx_path = f"{path}/index.faiss"
        map_path = f"{path}/id_map.pkl"
        if os.path.exists(idx_path) and os.path.exists(map_path):
            self.index = faiss.read_index(idx_path)
            with open(map_path, "rb") as f:
                state = pickle.load(f)
            self.id_map = state["id_map"]
            self.counter = state["counter"]
            stored_dim = state.get("dimension", 384)
            if stored_dim != self.dimension:
                print(f"[M9] WARNING: Stored embeddings dim={stored_dim} != model dim={self.dimension}. Re-index needed.")
            self.dimension = stored_dim
            return True
        return False
        with open(f"{path}/id_map.pkl", "wb") as f:
            pickle.dump(self.id_map, f)

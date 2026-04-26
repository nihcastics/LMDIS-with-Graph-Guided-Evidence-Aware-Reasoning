import json
import pickle
import os
import shutil
import networkx as nx
from networkx.readwrite import json_graph


class PersistentStorage:
    """
    Per-document file-based persistence.

    Each uploaded document gets its own directory:
        {base}/documents/{doc_id}/
            graph.json       — NetworkX graph (node-link JSON)
            embeddings.pkl   — FAISS index + id_map + counter
            metadata.json    — ingestion & processing metadata
            raw/             — original uploaded file (immutable)
            images/          — extracted page / figure images

    No database required.
    """

    def __init__(self, storage_dir="app/storage"):
        self.storage_dir = storage_dir
        self.documents_dir = os.path.join(storage_dir, "documents")
        os.makedirs(self.documents_dir, exist_ok=True)

    # ── helpers ────────────────────────────────────────────────────────────

    def _doc_dir(self, doc_id):
        """Return (and ensure existence of) the per-document directory."""
        d = os.path.join(self.documents_dir, doc_id)
        os.makedirs(d, exist_ok=True)
        return d

    def _doc_subdir(self, doc_id, subdir):
        """Return (and ensure existence of) a sub-directory inside the doc dir."""
        d = os.path.join(self._doc_dir(doc_id), subdir)
        os.makedirs(d, exist_ok=True)
        return d

    # ── graph ──────────────────────────────────────────────────────────────

    def save_graph(self, doc_id, G):
        """Save NetworkX graph to JSON file."""
        path = os.path.join(self._doc_dir(doc_id), "graph.json")
        data = json_graph.node_link_data(G, edges="links")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_graph(self, doc_id):
        """Load NetworkX graph from JSON file."""
        path = os.path.join(self._doc_dir(doc_id), "graph.json")
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            data = json.load(f)
        return json_graph.node_link_graph(data, edges="links")

    # ── embeddings ─────────────────────────────────────────────────────────

    def save_embeddings(self, doc_id, embed_manager):
        """Save embedding manager to pickle file."""
        path = os.path.join(self._doc_dir(doc_id), "embeddings.pkl")
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "index": embed_manager.index,
                    "id_map": embed_manager.id_map,
                    "counter": embed_manager.counter,
                },
                f,
            )

    def load_embeddings(self, doc_id, embed_manager):
        """Load embeddings into an existing manager."""
        path = os.path.join(self._doc_dir(doc_id), "embeddings.pkl")
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            data = pickle.load(f)
        embed_manager.index = data["index"]
        embed_manager.id_map = data["id_map"]
        embed_manager.counter = data["counter"]
        return embed_manager

    # ── metadata ───────────────────────────────────────────────────────────

    def save_metadata(self, doc_id, metadata):
        """Save document metadata."""
        path = os.path.join(self._doc_dir(doc_id), "metadata.json")
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2)

    def load_metadata(self, doc_id):
        """Load document metadata."""
        path = os.path.join(self._doc_dir(doc_id), "metadata.json")
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            return json.load(f)

    # ── raw document ───────────────────────────────────────────────────────

    def raw_dir(self, doc_id):
        """Return the directory where the original uploaded file is stored."""
        return self._doc_subdir(doc_id, "raw")

    # ── extracted images ───────────────────────────────────────────────────

    def images_dir(self, doc_id):
        """Return the directory for extracted images of this document."""
        return self._doc_subdir(doc_id, "images")

    # ── listing / deletion ─────────────────────────────────────────────────

    def list_documents(self):
        """List all stored document IDs."""
        if not os.path.exists(self.documents_dir):
            return []
        return [
            d
            for d in os.listdir(self.documents_dir)
            if os.path.isdir(os.path.join(self.documents_dir, d))
        ]

    def delete_document(self, doc_id):
        """Remove all data for a single document."""
        doc_path = os.path.join(self.documents_dir, doc_id)
        if os.path.exists(doc_path):
            shutil.rmtree(doc_path)

    def clear_all(self):
        """Remove ALL stored documents."""
        if os.path.exists(self.documents_dir):
            shutil.rmtree(self.documents_dir)
        os.makedirs(self.documents_dir, exist_ok=True)

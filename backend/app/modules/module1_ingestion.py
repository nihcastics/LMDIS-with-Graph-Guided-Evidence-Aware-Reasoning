import uuid
import os
import hashlib
import pymupdf as fitz  # PyMuPDF
from datetime import datetime


def calculate_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def ingest_document(file_content, filename, raw_dir=None):
    """
    Module 1: Document Ingestion & Immutable Source Registration

    Parameters
    ----------
    file_content : bytes   — raw file bytes
    filename     : str     — original upload filename
    raw_dir      : str|None — directory to save the raw file into.
                              When *None* the caller (main.py) is expected
                              to provide a per-document directory via
                              PersistentStorage.raw_dir(doc_id).  A second
                              call with the resolved path will move the file.
    """
    # 1. Identity Assignment
    doc_id = str(uuid.uuid4())

    # 2. Determine save location
    if raw_dir is None:
        # Fallback: legacy flat directory (should not happen in normal flow)
        base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        raw_dir = os.path.join(base, "app", "storage", "documents", doc_id, "raw")

    os.makedirs(raw_dir, exist_ok=True)

    file_ext = os.path.splitext(filename)[1]
    save_path = os.path.join(raw_dir, f"{doc_id}{file_ext}")

    with open(save_path, "wb") as f:
        f.write(file_content)

    # 3. Integrity Fingerprinting
    checksum = calculate_sha256(save_path)

    # 4. Non-Content Metadata Extraction
    try:
        doc = fitz.open(save_path)
        page_count = doc.page_count
        doc.close()
    except Exception:
        page_count = 0

    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)

    return {
        "doc_id": doc_id,
        "filename": filename,
        "local_path": save_path,
        "filesize_mb": round(file_size_mb, 2),
        "page_count": page_count,
        "checksum": checksum,
        "ingested_at": datetime.utcnow().isoformat(),
    }

# 🧠 Lossless Multimodal Document Intelligence System (LMDIS)

*Because documents deserve better than crushed pixels and forgotten tables.*

LMDIS is a document intelligence pipeline that extracts, structures, and retrieves information from complex multimodal documents without losing anything important. No more OCR chaos. No more keyword guesswork. Just clean, traceable, and grounded answers.

---

## ✨ What Makes LMDIS Special?

**Lossless structural parsing**  
We preserve layout, hierarchy, text, tables, and visual elements exactly as they appear.

**Multimodal processing**  
Works with digital PDFs, scanned images, and hybrid documents using adaptive extraction (PyMuPDF, pdfplumber, PaddleOCR).

**Graph based representation**  
Documents become knowledge graphs (NetworkX) that capture relationships between components. This makes reasoning and traceability much smarter.

**Semantic retrieval**  
Forget brittle keyword matching. We use embedding based similarity (Sentence Transformers + FAISS) for context aware search.

**Evidence grounded responses**  
Every answer is backed by source documents with clear references. No hallucinations.

---

## 🧱 Architecture at a Glance

The pipeline runs in four stages:

1. **Document Ingestion & Parsing** – Extract text and layout using OCR and PDF tools.
2. **Semantic Processing & Graph Construction** – Build a structured graph and generate embeddings.
3. **Multi Strategy Retrieval** – Combine semantic search, graph traversal, and entity lookup.
4. **Response Generation & Verification** – Produce answers with evidence validation and traceability.

---

## 🛠️ Tech Stack (the good stuff)

- Backend: FastAPI  
- Language: Python  
- OCR: PaddleOCR  
- Parsing: PyMuPDF, pdfplumber  
- Embeddings: Sentence Transformers (BGE)  
- Vector Index: FAISS  
- Graph Engine: NetworkX  

---

## 📂 Project Layout

Here's what you'll find inside the repo:

- `backend/` – FastAPI server, core processing logic, and API endpoints  
- `frontend/` – UI or interface components (if any)  
- `app/` – Main application modules and pipeline orchestration  
- `.venv/` – Python virtual environment (local, not committed)  
- `.vscode/` – VS Code workspace settings  
- `start.bat` – **One click startup** after you set up the virtual environment and dependencies  
- `launcher.py` – Alternative entry point for manual launching  
- `test_pipeline.py` – Basic pipeline tests  
- `test_transformer_pipeline.py` – Tests for transformer based components  
- `README.md` – This file you're reading  
- `_pycache_/` – Python bytecode cache (ignored, safe to delete)  

> 🧙 The `start.bat` file is your friend. It fires up the backend, loads the necessary components, and gets LMDIS ready for action.

---

## 🚀 Getting Started

1. **Clone the repository**  
   `git clone https://github.com/yourusername/LMDIS.git`  
   `cd LMDIS`

2. **Set up a virtual environment**  
   `python -m venv .venv`  
   Activate it:  
   - Windows: `.venv\Scripts\activate`  
   - Mac/Linux: `source .venv/bin/activate`

3. **Install dependencies**  
   `pip install -r requirements.txt`  
   (If you don't have a requirements.txt yet, check the backend or app folder for one)

4. **Run the system**  
   Double click `start.bat` or run it from the terminal:  
   `.\start.bat`  

   That's it. The system will start and you can begin processing documents.

---

## 🧪 Testing

Use the included test scripts to verify everything works:

- `python test_pipeline.py` – basic pipeline sanity  
- `python test_transformer_pipeline.py` – embedding and graph construction tests

---

## 🧩 Use Cases

- Legal and compliance document analysis  
- Financial auditing and reporting  
- Healthcare record processing  
- Research and academic document analysis  
- Enterprise knowledge management  

---

## ⚠️ Limitations

We're honest about what this thing can't do (yet):

- Graph construction and semantic indexing are computationally heavy  
- Performance depends on input document quality, especially for scanned pages  
- Currently optimized for single document workflows (multi document reasoning is coming)

---

## 🔮 Future Work

- Multi document reasoning and cross document linking  
- Performance optimization for large scale deployments  
- Better OCR for noisy and handwritten inputs  
- Real time processing capabilities  

---

## 📬 Contact

Got questions, ideas, or just want to talk documents? Reach out:

- **Sachin S** – sachin.shiva1612@gmail.com  
- **Ayush Raj** – ayushraj0901@gmail.com  

---

## 📄 License & Copyright

This project is licensed under the **GNU General Public License (GPL)**.  
You are free to use, modify, and distribute this software under the terms of the GPL. See the `LICENSE` file for full details.

© 2026 Sachin S and Ayush Raj. All rights reserved.  
Unauthorized copying, modification, or distribution outside the license terms is prohibited.

---

*No black magic, just well structured documents and a bit of graph theory.*

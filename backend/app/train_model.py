import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import json
import os

# Path where your OCR line memory is stored
MEMORY_FILE = "backend/app/storage/memory.json"

# Output path for trained model
MODEL_PATH = "backend/app/model/document_ai_model.pkl"

def train_and_save_model():
    # 1. Load OCR memory lines
    if not os.path.exists(MEMORY_FILE):
        raise FileNotFoundError("memory.json not found. Run ingestion before training.")

    with open(MEMORY_FILE, "r") as f:
        memory_data = json.load(f)

    # Extract only the text lines
    texts = [item["text"] for item in memory_data]

    # 2. Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Create embeddings
    embeddings = model.encode(texts)

    # 3. Create FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    # 4. Save model + index + text memory into a pickle file
    os.makedirs("backend/app/model", exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "model_name": "all-MiniLM-L6-v2",
            "memory": memory_data,
            "embeddings": embeddings,
            "faiss_index": index
        }, f)

    print("Model trained & saved at:", MODEL_PATH)


if __name__ == "__main__":
    train_and_save_model()

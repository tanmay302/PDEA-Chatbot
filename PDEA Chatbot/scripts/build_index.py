# scripts/build_index.py
# Run this ONCE (or whenever your dataset changes) to build the FAISS index.
# python scripts/build_index.py

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.vector_store import VectorStore
from config.settings import DATA_PATH

if __name__ == "__main__":
    print("Building FAISS index from:", DATA_PATH)
    store = VectorStore()
    store.build_from_json(DATA_PATH)
    print("Done! Index saved to data/faiss.index")  
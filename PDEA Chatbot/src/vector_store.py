# src/vector_store.py
# Handles: building, saving, loading, and querying the FAISS index

import json
import os
import numpy as np
import faiss
from src.embedder import Embedder
from config.settings import (
    INDEX_PATH, EMBEDDINGS_PATH, METADATA_PATH,
    DATA_PATH, TOP_K, SIMILARITY_THRESHOLD
)


class VectorStore:
    def __init__(self):
        self.embedder = Embedder()
        self.index = None
        self.metadata = []  # Parallel list: index i → {"question": ..., "answer": ...}

    # ─── Build ─────────────────────────────────────────────────────────────────

    def build_from_json(self, json_path: str = DATA_PATH):
        """
        One-time setup: load QA pairs, embed, store in FAISS.
        We embed question + answer together for richer matching.
        """
        print("[VectorStore] Loading dataset...")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Support both {"qa_pairs": [...]} and flat [...] formats
        items = data.get("qa_pairs", data) if isinstance(data, dict) else data

        questions = []
        for item in items:
            # Combine Q+A so retrieval matches on both surface and content
            combined = f"Question: {item['question']} Answer: {item['answer']}"
            questions.append(combined)
            self.metadata.append({
                "question": item["question"],
                "answer": item["answer"],
            })

        print(f"[VectorStore] Embedding {len(questions)} QA pairs...")
        embeddings = self.embedder.embed(questions)

        dim = embeddings.shape[1]  # 384 for MiniLM
        # IndexFlatIP = exact inner-product search (= cosine sim on normalised vecs)
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        self._save(embeddings)
        print(f"[VectorStore] Index built: {self.index.ntotal} vectors, dim={dim}")

    # ─── Persist ───────────────────────────────────────────────────────────────

    def _save(self, embeddings: np.ndarray):
        os.makedirs("data", exist_ok=True)
        faiss.write_index(self.index, INDEX_PATH)
        np.save(EMBEDDINGS_PATH, embeddings)
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        print("[VectorStore] Saved index, embeddings, and metadata.")

    def load(self):
        """Load pre-built index from disk. Fast — no re-embedding needed."""
        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError(
                "FAISS index not found. Run: python scripts/build_index.py"
            )
        self.index = faiss.read_index(INDEX_PATH)
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        print(f"[VectorStore] Loaded index: {self.index.ntotal} vectors")

    # ─── Query ─────────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """
        Returns list of dicts: [{question, answer, score}, ...]
        Filtered by SIMILARITY_THRESHOLD. Empty list if no good match.
        """
        query_vec = self.embedder.embed_single(query)  # shape (1, 384)
        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue  # FAISS returns -1 when fewer results than top_k
            if float(score) >= SIMILARITY_THRESHOLD:
                results.append({
                    **self.metadata[idx],
                    "score": round(float(score), 4),
                })

        return results  # Already sorted by score (FAISS returns descending)
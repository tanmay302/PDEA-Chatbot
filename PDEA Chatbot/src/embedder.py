# src/embedder.py
# Single responsibility: convert text → dense vectors
# All embedding logic lives here. Nothing else touches the model directly.

import numpy as np
from sentence_transformers import SentenceTransformer
from config.settings import EMBEDDING_MODEL


class Embedder:
    _instance = None  # Singleton: load model only once

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = None
        return cls._instance

    def load(self):
        """Load model lazily — only when first needed."""
        if self._model is None:
            print(f"[Embedder] Loading model: {EMBEDDING_MODEL}")
            self._model = SentenceTransformer(EMBEDDING_MODEL)
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of strings → float32 numpy array of shape (N, 384).
        Normalised for cosine similarity via dot product.
        """
        self.load()
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,   # L2-normalise → dot product = cosine sim
            show_progress_bar=len(texts) > 50,
            batch_size=64,               # Tune down to 32 if RAM is tight
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32)

    def embed_single(self, text: str) -> np.ndarray:
        """Embed one string → shape (1, 384). Used at query time."""
        return self.embed([text])
# src/rag_pipeline.py
# This is the single entry point your main.py and Flask API both call.
# It wires together: VectorStore + (optional) GeminiClient

from src.vector_store import VectorStore
from config.settings import USE_GEMINI, SIMILARITY_THRESHOLD


class RAGPipeline:
    def __init__(self):
        self.store = VectorStore()
        self.store.load()
        self.gemini = None
        if USE_GEMINI:
            try:
                from src.gemini_api import GeminiClient
                self.gemini = GeminiClient()
                print("[RAG] Gemini fallback enabled.")
            except Exception as e:
                print(f"[RAG] Gemini disabled: {e}")

    def answer(self, query: str) -> dict:
        """
        Main method. Returns:
        {
          "answer": str,
          "source": "local" | "gemini" | "fallback",
          "results": [...],   # retrieved chunks
          "query": str,
        }
        """
        query = query.strip()
        if not query:
            return self._fallback(query, "Empty query.")

        results = self.store.search(query)

        # ── Case 1: Good local match found ─────────────────────────────────────
        if results:
            top = results[0]
            # If Gemini is enabled and score is mediocre, synthesise a better answer
            if self.gemini and top["score"] < 0.6:
                gemini_answer = self.gemini.generate(query, results)
                if gemini_answer:
                    return {
                        "query": query,
                        "answer": gemini_answer,
                        "source": "gemini",
                        "results": results,
                    }
            # Default: return best local match answer directly
            return {
                "query": query,
                "answer": top["answer"],
                "source": "local",
                "results": results,
            }

        # ── Case 2: No match above threshold ───────────────────────────────────
        if self.gemini:
            # Try raw Gemini without context (last resort)
            gemini_answer = self.gemini.generate(query, [])
            if gemini_answer:
                return {
                    "query": query,
                    "answer": gemini_answer,
                    "source": "gemini",
                    "results": [],
                }

        return self._fallback(query)

    def _fallback(self, query: str, reason: str = "") -> dict:
        return {
            "query": query,
            "answer": (
                "I'm sorry, I don't have information about that. "
                "Please contact the college office or visit pdea.org.in"
            ),
            "source": "fallback",
            "results": [],
        }
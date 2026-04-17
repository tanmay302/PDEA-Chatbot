# src/rag_pipeline.py
# Orchestrates: VectorStore + optional GeminiClient
# Single entry point for both main.py and app.py

from src.vector_store import VectorStore
from config.settings import USE_GEMINI, SIMILARITY_THRESHOLD


def _safe_str(val) -> str:
    """Guarantee any answer value becomes a plain string."""
    if val is None:
        return ""
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, bool):
        return "Yes" if val else "No"
    if isinstance(val, (int, float)):
        return str(val)
    if isinstance(val, list):
        return ", ".join(_safe_str(x) for x in val)
    if isinstance(val, dict):
        parts = []
        for k, v in val.items():
            label = k.replace("_", " ").capitalize()
            converted = _safe_str(v)
            if converted:
                parts.append(f"{label}: {converted}")
        return ". ".join(parts)
    return str(val)


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
        Returns:
        {
          "answer":  str,
          "source":  "local" | "gemini" | "fallback",
          "results": [...],
          "query":   str,
        }
        """
        query = query.strip()
        if not query:
            return self._fallback(query)

        results = self.store.search(query)

        # Sanitise all result answers to strings
        for r in results:
            r["answer"] = _safe_str(r.get("answer", ""))

        # ── Case 1: Good local match ──────────────────────────────────────────
        if results:
            top = results[0]

            if self.gemini and top["score"] < 0.6:
                gemini_answer = self.gemini.generate(query, results)
                if gemini_answer:
                    return {
                        "query":   query,
                        "answer":  _safe_str(gemini_answer),
                        "source":  "gemini",
                        "results": results,
                    }

            return {
                "query":   query,
                "answer":  _safe_str(top["answer"]),
                "source":  "local",
                "results": results,
            }

        # ── Case 2: No match — try Gemini raw ────────────────────────────────
        if self.gemini:
            gemini_answer = self.gemini.generate(query, [])
            if gemini_answer:
                return {
                    "query":   query,
                    "answer":  _safe_str(gemini_answer),
                    "source":  "gemini",
                    "results": [],
                }

        return self._fallback(query)

    def _fallback(self, query: str = "") -> dict:
        return {
            "query":   query,
            "answer":  (
                "I'm sorry, I don't have information about that. "
                "Please contact the college office or visit pdea.org.in"
            ),
            "source":  "fallback",
            "results": [],
        }
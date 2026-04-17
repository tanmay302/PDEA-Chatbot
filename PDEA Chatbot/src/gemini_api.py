# src/gemini_api.py
# Gemini is optional. Only called when local retrieval confidence is low.
# Hard session cap to prevent runaway API costs.

import os
import google.generativeai as genai
from config.settings import GEMINI_MODEL, MAX_GEMINI_CALLS_PER_SESSION
from dotenv import load_dotenv

load_dotenv()


class GeminiClient:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in .env")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(GEMINI_MODEL)
        self._calls_this_session = 0

    def generate(self, query: str, context_chunks: list[dict]) -> str:
        """
        Generate an answer grounded in retrieved context.
        Returns None if session cap exceeded.
        """
        if self._calls_this_session >= MAX_GEMINI_CALLS_PER_SESSION:
            return None  # Caller handles this gracefully

        context_text = "\n".join(
            f"Q: {c['question']}\nA: {c['answer']}"
            for c in context_chunks
        )

        prompt = f"""You are a helpful assistant for PDEA College, Pune.
Use ONLY the context below to answer the student's question.
If the context doesn't contain the answer, say "I don't have that information."

Context:
{context_text}

Student question: {query}
Answer:"""

        try:
            response = self.model.generate_content(prompt)
            self._calls_this_session += 1
            return response.text.strip()
        except Exception as e:
            print(f"[Gemini] Error: {e}")
            return None
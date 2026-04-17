# src/api.py — Flask REST API
# Run: python src/api.py

from flask import Flask, request, jsonify
from src.rag_pipeline import RAGPipeline

app = Flask(__name__)
pipeline = RAGPipeline()  # Loaded once at startup


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "query is required"}), 400

    result = pipeline.answer(query)
    return jsonify({
        "answer": result["answer"],
        "source": result["source"],
        "confidence": result["results"][0]["score"] if result["results"] else 0.0,
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "vectors": pipeline.store.index.ntotal})


if __name__ == "__main__":
    app.run(debug=False, port=5000)
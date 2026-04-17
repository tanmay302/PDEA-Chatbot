# config/settings.py — all tuneable parameters in one place

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Fast, lightweight, 384-dim vectors. Free and local.

TOP_K = 3                    # Number of results to retrieve
SIMILARITY_THRESHOLD = 0.30  # Cosine similarity floor (0–1). Below this = no match.
USE_GEMINI = False           # Set True to enable Gemini as fallback

DATA_PATH = "data/pdea_chatbot_qa.json"
INDEX_PATH = "data/faiss.index"
EMBEDDINGS_PATH = "data/embeddings.npy"
METADATA_PATH = "data/metadata.json"

GEMINI_MODEL = "gemini-1.5-flash"  # Cheapest Gemini model
MAX_GEMINI_CALLS_PER_SESSION = 5   # Hard cap to control costs
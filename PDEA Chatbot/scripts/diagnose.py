# scripts/diagnose.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.vector_store import VectorStore

store = VectorStore()
store.load()

test_queries = [
    "Courses",
    "What courses are available?",
    "Which programs does PDEA offer?",
    "admission",
    "fees",
]

print(f"\n{'='*60}")
print(f"Total vectors in index: {store.index.ntotal}")
print(f"{'='*60}\n")

for query in test_queries:
    # Search with NO threshold — raw scores
    import numpy as np
    vec = store.embedder.embed_single(query)
    scores, indices = store.index.search(vec, 5)
    
    print(f"Query: '{query}'")
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        q = store.metadata[idx]['question'][:55]
        print(f"  {score:.4f} | {q}")
    print()
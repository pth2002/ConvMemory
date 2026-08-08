"""ConvMemory on top of a plain numpy vector store. No framework involved.

    python examples/integrations/basic_vector_store.py

This is the shape every other integration in this folder reduces to:

    store.search(query, top_k=200)   ->  ConvMemory.rerank_embeddings(...)  ->  top 5
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from convmemory import ConvMemory

DEMO_DATA = Path(__file__).resolve().parents[1] / "demo_memories.json"


class VectorStore:
    """The memory store you already have."""

    def __init__(self, encoder):
        self.encoder = encoder
        self.ids: list[str] = []
        self.texts: list[str] = []
        self.vectors: np.ndarray | None = None

    def add(self, memories):
        """Insert memories in chronological order and embed them once."""
        memories = list(memories)
        self.ids.extend(m["id"] for m in memories)
        self.texts.extend(m["text"] for m in memories)
        # `ConvMemory.encode` returns unit-norm float32 vectors.
        new_vectors = self.encoder.encode([m["text"] for m in memories])
        self.vectors = (
            new_vectors if self.vectors is None else np.vstack([self.vectors, new_vectors])
        )

    def search(self, query_vector, top_k):
        scores = self.vectors @ query_vector
        order = np.argsort(-scores)[:top_k]
        return [int(i) for i in order]


def main():
    model = ConvMemory.from_pretrained("Purdy0228/ConvMemory-LoCoMo-MPNet", device="cpu")

    store = VectorStore(encoder=model)
    payload = json.loads(DEMO_DATA.read_text(encoding="utf-8"))
    store.add({"id": f"m{i}", "text": text} for i, text in enumerate(payload["memories"]))

    # Optional: cache lexical signatures once for a stable store.
    model.prewarm_lexical([{"id": i, "text": t} for i, t in zip(store.ids, store.texts)])

    query = "Why did we remove the Redis cache?"
    query_vector = model.encode([query])[0]

    # 1. your existing search, widened to a candidate pool
    candidate_indices = store.search(query_vector, top_k=200)

    # 2. ConvMemory reorders that pool, reusing the embeddings you already have
    ranked = model.rerank_embeddings(
        query_embedding=query_vector,
        memory_embeddings=store.vectors,
        memory_ids=store.ids,
        memory_texts=store.texts,
        query=query,
        candidate_indices=candidate_indices,
        top_k=5,
    )

    print(f"Q: {query}\n")
    print("dense only:")
    for rank, index in enumerate(candidate_indices[:5], start=1):
        print(f"  {rank}. {store.texts[index]}")
    print("\ndense + ConvMemory:")
    for item in ranked:
        print(f"  {item.rank}. {item.text}")


if __name__ == "__main__":
    main()

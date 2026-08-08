"""A plain agent memory loop with ConvMemory in the retrieval path.

    python examples/integrations/custom_agent.py

No framework, no LLM call — the point is where the reranker sits and what the
agent ends up reading. Swap `build_prompt` for your real model call.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from convmemory import ConvMemory

DEMO_DATA = Path(__file__).resolve().parents[1] / "demo_memories.json"

CANDIDATE_POOL = 200
CONTEXT_BUDGET = 6


class AgentMemory:
    """Append-only memory with a rerank stage in front of the prompt."""

    def __init__(self, reranker: ConvMemory):
        self.reranker = reranker
        self.ids: list[str] = []
        self.texts: list[str] = []
        self.vectors: np.ndarray | None = None

    def remember(self, text: str) -> str:
        memory_id = f"m{len(self.ids)}"
        vector = self.reranker.encode([text])
        self.ids.append(memory_id)
        self.texts.append(text)
        self.vectors = vector if self.vectors is None else np.vstack([self.vectors, vector])
        return memory_id

    def recall(self, query: str, budget: int = CONTEXT_BUDGET):
        """Return the memories the agent should read for this query."""
        if not self.ids:
            return []

        query_vector = self.reranker.encode([query])[0]
        dense_order = np.argsort(-(self.vectors @ query_vector))[:CANDIDATE_POOL]

        ranked = self.reranker.rerank_embeddings(
            query_embedding=query_vector,
            memory_embeddings=self.vectors,
            memory_ids=self.ids,
            memory_texts=self.texts,
            query=query,
            candidate_indices=[int(i) for i in dense_order],
            top_k=budget,
        )
        return [{"id": item.memory_id, "text": item.text, "score": item.score} for item in ranked]


def build_prompt(query: str, memories) -> str:
    lines = "\n".join(f"- {m['text']}" for m in memories)
    return (
        "Relevant memories:\n"
        f"{lines}\n\n"
        f"User: {query}\n"
        "Assistant:"
    )


def main():
    model = ConvMemory.from_pretrained("Purdy0228/ConvMemory-LoCoMo-MPNet", device="cpu")
    memory = AgentMemory(model)

    payload = json.loads(DEMO_DATA.read_text(encoding="utf-8"))
    for text in payload["memories"]:
        memory.remember(text)
    print(f"stored {len(memory.ids)} memories\n")

    query = "What database are we using for the analytics store now?"
    recalled = memory.recall(query)
    print(build_prompt(query, recalled))


if __name__ == "__main__":
    main()

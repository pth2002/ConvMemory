"""Rerank mem0 search results with ConvMemory.

mem0 keeps the memories; ConvMemory only reorders what mem0 returns:

    mem0.search(query, limit=200) -> ConvMemory -> top 10 -> your agent

Requires `mem0ai`, which ConvMemory does not depend on:

    pip install convmemory mem0ai

Written against the `Memory.search(...)` return shape used by mem0 v0.1.x, which
returns either `{"results": [...]}` or a bare list depending on version. Both are
handled below. This file is not exercised by this repo's CI, because mem0 is not
a dependency. If it breaks against your version, please open an issue with the
shape of what `search` returned.
"""

from __future__ import annotations

from typing import Any

from convmemory import ConvMemory

CANDIDATE_LIMIT = 200
TOP_N = 10


def _as_records(search_result: Any) -> list[dict]:
    """mem0 returns {'results': [...]} on newer versions and a list on older ones."""
    if isinstance(search_result, dict):
        return list(search_result.get("results", []))
    return list(search_result)


def _memory_text(record: dict) -> str:
    for key in ("memory", "text", "content"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    raise KeyError(f"no memory text field found in mem0 record: {sorted(record)}")


class ConvMemoryReranked:
    """Thin wrapper: search wide with mem0, return a reranked short list."""

    def __init__(self, mem0_client, reranker: ConvMemory, top_n: int = TOP_N):
        self.mem0 = mem0_client
        self.reranker = reranker
        self.top_n = top_n

    @classmethod
    def from_pretrained(
        cls,
        mem0_client,
        checkpoint: str = "Purdy0228/ConvMemory-LoCoMo-MPNet",
        device: str = "cpu",
        top_n: int = TOP_N,
    ):
        return cls(mem0_client, ConvMemory.from_pretrained(checkpoint, device=device), top_n)

    def search(self, query: str, limit: int = CANDIDATE_LIMIT, **mem0_kwargs) -> list[dict]:
        records = _as_records(self.mem0.search(query=query, limit=limit, **mem0_kwargs))
        if not records:
            return []

        memories = [
            {"id": str(record.get("id", index)), "text": _memory_text(record)}
            for index, record in enumerate(records)
        ]
        by_id = {memory["id"]: record for memory, record in zip(memories, records)}

        ranked = self.reranker.rerank(query=query, memories=memories, top_k=self.top_n)
        return [
            {**by_id[item.memory_id], "convmemory_score": item.score, "rank": item.rank}
            for item in ranked
        ]


if __name__ == "__main__":
    # Sketch of the wiring; needs a configured mem0 client to run.
    #
    # from mem0 import Memory
    #
    # client = ConvMemoryReranked.from_pretrained(Memory(), top_n=10)
    # for hit in client.search("what did we decide about the analytics store?", user_id="alice"):
    #     print(hit["rank"], round(hit["convmemory_score"], 4), hit["memory"])

    class _FakeMem0:
        def search(self, query, limit, **kwargs):
            return {
                "results": [
                    {"id": "m1", "memory": "we moved the analytics store to ClickHouse", "score": 0.41},
                    {"id": "m2", "memory": "the offsite is in Porto in May", "score": 0.38},
                    {"id": "m3", "memory": "Postgres was the original analytics store", "score": 0.36},
                ]
            }

    client = ConvMemoryReranked.from_pretrained(_FakeMem0(), top_n=3)
    for hit in client.search("what database do we use for analytics?"):
        print(hit["rank"], round(hit["convmemory_score"], 4), hit["memory"])

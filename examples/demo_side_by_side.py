"""The case where ConvMemory does NOT help: a small, lexically easy memory pool.

    python examples/demo_side_by_side.py

This runs dense retrieval and dense + ConvMemory side by side over the ~116
synthetic memories in ``examples/demo_memories.json``. On a pool that small,
plain dense search already puts the gold memory in the top 5 for every query, so
the reranker has nothing to recover and sometimes reorders for the worse.

That is the honest boundary of the method, and it is kept in the repo on
purpose. For the case ConvMemory is built for -- hundreds to thousands of
candidates, where dense recall degrades -- see ``examples/demo_locomo.py``.

The pool here is synthetic and illustrative. Benchmark numbers come from LoCoMo
and LongMemEval; see docs/BENCHMARKS.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from convmemory import ConvMemory

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "demo_memories.json"
DEFAULT_CHECKPOINT = "Purdy0228/ConvMemory-LoCoMo-MPNet"


def dense_top_k(model, query, memories, top_k):
    """Cosine-similarity baseline over the same embeddings ConvMemory uses."""
    query_vec = model.encode([query])[0]
    memory_vecs = model.encode([m["text"] for m in memories])

    query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    memory_vecs = memory_vecs / (
        np.linalg.norm(memory_vecs, axis=1, keepdims=True) + 1e-12
    )
    scores = memory_vecs @ query_vec
    order = np.argsort(-scores)[:top_k]
    return [(memories[i]["text"], float(scores[i])) for i in order]


def is_gold(text, gold_contains):
    lowered = text.lower()
    return any(needle.lower() in lowered for needle in gold_contains)


def render_column(rows, gold_contains, width):
    lines = []
    for rank, (text, _score) in enumerate(rows, start=1):
        mark = "OK " if is_gold(text, gold_contains) else "   "
        body = text if len(text) <= width else text[: width - 1] + "…"
        lines.append(f"{rank}. {mark}{body}")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--width", type=int, default=62)
    args = parser.parse_args()

    payload = json.loads(args.data.read_text(encoding="utf-8"))
    memories = [
        {"id": f"m{i}", "text": text}
        for i, text in enumerate(payload["memories"])
    ]

    print(f"Loading {args.checkpoint} on {args.device} ...")
    model = ConvMemory.from_pretrained(args.checkpoint, device=args.device)
    print(f"Memory pool: {len(memories)} memories\n")

    dense_hits = 0
    conv_hits = 0

    for case in payload["queries"]:
        query = case["query"]
        gold = case["gold_contains"]

        dense_rows = dense_top_k(model, query, memories, args.top_k)
        conv_rows = [
            (item.text, item.score)
            for item in model.rerank(query=query, memories=memories, top_k=args.top_k)
        ]

        dense_hit = any(is_gold(text, gold) for text, _ in dense_rows)
        conv_hit = any(is_gold(text, gold) for text, _ in conv_rows)
        dense_hits += int(dense_hit)
        conv_hits += int(conv_hit)

        print("=" * (args.width * 2 + 7))
        print(f"Q: {query}")
        print("=" * (args.width * 2 + 7))

        left = render_column(dense_rows, gold, args.width)
        right = render_column(conv_rows, gold, args.width)
        header_left = f"dense only (hit={dense_hit})"
        header_right = f"dense + ConvMemory (hit={conv_hit})"
        print(f"{header_left:<{args.width + 3}} | {header_right}")
        print("-" * (args.width + 3) + "-+-" + "-" * (args.width + 3))
        for i in range(max(len(left), len(right))):
            lhs = left[i] if i < len(left) else ""
            rhs = right[i] if i < len(right) else ""
            print(f"{lhs:<{args.width + 3}} | {rhs}")
        print()

    total = len(payload["queries"])
    print(f"gold in top-{args.top_k}: dense {dense_hits}/{total}, "
          f"dense + ConvMemory {conv_hits}/{total}")


if __name__ == "__main__":
    main()

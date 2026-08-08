"""Real before/after demo on LoCoMo: dense retrieval vs dense + ConvMemory.

This is the reproducible version of the README's side-by-side example. It runs
on the held-out conversations of the released checkpoint's split, so the numbers
it prints are not measured on training conversations.

Get the data first (LoCoMo is a public research dataset):

    https://github.com/snap-research/locomo  ->  data/locomo10.json

Then:

    python examples/demo_locomo.py --data data/locomo10.json --device cuda

Scope: LoCoMo is in-domain for the released checkpoint (it was trained on the
other half of the same dataset family). This demo shows the retrieval-stage
before/after, not out-of-domain generalization. See the README's "Where it does
not help" section for the boundaries.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from convmemory import ConvMemory

DEFAULT_CHECKPOINT = "Purdy0228/ConvMemory-LoCoMo-MPNet"


def session_number(key: str) -> int:
    try:
        return int(key.split("_")[1])
    except (IndexError, ValueError):
        return 0


def load_locomo(path: Path):
    """Flatten LoCoMo into (sample_id, query, gold ids, memory pool) examples."""
    data = json.loads(path.read_text(encoding="utf-8"))
    samples = {}
    for sample in data:
        conversation = sample["conversation"]
        session_keys = sorted(
            (
                key
                for key in conversation
                if key.startswith("session_")
                and not key.endswith("date_time")
                and isinstance(conversation[key], list)
            ),
            key=session_number,
        )
        memories, dia_ids = [], set()
        for key in session_keys:
            for turn in conversation[key]:
                dia_ids.add(turn["dia_id"])
                memories.append(
                    {
                        "id": turn["dia_id"],
                        "text": f"{turn.get('speaker', 'speaker')}: {turn.get('text', '')}",
                    }
                )

        questions = []
        for qa in sample["qa"]:
            evidence = [x for x in qa.get("evidence", []) if x in dia_ids]
            if not evidence:
                continue
            questions.append(
                {
                    "query": qa["question"],
                    "answer": qa.get("answer", ""),
                    "gold": set(evidence),
                }
            )
        samples[sample["sample_id"]] = {"memories": memories, "questions": questions}
    return samples


def held_out_sample_ids(sample_ids, seed: int, dev_ratio: float):
    """Same conversation-level split the training/eval scripts use."""
    ordered = sorted(sample_ids)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    split = max(1, int(len(ordered) * dev_ratio))
    dev = set(ordered[:split])
    return [sid for sid in sorted(sample_ids) if sid not in dev]


def normalize(matrix):
    return matrix / (np.linalg.norm(matrix, axis=-1, keepdims=True) + 1e-12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/locomo10.json"))
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=23, help="split seed of the released checkpoint")
    parser.add_argument("--dev-ratio", type=float, default=0.5)
    parser.add_argument("--split", choices=["test", "all"], default="test")
    parser.add_argument("--examples", type=int, default=3, help="side-by-side cases to print")
    parser.add_argument("--width", type=int, default=86)
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(
            f"{args.data} not found. Download locomo10.json from "
            "https://github.com/snap-research/locomo and pass --data."
        )

    samples = load_locomo(args.data)
    if args.split == "test":
        selected = held_out_sample_ids(samples.keys(), args.seed, args.dev_ratio)
        print(f"Held-out conversations (seed {args.seed}): {', '.join(selected)}")
    else:
        selected = sorted(samples)
        print(f"All conversations: {', '.join(selected)}")

    model = ConvMemory.from_pretrained(args.checkpoint, device=args.device)

    stats = {"n": 0, "dense@5": 0, "conv@5": 0, "dense@10": 0, "conv@10": 0}
    encode_seconds = 0.0
    dense_seconds = 0.0
    conv_seconds = 0.0
    showcase = []

    for sample_id in selected:
        sample = samples[sample_id]
        memories = sample["memories"]
        texts = [m["text"] for m in memories]
        ids = [m["id"] for m in memories]
        # Memory embeddings are computed once per conversation, the way a real
        # memory store would hold them. `encode` already returns unit vectors.
        memory_vecs = model.encode(texts)
        model.prewarm_lexical(memories)

        for question in sample["questions"]:
            query = question["query"]
            gold = question["gold"]

            started = time.perf_counter()
            query_vec = model.encode([query])[0]
            encode_seconds += time.perf_counter() - started

            started = time.perf_counter()
            dense_order = np.argsort(-(memory_vecs @ query_vec))[:10]
            dense_seconds += time.perf_counter() - started
            dense_ids = [memories[i]["id"] for i in dense_order]

            started = time.perf_counter()
            ranked = model.rerank_embeddings(
                query_embedding=query_vec,
                memory_embeddings=memory_vecs,
                memory_ids=ids,
                memory_texts=texts,
                query=query,
                top_k=10,
            )
            conv_seconds += time.perf_counter() - started
            conv_ids = [item.memory_id for item in ranked]

            stats["n"] += 1
            stats["dense@5"] += bool(gold & set(dense_ids[:5]))
            stats["conv@5"] += bool(gold & set(conv_ids[:5]))
            stats["dense@10"] += bool(gold & set(dense_ids[:10]))
            stats["conv@10"] += bool(gold & set(conv_ids[:10]))

            if not (gold & set(dense_ids[:5])) and (gold & set(conv_ids[:5])):
                showcase.append(
                    {
                        "sample_id": sample_id,
                        "pool": len(memories),
                        "gold_rank": min(
                            conv_ids.index(g) + 1 for g in gold if g in conv_ids
                        ),
                        "query": query,
                        "answer": question["answer"],
                        "gold": gold,
                        "dense": dense_ids[:5],
                        "conv": conv_ids[:5],
                        "texts": {m["id"]: m["text"] for m in memories},
                    }
                )

    total = stats["n"]
    print(f"\nQuestions: {total}")
    print(f"Candidate pools: {min(len(samples[s]['memories']) for s in selected)}"
          f"-{max(len(samples[s]['memories']) for s in selected)} memories\n")
    print(f"{'metric':<12}{'dense only':>14}{'dense + ConvMemory':>22}")
    for k in (5, 10):
        dense = stats[f"dense@{k}"]
        conv = stats[f"conv@{k}"]
        print(f"hit@{k:<8}{dense:>7} ({dense / total:.1%}){conv:>13} ({conv / total:.1%})")
    print(
        f"\nper-query time (memory embeddings precomputed, lexical signatures prewarmed):"
        f"\n  query encoding (shared by both paths): {encode_seconds / total * 1000:.2f} ms"
        f"\n  dense scoring:                         {dense_seconds / total * 1000:.2f} ms"
        f"\n  ConvMemory reranking:                  {conv_seconds / total * 1000:.2f} ms"
    )
    print(f"\nquestions where dense missed at top-5 and ConvMemory recovered: {len(showcase)}")

    showcase.sort(key=lambda case: (case["gold_rank"], -case["pool"]))
    for case in showcase[: args.examples]:
        print("\n" + "=" * (args.width + 6))
        print(f"Q: {case['query']}")
        print(f"A: {case['answer']}   [pool: {case['pool']} memories]")
        print("=" * (args.width + 6))
        for label, ids in (("dense only", case["dense"]), ("dense + ConvMemory", case["conv"])):
            print(f"\n{label}:")
            for rank, mid in enumerate(ids, start=1):
                mark = "OK " if mid in case["gold"] else "   "
                text = case["texts"][mid]
                body = text if len(text) <= args.width else text[: args.width - 1] + "…"
                print(f"  {rank}. {mark}{body}")


if __name__ == "__main__":
    main()

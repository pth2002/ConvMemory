"""Does it matter how the candidate pool is handed to ConvMemory?

Issue #1 (OpenViking) feeds a dense top-60 to the reranker. The released model
was trained and evaluated with the whole memory store in store order plus a
candidate subset, so its `position` feature and its neighbor windows refer to
the store. This script measures, on the held-out LoCoMo conversations of the
released checkpoint's split, the same 60-candidate pool handed over four ways:

  dense        dense top-60 order, no reranking
  full         whole store in store order, candidate_indices = dense top-60
  full_local   same, window_mode="candidate_local" (windows only around candidates)
  only_rank    memories = the 60 candidates in dense-rank order (README 3-line / adapters path)
  only_store   memories = the 60 candidates re-sorted into store order
  ref500       whole store, default top-500 pool (the README demo setting)

    python experiments/issue_1/order_check.py --data data/locomo10.json --device cpu

Results from the run reported in issue #1 are in order_check_results.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from math import comb
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "examples"))
from demo_locomo import held_out_sample_ids, load_locomo  # noqa: E402

from convmemory import ConvMemory  # noqa: E402

ARMS = ["dense", "full", "full_local", "only_rank", "only_store", "ref500"]


def exact_mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def first_gold_rank(ids, gold):
    for r, mid in enumerate(ids, start=1):
        if mid in gold:
            return r
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=HERE.parents[1] / "data" / "locomo10.json")
    ap.add_argument("--checkpoint", default="Purdy0228/ConvMemory-LoCoMo-MPNet")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--pool", type=int, default=60)
    ap.add_argument("--out", type=Path, default=HERE / "order_check_results.json")
    args = ap.parse_args()

    samples = load_locomo(args.data)
    selected = held_out_sample_ids(samples.keys(), args.seed, 0.5)
    print("held-out conversations:", ", ".join(selected), flush=True)
    model = ConvMemory.from_pretrained(args.checkpoint, device=args.device)

    ranks = {arm: [] for arm in ARMS}
    secs = {arm: 0.0 for arm in ARMS}
    in_pool = []
    pools = []

    for sid in selected:
        memories = samples[sid]["memories"]
        ids = [m["id"] for m in memories]
        texts = [m["text"] for m in memories]
        pools.append(len(memories))
        vecs = model.encode(texts)
        model.prewarm_lexical(memories)
        print(f"{sid}: {len(memories)} memories, {len(samples[sid]['questions'])} questions", flush=True)

        for q in samples[sid]["questions"]:
            query, gold = q["query"], q["gold"]
            qv = model.encode([query])[0]
            order = np.argsort(-(vecs @ qv))
            cand = [int(i) for i in order[: args.pool]]
            in_pool.append(any(ids[i] in gold for i in cand))
            ranks["dense"].append(first_gold_rank([ids[i] for i in cand], gold))

            def run(arm, **kw):
                t0 = time.perf_counter()
                res = model.rerank_embeddings(query_embedding=qv, query=query, **kw)
                secs[arm] += time.perf_counter() - t0
                ranks[arm].append(first_gold_rank([r.memory_id for r in res], gold))

            run("full", memory_embeddings=vecs, memory_ids=ids, memory_texts=texts, candidate_indices=cand)
            run("full_local", memory_embeddings=vecs, memory_ids=ids, memory_texts=texts,
                candidate_indices=cand, window_mode="candidate_local")
            run("only_rank", memory_embeddings=vecs[cand], memory_ids=[ids[i] for i in cand],
                memory_texts=[texts[i] for i in cand])
            store = sorted(cand)
            run("only_store", memory_embeddings=vecs[store], memory_ids=[ids[i] for i in store],
                memory_texts=[texts[i] for i in store])
            run("ref500", memory_embeddings=vecs, memory_ids=ids, memory_texts=texts)

    n = len(ranks["dense"])
    hit = lambda arm, k: np.array([0 < r <= k for r in ranks[arm]])
    mrr = lambda arm: float(np.mean([1.0 / r if r else 0.0 for r in ranks[arm]]))

    summary = {"n": n, "pool": args.pool, "store_sizes": pools, "ceiling_hit@pool": float(np.mean(in_pool)), "arms": {}}
    print(f"\nquestions {n}, store sizes {min(pools)}-{max(pools)}, gold inside dense top-{args.pool}: "
          f"{sum(in_pool)} ({np.mean(in_pool):.1%})\n")
    print(f"{'arm':<12}{'hit@5':>14}{'hit@10':>14}{'MRR':>8}{'ms/q':>9}   vs dense @10 (helped/harmed, p)")
    for arm in ARMS:
        h5, h10 = hit(arm, 5), hit(arm, 10)
        row = {"hit@5": int(h5.sum()), "hit@10": int(h10.sum()), "mrr": mrr(arm),
               "ms_per_query": secs[arm] / n * 1000}
        line = f"{arm:<12}{h5.sum():>6} ({h5.mean():5.1%}){h10.sum():>6} ({h10.mean():5.1%}){row['mrr']:>8.3f}{row['ms_per_query']:>9.1f}"
        if arm != "dense":
            d10 = hit("dense", 10)
            b, c = int((h10 & ~d10).sum()), int((~h10 & d10).sum())
            row["vs_dense@10"] = {"helped": b, "harmed": c, "p": exact_mcnemar(b, c)}
            line += f"   {b}/{c}, p={exact_mcnemar(b, c):.3g}"
        summary["arms"][arm] = row
        print(line)

    print("\npaired, hit@10:")
    for a, b_ in (("full", "only_rank"), ("full", "only_store"), ("full_local", "full"), ("only_store", "only_rank")):
        x, y = hit(a, 10), hit(b_, 10)
        b, c = int((x & ~y).sum()), int((~x & y).sum())
        summary.setdefault("paired@10", {})[f"{a}_vs_{b_}"] = {"a_only": b, "b_only": c, "p": exact_mcnemar(b, c)}
        print(f"  {a} vs {b_}: {a} only {b}, {b_} only {c}, p={exact_mcnemar(b, c):.3g}")

    args.out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("\nsaved", args.out)


if __name__ == "__main__":
    main()

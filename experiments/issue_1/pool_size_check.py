"""How does the candidate pool size change reranking quality?

Follow-up to issue #1, which asked about the `ref500` arm in
order_check_results.json (whole store, 500-candidate pool). Every arm here uses
the intended hand-off: the whole store in store order plus `candidate_indices`
= dense top-N in the model's own embedding space.

  OPC student   Purdy0228/ConvMemory-OPC-Student-BGE (bge-base-zh-v1.5 space),
                trained on a 20-candidate pool. Settings zh->zh, zh->en, en->en.
  English v1    Purdy0228/ConvMemory-LoCoMo-MPNet (all-mpnet-base-v2 space),
                trained on a 500-candidate pool. Setting en->en.

Pools 20 / 60 / 200 / 500 on the held-out LoCoMo conversations (937 questions,
stores of 369 to 680 units). Quality runs on the GPU when available. CPU latency
is measured separately on the first 100 questions of zh->zh (OPC) and en->en
(English v1).

    python experiments/issue_1/pool_size_check.py
"""

from __future__ import annotations

import json
import sys
import time
from math import comb
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from m3_and_opc_check import settings  # noqa: E402

from convmemory import ConvMemory, OPCConvMemory  # noqa: E402

POOLS = [20, 60, 200, 500]
DOC_CAP = 400
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LATENCY_QUESTIONS = 100


def exact_mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def first_gold_rank(ids, gold):
    for r, mid in enumerate(ids, start=1):
        if mid in gold:
            return r
    return 0


def evaluate(model, convs, pools, latency_limit=0):
    """Return per-question gold ranks for dense and for the reranker at each pool size."""
    ranks = {"dense": [], **{f"pool{n}": [] for n in pools}}
    ceiling = {n: [] for n in pools}
    latency = {n: [] for n in pools}
    seen = 0
    for sid, memories, questions in convs:
        ids = [m["id"] for m in memories]
        texts = [m["text"][:DOC_CAP] for m in memories]
        vecs = model.encode(texts)
        qvecs = model.encode([q["query"] for q in questions])
        if hasattr(model, "prewarm_lexical"):
            model.prewarm_lexical([{"id": i, "text": t} for i, t in zip(ids, texts)])
        for k, q in enumerate(questions):
            gold = q["gold"]
            order = [int(i) for i in np.argsort(-(vecs @ qvecs[k]))]
            ranks["dense"].append(first_gold_rank([ids[i] for i in order[:10]], gold))
            for n in pools:
                cand = order[:n]
                ceiling[n].append(any(ids[i] in gold for i in cand))
                t0 = time.perf_counter()
                res = model.rerank_embeddings(query_embedding=qvecs[k], memory_embeddings=vecs, memory_ids=ids,
                                              memory_texts=texts, query=q["query"], candidate_indices=cand)
                if seen < latency_limit:
                    latency[n].append((time.perf_counter() - t0) * 1000)
                ranks[f"pool{n}"].append(first_gold_rank([r.memory_id for r in res], gold))
            seen += 1
            if latency_limit and seen >= latency_limit:
                return ranks, ceiling, latency
    return ranks, ceiling, latency


def summarize(name, ranks, ceiling, pools):
    hit = lambda a, k: np.array([0 < r <= k for r in ranks[a]])
    out = {"n": len(ranks["dense"]), "dense_hit@5": int(hit("dense", 5).sum()),
           "dense_hit@10": int(hit("dense", 10).sum()), "pools": {}}
    print(f"\n== {name}: n={out['n']}, dense hit@5 {out['dense_hit@5']} hit@10 {out['dense_hit@10']}")
    base = hit("pool60", 10)
    for n in pools:
        h5, h10 = hit(f"pool{n}", 5), hit(f"pool{n}", 10)
        d = hit("dense", 10)
        row = {"ceiling": int(np.sum(ceiling[n])), "hit@5": int(h5.sum()), "hit@10": int(h10.sum()),
               "vs_dense@10": [int((h10 & ~d).sum()), int((~h10 & d).sum()), exact_mcnemar(int((h10 & ~d).sum()), int((~h10 & d).sum()))],
               "vs_pool60@10": [int((h10 & ~base).sum()), int((~h10 & base).sum()), exact_mcnemar(int((h10 & ~base).sum()), int((~h10 & base).sum()))]}
        out["pools"][n] = row
        print(f"  pool {n:>3}: gold in pool {row['ceiling']:>4}  hit@5 {row['hit@5']:>4}  hit@10 {row['hit@10']:>4}"
              f"  vs dense {row['vs_dense@10'][0]}/{row['vs_dense@10'][1]} p={row['vs_dense@10'][2]:.3g}"
              f"  vs pool60 {row['vs_pool60@10'][0]}/{row['vs_pool60@10'][1]} p={row['vs_pool60@10'][2]:.3g}")
    return out


def main():
    data = settings()
    report = {"pools": POOLS, "device_quality": DEVICE, "opc": {}, "english_v1": {}, "cpu_latency_ms": {}}

    opc = OPCConvMemory.from_pretrained("Purdy0228/ConvMemory-OPC-Student-BGE", device=DEVICE)
    for name in ("zh->zh", "zh->en", "en->en"):
        ranks, ceiling, _ = evaluate(opc, data[name], POOLS)
        report["opc"][name] = summarize(f"OPC student, {name}", ranks, ceiling, POOLS)

    eng = ConvMemory.from_pretrained("Purdy0228/ConvMemory-LoCoMo-MPNet", device=DEVICE)
    ranks, ceiling, _ = evaluate(eng, data["en->en"], POOLS)
    report["english_v1"]["en->en"] = summarize("English v1, en->en", ranks, ceiling, POOLS)

    for label, cls, ckpt, setting in (("opc_zh->zh", OPCConvMemory, "Purdy0228/ConvMemory-OPC-Student-BGE", "zh->zh"),
                                      ("english_v1_en->en", ConvMemory, "Purdy0228/ConvMemory-LoCoMo-MPNet", "en->en")):
        model = cls.from_pretrained(ckpt, device="cpu")
        _, _, latency = evaluate(model, data[setting], POOLS, latency_limit=LATENCY_QUESTIONS)
        report["cpu_latency_ms"][label] = {n: float(np.median(v)) for n, v in latency.items()}
        print(f"\nCPU median ms per query, {label}:", {n: round(float(np.median(v)), 1) for n, v in latency.items()})

    (HERE / "pool_size_check_results.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print("\nsaved pool_size_check_results.json")


if __name__ == "__main__":
    main()

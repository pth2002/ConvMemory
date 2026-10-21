"""Follow-up checks for issue #1 (OpenViking).

A. The Chinese OPC student, handed the same candidate pool two ways:
   whole store in store order + candidate_indices, versus only the candidates
   in dense-rank order. Dense baseline = its own bge-base-zh space (as in the
   issue's §4a).
B. BGE-M3's own multi-vector (ColBERT) and sparse outputs as a fast reranker
   over a bge-m3 dense top-60 (the issue's pipeline), with memory-side vectors
   precomputed.

Three query/store settings on the held-out LoCoMo conversations (seed 23):
  zh->zh   Chinese questions, Chinese store   (data/locomo10_zh_v567.json)
  zh->en   Chinese questions, English store   (cross-lingual, the issue's "en" group)
  en->en   English questions, English store

M3 scoring follows FlagEmbedding's BGEM3 reference: dense = normalized CLS,
sparse = relu(sparse_linear(h)) max-pooled per token id without special tokens,
colbert = normalized colbert_linear(h[:, 1:]) with MaxSim averaged over query tokens.

    python experiments/issue_1/m3_and_opc_check.py              # full run
    python experiments/issue_1/m3_and_opc_check.py --selftest   # BGE-M3 check only

The Chinese file is our gpt-4o-mini translation of LoCoMo and is not
redistributed. Results from the run reported in issue #1 are in
m3_and_opc_check_results.json.
"""

from __future__ import annotations

import json
import sys
import time
from math import comb
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "examples"))
from demo_locomo import held_out_sample_ids, load_locomo  # noqa: E402

POOL = 60
DOC_CAP = 400          # the issue caps documents at 400 characters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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


# --------------------------------------------------------------------------- BGE-M3
class M3:
    def __init__(self, device):
        path = Path(snapshot_download("BAAI/bge-m3", allow_patterns=["*.json", "*.model", "colbert_linear.pt", "sparse_linear.pt", "pytorch_model.bin"]))
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()
        dim = self.model.config.hidden_size
        self.colbert = torch.nn.Linear(dim, dim)
        self.colbert.load_state_dict(torch.load(path / "colbert_linear.pt", map_location="cpu"))
        self.sparse = torch.nn.Linear(dim, 1)
        self.sparse.load_state_dict(torch.load(path / "sparse_linear.pt", map_location="cpu"))
        self.colbert.to(device, self.model.dtype).eval()
        self.sparse.to(device, self.model.dtype).eval()
        self.device = device
        self.special = {self.tok.cls_token_id, self.tok.eos_token_id, self.tok.pad_token_id, self.tok.unk_token_id}

    @torch.no_grad()
    def encode(self, texts, batch=32, max_length=512):
        dense, sparse, colbert = [], [], []
        for i in range(0, len(texts), batch):
            enc = self.tok(texts[i:i + batch], padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(self.device)
            h = self.model(**enc).last_hidden_state
            d = torch.nn.functional.normalize(h[:, 0].float(), dim=-1)
            w = torch.relu(self.sparse(h)).squeeze(-1).float()
            c = self.colbert(h[:, 1:]).float() * enc["attention_mask"][:, 1:, None]
            c = torch.nn.functional.normalize(c, dim=-1)
            for j in range(h.shape[0]):
                n = int(enc["attention_mask"][j].sum())
                dense.append(d[j].cpu().numpy())
                ids = enc["input_ids"][j, :n].tolist()
                wj = w[j, :n].tolist()
                sw = {}
                for t, x in zip(ids, wj):
                    if t in self.special or x <= 0:
                        continue
                    if x > sw.get(t, 0.0):
                        sw[t] = x
                sparse.append(sw)
                colbert.append(c[j, : n - 1].cpu().numpy().astype(np.float16))
        return np.stack(dense), sparse, colbert


def sparse_score(q, p):
    return sum(w * p[t] for t, w in q.items() if t in p)


def colbert_score(q, p):
    return float((q.astype(np.float32) @ p.astype(np.float32).T).max(axis=1).mean())


def selftest(m3):
    """Reproduce the example scores published in the BAAI/bge-m3 README."""
    s1 = ["What is BGE M3?", "Defination of BM25"]
    s2 = ["BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.",
          "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"]
    d1, w1, c1 = m3.encode(s1)
    d2, w2, c2 = m3.encode(s2)
    got = {
        "dense": np.round(d1 @ d2.T, 4).tolist(),
        "lexical_q0_p0": round(sparse_score(w1[0], w2[0]), 4),
        "lexical_q0_q1": round(sparse_score(w1[0], w1[1]), 4),
        "colbert_q0_p0": round(colbert_score(c1[0], c2[0]), 4),
        "colbert_q0_p1": round(colbert_score(c1[0], c2[1]), 4),
    }
    want = {"dense": [[0.6265, 0.3477], [0.3499, 0.678]], "lexical_q0_p0": 0.1955, "lexical_q0_q1": 0.0,
            "colbert_q0_p0": 0.7797, "colbert_q0_p1": 0.4620}
    print("selftest got ", got)
    print("selftest want", want)
    ok = (np.allclose(got["dense"], want["dense"], atol=0.01)
          and abs(got["lexical_q0_p0"] - want["lexical_q0_p0"]) < 0.01 and got["lexical_q0_q1"] == 0.0
          and abs(got["colbert_q0_p0"] - want["colbert_q0_p0"]) < 0.01 and abs(got["colbert_q0_p1"] - want["colbert_q0_p1"]) < 0.01)
    if not ok:
        raise SystemExit("BGE-M3 selftest failed")
    print("selftest passed")


# --------------------------------------------------------------------------- data
def settings():
    en = load_locomo(ROOT / "data" / "locomo10.json")
    zh = load_locomo(ROOT / "data" / "locomo10_zh_v567.json")
    for sid in en:
        assert [sorted(q["gold"]) for q in en[sid]["questions"]] == [sorted(q["gold"]) for q in zh[sid]["questions"]], sid
        assert [m["id"] for m in en[sid]["memories"]] == [m["id"] for m in zh[sid]["memories"]], sid
    held = held_out_sample_ids(en.keys(), 23, 0.5)
    out = {}
    for name, qsrc, msrc in (("zh->zh", zh, zh), ("zh->en", zh, en), ("en->en", en, en)):
        out[name] = [(sid, msrc[sid]["memories"], qsrc[sid]["questions"]) for sid in held]
    return out


# --------------------------------------------------------------------------- run
def main():
    data = settings()
    m3 = M3(DEVICE)
    selftest(m3)
    if "--selftest" in sys.argv:
        return
    from convmemory import OPCConvMemory
    # CPU, to match the issue's hardware class for the latency numbers
    opc = OPCConvMemory.from_pretrained("Purdy0228/ConvMemory-OPC-Student-BGE", device="cpu")

    report = {"pool": POOL, "doc_cap": DOC_CAP, "settings": {}}
    lat = {"colbert_ms": [], "sparse_ms": [], "opc_full_ms": [], "opc_rank_ms": []}
    for name, convs in data.items():
        ranks = {k: [] for k in ["m3_dense", "m3_sparse", "m3_colbert", "m3_all", "zh_dense", "opc_full", "opc_rank"]}
        in_pool_m3, in_pool_zh = [], []
        tok_counts = []
        for sid, memories, questions in convs:
            ids = [m["id"] for m in memories]
            texts = [m["text"][:DOC_CAP] for m in memories]
            md, ms, mc = m3.encode(texts)
            tok_counts += [len(x) for x in mc]
            qd, qs, qc = m3.encode([q["query"] for q in questions])
            zvec = opc.encode(texts)
            zq = opc.encode([q["query"] for q in questions])
            for k, q in enumerate(questions):
                gold = q["gold"]
                # B: bge-m3 dense top-60, then rerank inside it
                dense_scores = md @ qd[k]
                cand = [int(i) for i in np.argsort(-dense_scores)[:POOL]]
                in_pool_m3.append(any(ids[i] in gold for i in cand))
                ranks["m3_dense"].append(first_gold_rank([ids[i] for i in cand], gold))
                t0 = time.perf_counter()
                sp = np.array([sparse_score(qs[k], ms[i]) for i in cand])
                lat["sparse_ms"].append((time.perf_counter() - t0) * 1000)
                t0 = time.perf_counter()
                cb = np.array([colbert_score(qc[k], mc[i]) for i in cand])
                lat["colbert_ms"].append((time.perf_counter() - t0) * 1000)
                dn = dense_scores[cand]
                for arm, s in (("m3_sparse", sp), ("m3_colbert", cb), ("m3_all", 0.4 * dn + 0.2 * sp + 0.4 * cb)):
                    order = np.argsort(-s, kind="stable")
                    ranks[arm].append(first_gold_rank([ids[cand[i]] for i in order], gold))
                # A: OPC student in its own bge-base-zh space
                zs = zvec @ zq[k]
                zc = [int(i) for i in np.argsort(-zs)[:POOL]]
                in_pool_zh.append(any(ids[i] in gold for i in zc))
                ranks["zh_dense"].append(first_gold_rank([ids[i] for i in zc], gold))
                t0 = time.perf_counter()
                r = opc.rerank_embeddings(query_embedding=zq[k], memory_embeddings=zvec, memory_ids=ids,
                                          memory_texts=texts, query=q["query"], candidate_indices=zc)
                lat["opc_full_ms"].append((time.perf_counter() - t0) * 1000)
                ranks["opc_full"].append(first_gold_rank([x.memory_id for x in r], gold))
                t0 = time.perf_counter()
                r = opc.rerank_embeddings(query_embedding=zq[k], memory_embeddings=zvec[zc], memory_ids=[ids[i] for i in zc],
                                          memory_texts=[texts[i] for i in zc], query=q["query"])
                lat["opc_rank_ms"].append((time.perf_counter() - t0) * 1000)
                ranks["opc_rank"].append(first_gold_rank([x.memory_id for x in r], gold))
            print(f"{name} {sid}: {len(memories)} memories, {len(questions)} questions", flush=True)

        n = len(ranks["m3_dense"])
        hit = lambda a, k: np.array([0 < r <= k for r in ranks[a]])
        mrr = lambda a: float(np.mean([1 / r if r else 0 for r in ranks[a]]))
        rep = {"n": n, "ceiling_m3@60": float(np.mean(in_pool_m3)), "ceiling_zh@60": float(np.mean(in_pool_zh)),
               "memory_tokens_mean": float(np.mean(tok_counts)), "arms": {}, "paired@10": {}}
        print(f"\n== {name}: n={n}, gold in bge-m3 top-60 {np.mean(in_pool_m3):.1%}, in bge-base-zh top-60 {np.mean(in_pool_zh):.1%}")
        for arm in ranks:
            rep["arms"][arm] = {"hit@5": int(hit(arm, 5).sum()), "hit@10": int(hit(arm, 10).sum()), "mrr": mrr(arm)}
            print(f"  {arm:<11} hit@5 {hit(arm, 5).sum():>4} ({hit(arm, 5).mean():5.1%})  hit@10 {hit(arm, 10).sum():>4} ({hit(arm, 10).mean():5.1%})  MRR {mrr(arm):.3f}")
        for a, b in (("m3_colbert", "m3_dense"), ("m3_all", "m3_dense"), ("m3_sparse", "m3_dense"),
                     ("opc_full", "zh_dense"), ("opc_rank", "zh_dense"), ("opc_full", "opc_rank")):
            x, y = hit(a, 10), hit(b, 10)
            hb, hc = int((x & ~y).sum()), int((~x & y).sum())
            rep["paired@10"][f"{a}_vs_{b}"] = {"a_only": hb, "b_only": hc, "p": exact_mcnemar(hb, hc)}
            print(f"  {a} vs {b} @10: {hb}/{hc}, p={exact_mcnemar(hb, hc):.3g}")
        report["settings"][name] = rep

    report["latency_ms_mean"] = {k: float(np.mean(v)) for k, v in lat.items()}
    report["device"] = DEVICE
    print("\nmean latency (ms per query, 60 candidates, vectors precomputed):", {k: round(v, 2) for k, v in report["latency_ms_mean"].items()})
    (HERE / "m3_and_opc_check_results.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print("saved m3_and_opc_check_results.json")


if __name__ == "__main__":
    main()

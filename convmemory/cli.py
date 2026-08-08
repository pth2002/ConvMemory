"""Command line interface for ConvMemory.

    convmemory benchmark --queries queries.jsonl --memories memories.jsonl

Runs your own memory store through dense retrieval and through dense +
ConvMemory, and prints Recall@k, MRR and latency for both. The point is that you
do not have to trust the numbers in the README.

Scope: the "before" row is cosine similarity in the loaded checkpoint's own
embedding space, because that is the space ConvMemory can score in. It is not a
stand-in for a production retriever built on a different embedding model. Read
the result as "what the reranker adds on top of this space"; carrying it to
another space means retraining there.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .metrics import hit_at_k, mrr, recall_at_k


def _read_jsonl(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise SystemExit(f"{path}:{line_number}: invalid JSON ({error})")
    if not records:
        raise SystemExit(f"{path}: no records found")
    return records


def _load_memories(path: Path):
    """memories.jsonl -> {group: [{'id':..., 'text':...}, ...]} in file order."""
    groups = defaultdict(list)
    seen = set()
    for index, record in enumerate(_read_jsonl(path)):
        if "text" not in record:
            raise SystemExit(f"{path}: record {index} has no 'text' field")
        memory_id = str(record.get("id", index))
        group = str(record.get("group", "__all__"))
        if (group, memory_id) in seen:
            raise SystemExit(f"{path}: duplicate memory id {memory_id!r} in group {group!r}")
        seen.add((group, memory_id))
        groups[group].append({"id": memory_id, "text": str(record["text"])})
    return groups


def _load_queries(path: Path, limit=None):
    queries = []
    for index, record in enumerate(_read_jsonl(path)):
        if "query" not in record:
            raise SystemExit(f"{path}: record {index} has no 'query' field")
        gold = record.get("gold_ids", record.get("gold_id"))
        if gold is None:
            raise SystemExit(f"{path}: record {index} has no 'gold_ids' field")
        if isinstance(gold, (str, int)):
            gold = [gold]
        queries.append(
            {
                "query": str(record["query"]),
                "gold_ids": [str(g) for g in gold],
                "group": str(record.get("group", "__all__")),
            }
        )
        if limit is not None and len(queries) >= limit:
            break
    return queries


class _Accumulator:
    def __init__(self, ks):
        self.ks = ks
        self.recall = {k: 0.0 for k in ks}
        self.hit = {k: 0.0 for k in ks}
        self.mrr = 0.0
        self.n = 0

    def add(self, ranked_ids, gold_ids):
        for k in self.ks:
            self.recall[k] += recall_at_k(ranked_ids, gold_ids, k)
            self.hit[k] += hit_at_k(ranked_ids, gold_ids, k)
        self.mrr += mrr(ranked_ids, gold_ids)
        self.n += 1

    def summary(self):
        n = max(1, self.n)
        out = {"questions": self.n, "mrr": self.mrr / n}
        for k in self.ks:
            out[f"recall@{k}"] = self.recall[k] / n
            out[f"hit@{k}"] = self.hit[k] / n
        return out


def _print_block(title, summary, ks, seconds_per_query):
    print(f"\n{title}")
    for k in ks:
        print(f"  Recall@{k:<3} {summary[f'recall@{k}']:.4f}     Hit@{k:<3} {summary[f'hit@{k}']:.4f}")
    print(f"  MRR       {summary['mrr']:.4f}")
    print(f"  Latency   {seconds_per_query * 1000:.2f} ms/query")


def _delta_line(label, before, after, higher_is_better=True):
    delta = after - before
    arrow = "+" if delta >= 0 else ""
    verdict = ""
    if abs(delta) >= 1e-9:
        good = delta > 0 if higher_is_better else delta < 0
        verdict = "  better" if good else "  worse"
    return f"  {label:<12} {before:.4f}  ->  {after:.4f}   ({arrow}{delta:.4f}){verdict}"


def run_benchmark(args) -> int:
    from .api import ConvMemory

    queries_path = Path(args.queries)
    memories_path = Path(args.memories)
    for path in (queries_path, memories_path):
        if not path.exists():
            raise SystemExit(f"{path}: file not found")

    groups = _load_memories(memories_path)
    queries = _load_queries(queries_path, limit=args.limit)

    missing = {q["group"] for q in queries} - set(groups)
    if missing:
        raise SystemExit(
            "queries reference memory groups that are not in the memory file: "
            + ", ".join(sorted(missing)[:5])
        )

    ks = sorted({int(k) for k in args.k})
    pool_sizes = [len(v) for v in groups.values()]
    print(f"memories: {sum(pool_sizes)} in {len(groups)} group(s), "
          f"pool size {min(pool_sizes)}-{max(pool_sizes)}")
    print(f"queries:  {len(queries)}")
    print(f"loading {args.checkpoint} on {args.device} ...")

    model = ConvMemory.from_pretrained(args.checkpoint, device=args.device)

    encoded = {}
    encode_started = time.perf_counter()
    for group, memories in groups.items():
        encoded[group] = model.encode([m["text"] for m in memories])
        model.prewarm_lexical(memories)
    index_seconds = time.perf_counter() - encode_started
    print(f"indexed {sum(pool_sizes)} memories in {index_seconds:.1f}s "
          f"(one-off, not counted in per-query latency)")

    dense_acc = _Accumulator(ks)
    conv_acc = _Accumulator(ks)
    dense_seconds = 0.0
    conv_seconds = 0.0
    encode_seconds = 0.0
    max_k = max(max(ks), args.top_k)

    for position, item in enumerate(queries, start=1):
        memories = groups[item["group"]]
        memory_vecs = encoded[item["group"]]
        ids = [m["id"] for m in memories]
        texts = [m["text"] for m in memories]

        started = time.perf_counter()
        query_vec = model.encode([item["query"]])[0]
        encode_seconds += time.perf_counter() - started

        started = time.perf_counter()
        order = np.argsort(-(memory_vecs @ query_vec))[: max(max_k, args.candidate_pool)]
        dense_seconds += time.perf_counter() - started
        dense_ids = [ids[i] for i in order]

        candidate_indices = [int(i) for i in order[: args.candidate_pool]]
        started = time.perf_counter()
        ranked = model.rerank_embeddings(
            query_embedding=query_vec,
            memory_embeddings=memory_vecs,
            memory_ids=ids,
            memory_texts=texts,
            query=item["query"],
            candidate_indices=candidate_indices,
            top_k=max_k,
        )
        conv_seconds += time.perf_counter() - started
        conv_ids = [result.memory_id for result in ranked]

        dense_acc.add(dense_ids, item["gold_ids"])
        conv_acc.add(conv_ids, item["gold_ids"])

        if args.progress and position % args.progress == 0:
            print(f"  ... {position}/{len(queries)}", flush=True)

    n = max(1, len(queries))
    dense_summary = dense_acc.summary()
    conv_summary = conv_acc.summary()

    embedding_space = getattr(model, "embedding_model_name", None) or "the checkpoint's embedding space"
    print(f"\nquery encoding (shared by both paths): {encode_seconds / n * 1000:.2f} ms/query")
    _print_block(
        f"Before ConvMemory (dense only, {embedding_space})", dense_summary, ks, dense_seconds / n
    )
    _print_block("After ConvMemory", conv_summary, ks, conv_seconds / n)

    print("\nDelta")
    for k in ks:
        print(_delta_line(f"Recall@{k}", dense_summary[f"recall@{k}"], conv_summary[f"recall@{k}"]))
    print(_delta_line("MRR", dense_summary["mrr"], conv_summary["mrr"]))
    print(
        f"  {'Latency':<12} {dense_seconds / n * 1000:.2f} ms  ->  "
        f"{conv_seconds / n * 1000:.2f} ms"
    )

    print(
        f"\nHow to read this: the 'before' row is cosine similarity in {embedding_space} --"
        "\nthe space this checkpoint scores in, not your production retriever. If you retrieve"
        "\nwith a different embedding model, this delta is not a forecast for your stack: the"
        "\ncheckpoint only scores in its own space, so moving spaces means retraining. That"
        "\ngain does travel -- retrained on BGE-large and E5-large, ConvMemory held +0.089 to"
        "\n+0.105 Recall@10 over raw dense in those spaces (docs/BENCHMARKS.md)."
    )

    if min(pool_sizes) < 100:
        print(
            "\nNote: your candidate pools are small. Reranking mostly pays off when "
            "dense recall is degrading over hundreds of candidates."
        )

    if args.json:
        payload = {
            "checkpoint": args.checkpoint,
            "device": args.device,
            "baseline_embedding_space": embedding_space,
            "baseline_caveat": (
                "The 'before' row is cosine similarity in the checkpoint's own embedding "
                "space, not the reporter's production retriever. The released checkpoint "
                "scores only in that space; using a different retriever requires retraining. "
                "Retrained on BGE-large and E5-large, ConvMemory held +0.089 to +0.105 "
                "Recall@10 over raw dense in those spaces."
            ),
            "queries": len(queries),
            "memory_groups": len(groups),
            "pool_size_min": min(pool_sizes),
            "pool_size_max": max(pool_sizes),
            "candidate_pool": args.candidate_pool,
            "index_seconds": index_seconds,
            "query_encode_ms": encode_seconds / n * 1000,
            "before": {**dense_summary, "ms_per_query": dense_seconds / n * 1000},
            "after": {**conv_summary, "ms_per_query": conv_seconds / n * 1000},
        }
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")

    return 0


def build_parser():
    parser = argparse.ArgumentParser(prog="convmemory", description=__doc__.split("\n")[0])
    subparsers = parser.add_subparsers(dest="command")

    benchmark = subparsers.add_parser(
        "benchmark",
        help="measure retrieval quality before and after ConvMemory on your own data",
    )
    benchmark.add_argument("--queries", required=True, help="JSONL: query, gold_ids, optional group")
    benchmark.add_argument("--memories", required=True, help="JSONL: id, text, optional group")
    benchmark.add_argument("--checkpoint", default="Purdy0228/ConvMemory-LoCoMo-MPNet")
    benchmark.add_argument("--device", default="cpu")
    benchmark.add_argument("--k", nargs="+", default=[5, 10], help="cutoffs for Recall@k / Hit@k")
    benchmark.add_argument("--top-k", type=int, default=10, help="results kept from the reranker")
    benchmark.add_argument(
        "--candidate-pool",
        type=int,
        default=500,
        help="how many dense candidates ConvMemory reranks",
    )
    benchmark.add_argument("--limit", type=int, default=None, help="only run the first N queries")
    benchmark.add_argument("--progress", type=int, default=0, help="print progress every N queries")
    benchmark.add_argument("--json", default=None, help="write the summary to this JSON file")
    benchmark.set_defaults(func=run_benchmark)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import CrossEncoder

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from convmemory import ConvMemory, CompressionRouteConfig, CompressionRouter
from convmemory.metrics import hit_at_k, mrr, recall_at_k
from experiments.support.convmem_chain_benchmark import prepare_encoded_example
from experiments.support.convmem_locomo_benchmark import load_locomo_examples
from experiments.support.convmem_longmemeval import SentenceTransformerTextEncoder, resolve_local_model_path
from experiments.support.locomo_notes import encode_memory_bank, load_locomo_memory_sets
from experiments.support.locomo_crossencoder_baseline import choose_split


ROUTE_PRESETS = {
    "fast": CompressionRouteConfig(
        note_depth=240,
        max_sources_per_note=5,
        max_candidates=450,
        raw_anchor=80,
    ),
    "balanced": CompressionRouteConfig(
        note_depth=240,
        max_sources_per_note=8,
        max_candidates=450,
        raw_anchor=50,
    ),
}


def build_note_banks(banks):
    obs = [dict(item, note_type="observation") for item in banks["observation_notes"]]
    events = [dict(item, note_type="event") for item in banks["event_notes"]]
    sessions = [dict(item, note_type="session") for item in banks["session_summaries"]]
    return {"all": obs + events + sessions}


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sync(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(device, fn):
    sync(device)
    start = time.perf_counter()
    value = fn()
    sync(device)
    return value, time.perf_counter() - start


def ranked_ids(results):
    return [result.memory_id for result in results]


def append_remaining(primary, fallback):
    seen = set(primary)
    return list(primary) + [memory_id for memory_id in fallback if memory_id not in seen]


def zscore(values):
    values = np.asarray(values, dtype=np.float32)
    std = float(values.std())
    if std < 1e-8:
        return values - float(values.mean())
    return (values - float(values.mean())) / std


def cross_encoder_rank(cross_encoder, query, memories, indices, batch_size):
    pairs = [(query, memories[int(idx)]["text"]) for idx in indices]
    scores = np.asarray(
        cross_encoder.predict(pairs, batch_size=batch_size, show_progress_bar=False),
        dtype=np.float32,
    )
    order = np.argsort(-scores)
    return [int(indices[int(i)]) for i in order], scores


def route_indices(item, sample_id, note_banks, encoded_notes, route_name):
    route_config = ROUTE_PRESETS[route_name]
    routed = CompressionRouter(route_config).route(
        query_embedding=item["query_embedding"],
        memory_embeddings=item["memory_embeddings"],
        memory_ids=item["memory_ids"],
        compressed_embeddings=encoded_notes[sample_id]["all"],
        compressed_memories=note_banks[sample_id]["all"],
    )
    indices = np.asarray(routed.candidate_indices, dtype=np.int64)
    if len(indices) == 0:
        raw_scores = item["memory_embeddings"] @ item["query_embedding"]
        indices = np.argsort(-raw_scores)[: route_config.raw_anchor]
    return indices


def add_row(rows, method, seed, item, ranked, elapsed, ce_pairs, candidate_count):
    rows.append(
        {
            "seed": seed,
            "method": method,
            "question_id": item["question_id"],
            "question_type": item["question_type"],
            "latency_ms": elapsed * 1000.0,
            "ce_pairs": int(ce_pairs),
            "candidate_count": int(candidate_count),
            "recall_at_10": recall_at_k(ranked, item["gold_memory_ids"], 10),
            "hit_at_10": hit_at_k(ranked, item["gold_memory_ids"], 10),
            "mrr": mrr(ranked[:10], item["gold_memory_ids"]),
        }
    )


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["method"], []).append(row)
    out = []
    for method, items in sorted(groups.items()):
        latencies = np.asarray([float(x["latency_ms"]) for x in items], dtype=np.float32)
        out.append(
            {
                "method": method,
                "questions": len(items),
                "avg_ce_pairs": float(np.mean([float(x["ce_pairs"]) for x in items])),
                "avg_candidates": float(np.mean([float(x["candidate_count"]) for x in items])),
                "recall_at_10": float(np.mean([float(x["recall_at_10"]) for x in items])),
                "hit_at_10": float(np.mean([float(x["hit_at_10"]) for x in items])),
                "mrr": float(np.mean([float(x["mrr"]) for x in items])),
                "mean_ms": float(np.mean(latencies)),
                "p50_ms": float(np.percentile(latencies, 50)),
                "p95_ms": float(np.percentile(latencies, 95)),
                "p99_ms": float(np.percentile(latencies, 99)),
                "queries_per_second": float(1000.0 / max(1e-9, float(np.mean(latencies)))),
            }
        )
    base = next((row for row in out if row["method"] == "cross_encoder_raw_top500"), None)
    if base:
        for row in out:
            row["speedup_vs_ce_raw_top500"] = base["mean_ms"] / max(1e-9, row["mean_ms"])
            row["ce_pair_ratio_vs_top500"] = row["avg_ce_pairs"] / max(1e-9, base["avg_ce_pairs"])
            row["delta_recall_vs_ce_raw_top500"] = row["recall_at_10"] - base["recall_at_10"]
            row["delta_mrr_vs_ce_raw_top500"] = row["mrr"] - base["mrr"]
    return sorted(out, key=lambda row: row["mean_ms"])


def write_report(path, summary, limit, warmup, args):
    focus = [
        "raw_vector",
        "convmemory_balanced_halo0",
        "convmemory_full_top500",
        "cross_encoder_raw_top500",
        "cross_encoder_raw_top100",
        "cascade_ce_full_top50",
        "cascade_fusion_full_top100_alpha0.40",
    ]
    by_method = {row["method"]: row for row in summary}
    lines = [
        "# v0.36 Latency Benchmark",
        "",
        f"Measured online reranking latency after embedding/note indexes are available. Warmup queries: {warmup}. Measured query cap per seed: {limit or 'all'}.",
        "",
        "Cross-encoder timings use `sentence-transformers.CrossEncoder.predict`, so query-candidate tokenization is included in the measured call. Query embedding and memory-side indexing are not included. Batch sizes are reported here to make serving assumptions explicit.",
        "",
        f"- Encoder batch size: {args.encoder_batch_size}",
        f"- Cross-encoder batch size: {args.cross_batch_size}",
        f"- Device: {args.device}",
        "",
        "| Method | Recall@10 | Hit@10 | MRR@10 | Mean ms | P50 ms | P95 ms | P99 ms | QPS | CE pairs | Speedup vs CE top500 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in focus:
        row = by_method.get(method)
        if not row:
            continue
        lines.append(
            f"| `{method}` | {row['recall_at_10']:.4f} | {row['hit_at_10']:.4f} | "
            f"{row['mrr']:.4f} | {row['mean_ms']:.1f} | {row['p50_ms']:.1f} | "
            f"{row['p95_ms']:.1f} | {row['p99_ms']:.1f} | {row['queries_per_second']:.1f} | "
            f"{row['avg_ce_pairs']:.1f} | "
            f"{row.get('speedup_vs_ce_raw_top500', 0.0):.2f}x |"
        )
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


def evaluate_item(seed, item, sample_id, note_banks, encoded_notes, convmemory, cross_encoder, args, device):
    rows = []
    memory_ids = item["memory_ids"]
    memories = item["memories"]
    id_to_idx = {memory_id: idx for idx, memory_id in enumerate(memory_ids)}

    def raw_step():
        scores = item["memory_embeddings"] @ item["query_embedding"]
        order = np.argsort(-scores)
        return order, [memory_ids[int(idx)] for idx in order]

    (raw_order, raw_ranked), raw_elapsed = timed(device, raw_step)
    add_row(rows, "raw_vector", seed, item, raw_ranked, raw_elapsed, 0, len(memory_ids))

    def conv_full_step():
        return convmemory.reranker.rerank_item(
            item,
            candidate_top_n=args.raw_top_n,
            window_mode="candidate_local",
        )

    conv_full, conv_full_elapsed = timed(device, conv_full_step)
    conv_full_ranked = ranked_ids(conv_full)
    add_row(
        rows,
        f"convmemory_full_top{args.raw_top_n}",
        seed,
        item,
        conv_full_ranked,
        conv_full_elapsed,
        0,
        min(args.raw_top_n, len(memory_ids)),
    )

    def conv_balanced_step():
        indices = route_indices(item, sample_id, note_banks, encoded_notes, "balanced")
        results = convmemory.reranker.rerank_item(
            item,
            candidate_indices=indices,
            window_mode="candidate_local",
        )
        return indices, results

    (balanced_indices, conv_balanced), conv_balanced_elapsed = timed(device, conv_balanced_step)
    conv_balanced_ranked = ranked_ids(conv_balanced)
    add_row(
        rows,
        "convmemory_balanced_halo0",
        seed,
        item,
        conv_balanced_ranked,
        conv_balanced_elapsed,
        0,
        len(balanced_indices),
    )

    raw_top500 = raw_order[: min(args.raw_top_n, len(raw_order))]
    raw_top100 = raw_order[: min(100, len(raw_order))]

    def ce_raw500_step():
        order, _ = cross_encoder_rank(cross_encoder, item["query"], memories, raw_top500, args.cross_batch_size)
        return [memory_ids[int(idx)] for idx in order]

    ce_raw500_ranked, ce_raw500_elapsed = timed(device, ce_raw500_step)
    add_row(
        rows,
        "cross_encoder_raw_top500",
        seed,
        item,
        append_remaining(ce_raw500_ranked, raw_ranked),
        ce_raw500_elapsed,
        len(raw_top500),
        len(raw_top500),
    )

    def ce_raw100_step():
        order, _ = cross_encoder_rank(cross_encoder, item["query"], memories, raw_top100, args.cross_batch_size)
        return [memory_ids[int(idx)] for idx in order]

    ce_raw100_ranked, ce_raw100_elapsed = timed(device, ce_raw100_step)
    add_row(
        rows,
        "cross_encoder_raw_top100",
        seed,
        item,
        append_remaining(ce_raw100_ranked, raw_ranked),
        ce_raw100_elapsed,
        len(raw_top100),
        len(raw_top100),
    )

    conv_full_top50_ids = conv_full_ranked[:50]
    conv_full_top100_ids = conv_full_ranked[:100]
    conv_full_top50_idx = np.asarray([id_to_idx[memory_id] for memory_id in conv_full_top50_ids], dtype=np.int64)
    conv_full_top100_idx = np.asarray([id_to_idx[memory_id] for memory_id in conv_full_top100_ids], dtype=np.int64)
    conv_score = {result.memory_id: float(result.score) for result in conv_full}

    def ce_full50_step():
        order, _ = cross_encoder_rank(cross_encoder, item["query"], memories, conv_full_top50_idx, args.cross_batch_size)
        return [memory_ids[int(idx)] for idx in order]

    ce_full50_ranked, ce_full50_elapsed = timed(device, ce_full50_step)
    add_row(
        rows,
        "cascade_ce_full_top50",
        seed,
        item,
        append_remaining(ce_full50_ranked, conv_full_ranked),
        conv_full_elapsed + ce_full50_elapsed,
        len(conv_full_top50_idx),
        len(conv_full_top50_idx),
    )

    def fusion_full100_step():
        order, ce_scores = cross_encoder_rank(cross_encoder, item["query"], memories, conv_full_top100_idx, args.cross_batch_size)
        ce_by_id = {memory_ids[int(idx)]: float(score) for idx, score in zip(conv_full_top100_idx, ce_scores)}
        conv_values = np.asarray([conv_score[memory_id] for memory_id in conv_full_top100_ids], dtype=np.float32)
        ce_values = np.asarray([ce_by_id[memory_id] for memory_id in conv_full_top100_ids], dtype=np.float32)
        fused = (1.0 - args.alpha) * zscore(conv_values) + args.alpha * zscore(ce_values)
        ranked = [conv_full_top100_ids[int(idx)] for idx in np.argsort(-fused)]
        return ranked

    fusion_top100_ranked, fusion_top100_elapsed = timed(device, fusion_full100_step)
    add_row(
        rows,
        f"cascade_fusion_full_top100_alpha{args.alpha:.2f}",
        seed,
        item,
        append_remaining(fusion_top100_ranked, conv_full_ranked),
        conv_full_elapsed + fusion_top100_elapsed,
        len(conv_full_top100_idx),
        len(conv_full_top100_idx),
    )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/locomo10.json")
    parser.add_argument("--checkpoint", default="checkpoints/convmemory-locomo-mpnet")
    parser.add_argument("--encoder-model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--embedding-cache", default=None)
    parser.add_argument("--embedding-cache-key", default=None)
    parser.add_argument("--cross-encoder-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 11, 23])
    parser.add_argument("--dev-ratio", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--raw-top-n", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--encoder-batch-size", type=int, default=128)
    parser.add_argument("--cross-batch-size", type=int, default=512)
    parser.add_argument("--out", default="results/v036/latency_benchmark")
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    samples, _ = load_locomo_memory_sets(ROOT / args.data)
    examples = load_locomo_examples(ROOT / args.data)
    convmemory = ConvMemory.from_pretrained(ROOT / args.checkpoint, device=device, embedding_model=False)
    encoder = SentenceTransformerTextEncoder(
        model_name=args.encoder_model,
        device=device,
        batch_size=args.encoder_batch_size,
        cache_path=args.embedding_cache,
        cache_key=args.embedding_cache_key or args.encoder_model,
    )
    cross_encoder = CrossEncoder(resolve_local_model_path(args.cross_encoder_model), device=device)
    note_banks = {}
    encoded_notes = {}
    for sample_id, banks in samples.items():
        note_banks[sample_id] = build_note_banks(banks)
        encoded_notes[sample_id] = {
            name: encode_memory_bank(encoder, notes)
            for name, notes in note_banks[sample_id].items()
        }

    rows = []
    for seed in args.seeds:
        split_examples = choose_split(examples, "test", args.dev_ratio, seed)
        total = len(split_examples) if args.limit <= 0 else min(args.limit + args.warmup, len(split_examples))
        selected_examples = split_examples[:total]
        for idx, example in enumerate(selected_examples, start=1):
            sample_id = example["question_id"].split("::", 1)[0]
            item = prepare_encoded_example(example, encoder, convmemory.config.window_size, convmemory.config.stride)
            item_rows = evaluate_item(
                seed,
                item,
                sample_id,
                note_banks,
                encoded_notes,
                convmemory,
                cross_encoder,
                args,
                device,
            )
            if idx > args.warmup:
                rows.extend(item_rows)
            if idx % 20 == 0 or idx == len(selected_examples):
                measured = max(0, idx - args.warmup)
                print(f"seed {seed}: processed {idx}/{len(selected_examples)} measured={measured}", flush=True)

    summary = summarize(rows)
    out_dir = ROOT / args.out
    write_csv(out_dir / "latency_detailed.csv", rows)
    write_csv(out_dir / "latency_summary.csv", summary)
    write_report(out_dir / "REPORT.md", summary, args.limit, args.warmup, args)


if __name__ == "__main__":
    main()

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

from convmemory import ConvMemory
from convmemory.metrics import hit_at_k, mrr, recall_at_k
from experiments.support.convmem_chain_benchmark import prepare_encoded_example
from experiments.support.convmem_locomo_benchmark import load_locomo_examples
from experiments.support.convmem_longmemeval import SentenceTransformerTextEncoder, resolve_local_model_path
from experiments.support.locomo_crossencoder_baseline import choose_split


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def add_metrics(rows, method, item, ranked_ids, elapsed_s, candidate_count):
    rows.append(
        {
            "method": method,
            "question_id": item["question_id"],
            "question_type": item["question_type"],
            "candidate_count": candidate_count,
            "latency_ms": elapsed_s * 1000.0,
            "recall_at_10": recall_at_k(ranked_ids, item["gold_memory_ids"], 10),
            "hit_at_10": hit_at_k(ranked_ids, item["gold_memory_ids"], 10),
            "mrr": mrr(ranked_ids, item["gold_memory_ids"]),
        }
    )


def summarize(rows):
    by_method = {}
    for row in rows:
        by_method.setdefault(row["method"], []).append(row)
    summary = []
    for method, items in by_method.items():
        summary.append(
            {
                "method": method,
                "questions": len(items),
                "avg_candidates": float(np.mean([float(x["candidate_count"]) for x in items])),
                "recall_at_10": float(np.mean([float(x["recall_at_10"]) for x in items])),
                "hit_at_10": float(np.mean([float(x["hit_at_10"]) for x in items])),
                "mrr": float(np.mean([float(x["mrr"]) for x in items])),
                "ms_per_query": float(np.mean([float(x["latency_ms"]) for x in items])),
            }
        )
    return sorted(summary, key=lambda x: x["ms_per_query"])


def cross_encoder_rank(cross_encoder, query, memories, indices, batch_size):
    pairs = [(query, memories[int(i)]["text"]) for i in indices]
    scores = cross_encoder.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=False,
    )
    order = [int(indices[i]) for i in np.argsort(-np.asarray(scores, dtype=np.float32))]
    return order


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/locomo10.json")
    parser.add_argument("--checkpoint", default="checkpoints/convmemory-locomo-mpnet")
    parser.add_argument("--encoder-model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--embedding-cache", default=None)
    parser.add_argument("--embedding-cache-key", default=None)
    parser.add_argument("--cross-encoder-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--candidate-top-n", type=int, default=500)
    parser.add_argument("--cascade-top-n", type=int, default=50)
    parser.add_argument("--encoder-batch-size", type=int, default=128)
    parser.add_argument("--cross-batch-size", type=int, default=64)
    parser.add_argument("--out", default="results/benchmark_cost")
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if args.device == "auto" and device != "cuda":
        device = "cpu"

    convmemory = ConvMemory.from_pretrained(
        args.checkpoint,
        device=device,
        embedding_model=False,
    )
    encoder = SentenceTransformerTextEncoder(
        model_name=args.encoder_model,
        device=device,
        batch_size=args.encoder_batch_size,
        cache_path=args.embedding_cache,
        cache_key=args.embedding_cache_key,
    )
    cross_encoder = CrossEncoder(resolve_local_model_path(args.cross_encoder_model), device=device)

    examples = load_locomo_examples(Path(args.data))
    test_examples = choose_split(examples, "test", 0.5, args.seed)
    if args.limit:
        test_examples = test_examples[: args.limit]

    rows = []
    for idx, example in enumerate(test_examples, start=1):
        item = prepare_encoded_example(
            example,
            encoder,
            convmemory.config.window_size,
            convmemory.config.stride,
        )
        memory_ids = item["memory_ids"]
        memories = item["memories"]
        query = item["query"]

        start = time.perf_counter()
        raw_scores = item["memory_embeddings"] @ item["query_embedding"]
        raw_order = np.argsort(-raw_scores)
        raw_ranked = [memory_ids[int(i)] for i in raw_order]
        raw_elapsed = time.perf_counter() - start
        add_metrics(rows, "raw_vector", item, raw_ranked, raw_elapsed, len(memory_ids))

        start = time.perf_counter()
        conv_results = convmemory.reranker.rerank_item(
            item,
            candidate_top_n=args.candidate_top_n,
        )
        conv_elapsed = time.perf_counter() - start
        conv_ranked = [x.memory_id for x in conv_results]
        add_metrics(
            rows,
            f"convmemory_top{args.candidate_top_n}",
            item,
            conv_ranked,
            conv_elapsed,
            min(args.candidate_top_n, len(memory_ids)),
        )

        cross_top_n = min(args.candidate_top_n, len(raw_order))
        cross_indices = raw_order[:cross_top_n]
        start = time.perf_counter()
        cross_order = cross_encoder_rank(
            cross_encoder,
            query,
            memories,
            cross_indices,
            args.cross_batch_size,
        )
        cross_elapsed = time.perf_counter() - start
        cross_ranked = [memory_ids[i] for i in cross_order]
        cross_seen = set(cross_ranked)
        cross_ranked.extend([x for x in raw_ranked if x not in cross_seen])
        add_metrics(
            rows,
            f"cross_encoder_top{args.candidate_top_n}",
            item,
            cross_ranked,
            cross_elapsed,
            cross_top_n,
        )

        conv_id_to_idx = {memory_id: i for i, memory_id in enumerate(memory_ids)}
        cascade_ids = conv_ranked[: min(args.cascade_top_n, len(conv_ranked))]
        cascade_indices = np.asarray([conv_id_to_idx[x] for x in cascade_ids], dtype=np.int64)
        start = time.perf_counter()
        cascade_order = cross_encoder_rank(
            cross_encoder,
            query,
            memories,
            cascade_indices,
            args.cross_batch_size,
        )
        ce50_elapsed = time.perf_counter() - start
        cascade_ranked = [memory_ids[i] for i in cascade_order]
        cascade_seen = set(cascade_ranked)
        cascade_ranked.extend([x for x in conv_ranked if x not in cascade_seen])
        add_metrics(
            rows,
            f"convmemory_top{args.candidate_top_n}_plus_ce_top{args.cascade_top_n}",
            item,
            cascade_ranked,
            conv_elapsed + ce50_elapsed,
            len(cascade_indices),
        )

        if idx % 10 == 0:
            print(f"benchmarked {idx}/{len(test_examples)}", flush=True)

    out_dir = Path(args.out)
    summary = summarize(rows)
    write_csv(out_dir / "cost_detailed.csv", rows)
    write_csv(out_dir / "cost_summary.csv", summary)

    print("\nCost benchmark")
    print(f"device: {device}")
    print(f"questions: {len(test_examples)}")
    print("method                                      recall@10 hit@10 mrr   ms/query")
    for row in summary:
        print(
            f"{row['method']:<43} "
            f"{row['recall_at_10']:.3f}     "
            f"{row['hit_at_10']:.3f}  "
            f"{row['mrr']:.3f} "
            f"{row['ms_per_query']:.1f}"
        )


if __name__ == "__main__":
    main()

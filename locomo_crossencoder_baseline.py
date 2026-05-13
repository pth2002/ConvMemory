import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import CrossEncoder

from convmem_chain_benchmark import hit_at_k, mrr, recall_at_k
from convmem_locomo_benchmark import load_locomo_examples
from convmem_longmemeval import SentenceTransformerTextEncoder, resolve_local_model_path


def cosine_scores(query, matrix):
    return matrix @ query


def sample_id(question_id):
    return question_id.split("::", 1)[0]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["method"], []).append(row)
    out = []
    for method, items in grouped.items():
        out.append(
            {
                "method": method,
                "questions": len(items),
                "recall_at_10": float(np.mean([float(x["recall_at_10"]) for x in items])),
                "hit_at_10": float(np.mean([float(x["hit_at_10"]) for x in items])),
                "mrr": float(np.mean([float(x["mrr"]) for x in items])),
            }
        )
    return sorted(out, key=lambda x: x["recall_at_10"], reverse=True)


def choose_split(examples, split_name, dev_ratio, seed):
    if split_name == "all":
        return examples
    sample_ids = sorted({sample_id(x["question_id"]) for x in examples})
    rng = random.Random(seed)
    rng.shuffle(sample_ids)
    split = max(1, int(len(sample_ids) * dev_ratio))
    dev_samples = set(sample_ids[:split])
    if split_name == "dev":
        return [x for x in examples if sample_id(x["question_id"]) in dev_samples]
    return [x for x in examples if sample_id(x["question_id"]) not in dev_samples]


def add_metric_row(rows, method, item, ranked_ids):
    rows.append(
        {
            "method": method,
            "question_id": item["question_id"],
            "question_type": item["question_type"],
            "gold_chain_len": len(item["gold_memory_ids"]),
            "recall_at_10": recall_at_k(ranked_ids, item["gold_memory_ids"], 10),
            "hit_at_10": hit_at_k(ranked_ids, item["gold_memory_ids"], 10),
            "mrr": mrr(ranked_ids, item["gold_memory_ids"]),
            "top20_ids": "||".join(ranked_ids[:20]),
            "gold_ids": "||".join(item["gold_memory_ids"]),
        }
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/locomo10.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split", choices=["all", "dev", "test"], default="test")
    parser.add_argument("--dev-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--encoder-model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--encoder-batch-size", type=int, default=16)
    parser.add_argument("--embedding-cache", default=None)
    parser.add_argument("--cross-encoder-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--cross-batch-size", type=int, default=32)
    parser.add_argument("--raw-top-ns", default="50,100,200")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="results/locomo/crossencoder_mpnet_test")
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if args.device == "auto" and device != "cuda":
        device = "cpu"

    examples = load_locomo_examples(Path(__file__).parent / args.data, limit=args.limit)
    examples = choose_split(examples, args.split, args.dev_ratio, args.seed)

    encoder = SentenceTransformerTextEncoder(
        model_name=args.encoder_model,
        device=device,
        batch_size=args.encoder_batch_size,
        cache_path=args.embedding_cache,
    )
    cross_encoder = CrossEncoder(
        resolve_local_model_path(args.cross_encoder_model),
        device=device,
    )
    raw_top_ns = [int(x.strip()) for x in args.raw_top_ns.split(",") if x.strip()]

    rows = []
    start = time.perf_counter()
    for idx, item in enumerate(examples, start=1):
        memory_ids = [m["id"] for m in item["memories"]]
        memory_texts = [m["text"] for m in item["memories"]]
        memory_embeddings = encoder.transform(memory_texts)
        query_embedding = encoder.transform([item["query"]])[0]
        raw_scores = cosine_scores(query_embedding, memory_embeddings)
        raw_order = list(np.argsort(-raw_scores))
        raw_ranked = [memory_ids[i] for i in raw_order]
        add_metric_row(rows, "raw_turn", item, raw_ranked)

        for top_n in raw_top_ns:
            candidate_indices = raw_order[:top_n]
            pairs = [(item["query"], memory_texts[i]) for i in candidate_indices]
            ce_scores = cross_encoder.predict(
                pairs,
                batch_size=args.cross_batch_size,
                show_progress_bar=False,
            )
            ce_order = np.argsort(-np.asarray(ce_scores))
            ce_ranked_indices = [candidate_indices[i] for i in ce_order]
            ce_ranked = [memory_ids[i] for i in ce_ranked_indices]
            remaining = [memory_ids[i] for i in raw_order if i not in set(ce_ranked_indices)]
            add_metric_row(rows, f"cross_encoder_rerank_top{top_n}", item, ce_ranked + remaining)

        if idx % 100 == 0:
            elapsed = time.perf_counter() - start
            print(f"processed {idx}/{len(examples)} questions, elapsed={elapsed:.1f}s")

    out_dir = Path(__file__).parent / args.out
    summary = summarize(rows)
    write_csv(out_dir / "crossencoder_detailed_results.csv", rows)
    write_csv(out_dir / "crossencoder_summary_results.csv", summary)

    elapsed = time.perf_counter() - start
    print("\nLoCoMo cross-encoder baseline")
    print(f"split: {args.split}")
    print(f"questions: {len(examples)}")
    print(f"device: {device}")
    print(f"encoder: {args.encoder_model}")
    print(f"cross encoder: {args.cross_encoder_model}")
    print(f"latency: {1000 * elapsed / max(1, len(examples)):.2f}ms/query")
    print("\nmethod                         questions recall@10 hit@10 mrr")
    for row in summary:
        print(
            f"{row['method']:<30} "
            f"{row['questions']:<9} "
            f"{row['recall_at_10']:.3f}     "
            f"{row['hit_at_10']:.3f}  "
            f"{row['mrr']:.3f}"
        )
    print(f"\nSaved: {out_dir / 'crossencoder_summary_results.csv'}")


if __name__ == "__main__":
    main()

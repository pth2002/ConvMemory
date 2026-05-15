import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from convmemory import ConvMemory
from convmemory.metrics import hit_at_k, mrr, recall_at_k
from convmemory.scoring import cosine_scores
from experiments.support.convmem_chain_benchmark import prepare_encoded_example
from experiments.support.convmem_locomo_benchmark import load_locomo_examples
from experiments.support.convmem_longmemeval import SentenceTransformerTextEncoder
from experiments.support.locomo_crossencoder_baseline import choose_split


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ranked_ids(results):
    return [result.memory_id for result in results]


def zscore(values):
    values = np.asarray(values, dtype=np.float32)
    std = float(values.std())
    if std < 1e-8:
        return values - float(values.mean())
    return (values - float(values.mean())) / std


def confidence_features(results):
    if not results:
        return {"top1_score": 0.0, "top2_margin": 0.0, "score_std": 0.0}
    scores = np.asarray([result.score for result in results[:20]], dtype=np.float32)
    norm = zscore(scores)
    return {
        "top1_score": float(results[0].score),
        "top1_zscore": float(norm[0]) if len(norm) else 0.0,
        "top2_margin": float(scores[0] - scores[1]) if len(scores) > 1 else 0.0,
        "score_std": float(scores.std()) if len(scores) else 0.0,
    }


def query_bucket(item):
    qtype = item.get("question_type", "unknown")
    gold_len = len(item["gold_memory_ids"])
    if gold_len <= 1:
        chain = "single_evidence"
    elif gold_len <= 3:
        chain = "short_chain"
    else:
        chain = "long_chain"
    return f"{qtype}::{chain}"


def add_case(rows, seed, item, raw_ranked, conv_results):
    conv_ranked = ranked_ids(conv_results)
    raw_hit = hit_at_k(raw_ranked, item["gold_memory_ids"], 10)
    conv_hit = hit_at_k(conv_ranked, item["gold_memory_ids"], 10)
    if conv_hit and not raw_hit:
        outcome = "convmemory_win"
    elif raw_hit and not conv_hit:
        outcome = "convmemory_loss"
    elif conv_hit and raw_hit:
        outcome = "both_hit"
    else:
        outcome = "both_miss"
    conf = confidence_features(conv_results)
    rows.append(
        {
            "seed": seed,
            "question_id": item["question_id"],
            "question_type": item["question_type"],
            "bucket": query_bucket(item),
            "outcome": outcome,
            "gold_chain_len": len(item["gold_memory_ids"]),
            "raw_recall_at_10": recall_at_k(raw_ranked, item["gold_memory_ids"], 10),
            "conv_recall_at_10": recall_at_k(conv_ranked, item["gold_memory_ids"], 10),
            "raw_hit_at_10": raw_hit,
            "conv_hit_at_10": conv_hit,
            "raw_mrr": mrr(raw_ranked, item["gold_memory_ids"]),
            "conv_mrr": mrr(conv_ranked, item["gold_memory_ids"]),
            "delta_recall_at_10": recall_at_k(conv_ranked, item["gold_memory_ids"], 10)
            - recall_at_k(raw_ranked, item["gold_memory_ids"], 10),
            "delta_mrr": mrr(conv_ranked, item["gold_memory_ids"]) - mrr(raw_ranked, item["gold_memory_ids"]),
            "top1_score": conf["top1_score"],
            "top1_zscore": conf["top1_zscore"],
            "top2_margin": conf["top2_margin"],
            "score_std": conf["score_std"],
            "query": item["query"],
            "gold_ids": "||".join(item["gold_memory_ids"]),
            "raw_top10": "||".join(raw_ranked[:10]),
            "conv_top10": "||".join(conv_ranked[:10]),
        }
    )


def summarize_cases(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["bucket"], row["outcome"])].append(row)
    out = []
    for (bucket, outcome), items in sorted(groups.items()):
        out.append(
            {
                "bucket": bucket,
                "outcome": outcome,
                "questions": len(items),
                "mean_delta_recall_at_10": float(np.mean([float(x["delta_recall_at_10"]) for x in items])),
                "mean_delta_mrr": float(np.mean([float(x["delta_mrr"]) for x in items])),
                "mean_top1_zscore": float(np.mean([float(x["top1_zscore"]) for x in items])),
                "mean_top2_margin": float(np.mean([float(x["top2_margin"]) for x in items])),
            }
        )
    return out


def calibration_bins(rows, score_field, target_field, bins):
    values = np.asarray([float(row[score_field]) for row in rows], dtype=np.float32)
    if len(values) == 0:
        return []
    cuts = np.quantile(values, np.linspace(0.0, 1.0, bins + 1))
    cuts[0] -= 1e-6
    cuts[-1] += 1e-6
    out = []
    for idx in range(bins):
        lo, hi = float(cuts[idx]), float(cuts[idx + 1])
        selected = [
            row for row in rows
            if lo <= float(row[score_field]) < hi
        ]
        if not selected:
            continue
        out.append(
            {
                "score_field": score_field,
                "target": target_field,
                "bin": idx,
                "low": lo,
                "high": hi,
                "questions": len(selected),
                "mean_score": float(np.mean([float(row[score_field]) for row in selected])),
                "empirical_success": float(np.mean([float(row[target_field]) for row in selected])),
                "mean_recall_at_10": float(np.mean([float(row["conv_recall_at_10"]) for row in selected])),
                "mean_mrr": float(np.mean([float(row["conv_mrr"]) for row in selected])),
            }
        )
    return out


def write_report(path, summary, calibration):
    total = sum(row["questions"] for row in summary)
    lines = [
        "# v0.42 Error Analysis And Calibration",
        "",
        f"Total bucketed cases: {total}",
        "",
        "## Outcome Buckets",
        "",
        "| Bucket | Outcome | Questions | Delta Recall@10 | Delta MRR | Top1 z-score | Top2 margin |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| `{row['bucket']}` | `{row['outcome']}` | {row['questions']} | "
            f"{row['mean_delta_recall_at_10']:+.4f} | {row['mean_delta_mrr']:+.4f} | "
            f"{row['mean_top1_zscore']:.4f} | {row['mean_top2_margin']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Calibration Bins",
            "",
            "| Score | Target | Bin | Questions | Score range | Empirical success | Recall@10 | MRR |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in calibration:
        lines.append(
            f"| `{row['score_field']}` | `{row['target']}` | {row['bin']} | {row['questions']} | "
            f"[{row['low']:.4f}, {row['high']:.4f}] | {row['empirical_success']:.4f} | "
            f"{row['mean_recall_at_10']:.4f} | {row['mean_mrr']:.4f} |"
        )
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/locomo10.json")
    parser.add_argument("--checkpoint", default="checkpoints/convmemory-locomo-mpnet")
    parser.add_argument("--encoder-model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--embedding-cache", default=None)
    parser.add_argument("--embedding-cache-key", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 11, 23, 31, 47])
    parser.add_argument("--split", choices=["dev", "test", "all"], default="test")
    parser.add_argument("--dev-ratio", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--candidate-top-n", type=int, default=500)
    parser.add_argument("--encoder-batch-size", type=int, default=128)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--out", default="results/v042/error_calibration")
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    examples = load_locomo_examples(ROOT / args.data)
    convmemory = ConvMemory.from_pretrained(ROOT / args.checkpoint, device=device, embedding_model=False)
    encoder = SentenceTransformerTextEncoder(
        model_name=args.encoder_model,
        device=device,
        batch_size=args.encoder_batch_size,
        cache_path=args.embedding_cache,
        cache_key=args.embedding_cache_key or args.encoder_model,
    )

    rows = []
    for seed in args.seeds:
        split_examples = choose_split(examples, args.split, args.dev_ratio, seed)
        if args.limit:
            split_examples = split_examples[: args.limit]
        for idx, example in enumerate(split_examples, start=1):
            item = prepare_encoded_example(example, encoder, convmemory.config.window_size, convmemory.config.stride)
            raw_scores = cosine_scores(item["query_embedding"], item["memory_embeddings"])
            raw_ranked = [item["memory_ids"][int(i)] for i in np.argsort(-raw_scores)]
            conv_results = convmemory.reranker.rerank_item(
                item,
                candidate_top_n=args.candidate_top_n,
                window_mode="candidate_local",
            )
            add_case(rows, seed, item, raw_ranked, conv_results)
            if idx % 100 == 0 or idx == len(split_examples):
                print(f"seed {seed}: {idx}/{len(split_examples)}", flush=True)

    summary = summarize_cases(rows)
    calibration = []
    calibration.extend(calibration_bins(rows, "top1_zscore", "conv_hit_at_10", args.bins))
    calibration.extend(calibration_bins(rows, "top2_margin", "conv_hit_at_10", args.bins))

    wins = sorted([row for row in rows if row["outcome"] == "convmemory_win"], key=lambda x: -float(x["delta_mrr"]))[:50]
    losses = sorted([row for row in rows if row["outcome"] == "convmemory_loss"], key=lambda x: float(x["delta_mrr"]))[:50]

    out_dir = ROOT / args.out
    write_csv(out_dir / "cases.csv", rows)
    write_csv(out_dir / "bucket_summary.csv", summary)
    write_csv(out_dir / "calibration_bins.csv", calibration)
    write_csv(out_dir / "top_wins.csv", wins)
    write_csv(out_dir / "top_losses.csv", losses)
    write_report(out_dir / "REPORT.md", summary, calibration)
    print(f"Saved v0.42 results to {out_dir}", flush=True)


if __name__ == "__main__":
    main()

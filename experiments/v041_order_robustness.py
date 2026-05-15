import argparse
import csv
import random
import sys
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
from experiments.v040_baselines_ablation_stats import feature_masked_rank


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def perturb_memories(memories, mode, rng, block_size):
    memories = list(memories)
    n_items = len(memories)
    if mode == "original":
        return memories
    if mode == "reverse":
        return list(reversed(memories))
    if mode.startswith("shuffle_"):
        rate = float(mode.split("_", 1)[1]) / 100.0
        count = int(round(n_items * rate))
        picked = sorted(rng.sample(range(n_items), count)) if count > 1 else []
        values = [memories[idx] for idx in picked]
        rng.shuffle(values)
        out = list(memories)
        for idx, value in zip(picked, values):
            out[idx] = value
        return out
    if mode.startswith("block_shuffle"):
        blocks = [memories[start : start + block_size] for start in range(0, n_items, block_size)]
        rng.shuffle(blocks)
        return [item for block in blocks for item in block]
    raise ValueError(f"Unknown perturbation mode: {mode}")


def add_row(rows, seed, perturbation, method, item, ranked):
    rows.append(
        {
            "seed": seed,
            "perturbation": perturbation,
            "method": method,
            "question_id": item["question_id"],
            "question_type": item["question_type"],
            "memory_count": len(item["memory_ids"]),
            "gold_chain_len": len(item["gold_memory_ids"]),
            "recall_at_5": recall_at_k(ranked, item["gold_memory_ids"], 5),
            "hit_at_5": hit_at_k(ranked, item["gold_memory_ids"], 5),
            "recall_at_10": recall_at_k(ranked, item["gold_memory_ids"], 10),
            "hit_at_10": hit_at_k(ranked, item["gold_memory_ids"], 10),
            "recall_at_20": recall_at_k(ranked, item["gold_memory_ids"], 20),
            "hit_at_20": hit_at_k(ranked, item["gold_memory_ids"], 20),
            "mrr": mrr(ranked, item["gold_memory_ids"]),
        }
    )


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["perturbation"], row["method"], row["seed"]), []).append(row)
    per_seed = []
    for (perturbation, method, seed), items in sorted(groups.items()):
        out = {"perturbation": perturbation, "method": method, "seed": seed, "questions": len(items)}
        for metric in ["recall_at_5", "hit_at_5", "recall_at_10", "hit_at_10", "recall_at_20", "hit_at_20", "mrr"]:
            out[metric] = float(np.mean([float(item[metric]) for item in items]))
        per_seed.append(out)

    by_group = {}
    for row in per_seed:
        by_group.setdefault((row["perturbation"], row["method"]), []).append(row)
    summary = []
    for (perturbation, method), items in sorted(by_group.items()):
        out = {"perturbation": perturbation, "method": method, "seeds": len(items)}
        for metric in ["recall_at_5", "hit_at_5", "recall_at_10", "hit_at_10", "recall_at_20", "hit_at_20", "mrr"]:
            values = np.asarray([float(item[metric]) for item in items], dtype=np.float32)
            out[f"{metric}_mean"] = float(values.mean())
            out[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary.append(out)
    return per_seed, summary


def write_report(path, summary):
    original = {
        row["method"]: row
        for row in summary
        if row["perturbation"] == "original"
    }
    lines = [
        "# v0.41 Order Robustness",
        "",
        "This experiment perturbs memory order to quantify how much ConvMemory depends on chronological structure.",
        "",
        "| Perturbation | Method | Recall@10 | Delta vs original | MRR | Delta MRR |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in sorted(summary, key=lambda x: (x["method"], x["perturbation"])):
        base = original.get(row["method"], row)
        lines.append(
            f"| `{row['perturbation']}` | `{row['method']}` | "
            f"{row['recall_at_10_mean']:.4f} | "
            f"{row['recall_at_10_mean'] - base['recall_at_10_mean']:+.4f} | "
            f"{row['mrr_mean']:.4f} | {row['mrr_mean'] - base['mrr_mean']:+.4f} |"
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
    parser.add_argument("--perturbations", default="original,shuffle_10,shuffle_25,shuffle_50,shuffle_100,block_shuffle,reverse")
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--encoder-batch-size", type=int, default=128)
    parser.add_argument("--out", default="results/v041/order_robustness")
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
    perturbations = [x.strip() for x in args.perturbations.split(",") if x.strip()]

    rows = []
    for seed in args.seeds:
        split_examples = choose_split(examples, args.split, args.dev_ratio, seed)
        if args.limit:
            split_examples = split_examples[: args.limit]
        for idx, example in enumerate(split_examples, start=1):
            for perturbation in perturbations:
                rng = random.Random(f"{seed}:{example['question_id']}:{perturbation}")
                perturbed = {
                    **example,
                    "memories": perturb_memories(example["memories"], perturbation, rng, args.block_size),
                }
                item = prepare_encoded_example(perturbed, encoder, convmemory.config.window_size, convmemory.config.stride)
                raw_scores = cosine_scores(item["query_embedding"], item["memory_embeddings"])
                raw_ranked = [item["memory_ids"][int(i)] for i in np.argsort(-raw_scores)]
                conv_ranked = feature_masked_rank(
                    convmemory,
                    item,
                    "full",
                    candidate_top_n=args.candidate_top_n,
                    window_mode="candidate_local",
                )
                add_row(rows, seed, perturbation, "raw_dense", item, raw_ranked)
                add_row(rows, seed, perturbation, "convmemory", item, conv_ranked)
            if idx % 100 == 0 or idx == len(split_examples):
                print(f"seed {seed}: {idx}/{len(split_examples)}", flush=True)

    per_seed, summary = summarize(rows)
    out_dir = ROOT / args.out
    write_csv(out_dir / "detailed.csv", rows)
    write_csv(out_dir / "summary_by_seed.csv", per_seed)
    write_csv(out_dir / "summary.csv", summary)
    write_report(out_dir / "REPORT.md", summary)
    print(f"Saved v0.41 results to {out_dir}", flush=True)


if __name__ == "__main__":
    main()

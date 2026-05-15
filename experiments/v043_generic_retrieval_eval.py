"""Generic retrieval-stage evaluator for non-LoCoMo memory datasets.

Input JSONL schema, one question per line:

{
  "question_id": "example-1",
  "question_type": "optional",
  "query": "What did the user reschedule?",
  "memories": [{"id": "m1", "text": "..."}, ...],
  "gold_memory_ids": ["m7", "m8"]
}

This makes it possible to evaluate ConvMemory on converted datasets such as
MSC, Multi-Session Chat, agent scratchpads, HotpotQA, or MuSiQue without adding
dataset-specific logic to the library.
"""

import argparse
import csv
import json
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
from experiments.support.convmem_longmemeval import SentenceTransformerTextEncoder
from experiments.v040_baselines_ablation_stats import baseline_rankings, feature_masked_rank


def load_jsonl(path, limit=0):
    examples = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            memories = item.get("memories") or []
            gold = {str(x) for x in item.get("gold_memory_ids", [])}
            memory_ids = {str(memory.get("id")) for memory in memories}
            gold = [memory_id for memory_id in gold if memory_id in memory_ids]
            if not memories or not gold:
                continue
            examples.append(
                {
                    "question_id": str(item.get("question_id", len(examples))),
                    "question_type": str(item.get("question_type", "unknown")),
                    "query": str(item["query"]),
                    "memories": [
                        {"id": str(memory.get("id", idx)), "text": str(memory.get("text", ""))}
                        for idx, memory in enumerate(memories)
                    ],
                    "gold_memory_ids": gold,
                }
            )
            if limit and len(examples) >= limit:
                break
    return examples


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def add_row(rows, dataset, method, item, ranked):
    rows.append(
        {
            "dataset": dataset,
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
        groups.setdefault((row["dataset"], row["method"]), []).append(row)
    out = []
    for (dataset, method), items in sorted(groups.items()):
        summary = {"dataset": dataset, "method": method, "questions": len(items)}
        for metric in ["recall_at_5", "hit_at_5", "recall_at_10", "hit_at_10", "recall_at_20", "hit_at_20", "mrr"]:
            summary[metric] = float(np.mean([float(item[metric]) for item in items]))
        out.append(summary)
    return sorted(out, key=lambda x: (x["dataset"], -x["recall_at_10"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/convmemory-locomo-mpnet")
    parser.add_argument("--encoder-model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--embedding-cache", default=None)
    parser.add_argument("--embedding-cache-key", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--candidate-top-n", type=int, default=500)
    parser.add_argument("--encoder-batch-size", type=int, default=128)
    parser.add_argument("--out", default="results/v043/generic_retrieval_eval")
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    examples = load_jsonl(args.jsonl, limit=args.limit)
    convmemory = ConvMemory.from_pretrained(ROOT / args.checkpoint, device=device, embedding_model=False)
    encoder = SentenceTransformerTextEncoder(
        model_name=args.encoder_model,
        device=device,
        batch_size=args.encoder_batch_size,
        cache_path=args.embedding_cache,
        cache_key=args.embedding_cache_key or args.encoder_model,
    )

    rows = []
    for idx, example in enumerate(examples, start=1):
        item = prepare_encoded_example(example, encoder, convmemory.config.window_size, convmemory.config.stride)
        for method, ranked in baseline_rankings(item, convmemory.config.window_size, [0.03]).items():
            add_row(rows, args.dataset_name, method, item, ranked)
        conv_ranked = feature_masked_rank(
            convmemory,
            item,
            "full",
            candidate_top_n=args.candidate_top_n,
            window_mode="candidate_local",
        )
        add_row(rows, args.dataset_name, "convmemory", item, conv_ranked)
        raw_scores = cosine_scores(item["query_embedding"], item["memory_embeddings"])
        raw_ranked = [item["memory_ids"][int(i)] for i in np.argsort(-raw_scores)]
        add_row(rows, args.dataset_name, "raw_dense_check", item, raw_ranked)
        if idx % 100 == 0 or idx == len(examples):
            print(f"{args.dataset_name}: {idx}/{len(examples)}", flush=True)

    out_dir = ROOT / args.out / args.dataset_name
    write_csv(out_dir / "detailed.csv", rows)
    write_csv(out_dir / "summary.csv", summarize(rows))
    print(f"Saved generic evaluation to {out_dir}", flush=True)


if __name__ == "__main__":
    main()

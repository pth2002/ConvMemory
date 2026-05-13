import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch

from .convmem_chain_benchmark import (
    evaluate_item,
    load_chain_examples,
    prepare_encoded_example,
    summarize,
    write_csv,
)
from .convmem_longmemeval import ConvMemoryEncoder, SentenceTransformerTextEncoder, TfidfTextEncoder, train_conv

import json


def session_number(key):
    return int(key.split("_")[1])


def load_locomo_examples(path, limit=None):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    examples = []
    for sample in data:
        conversation = sample["conversation"]
        session_keys = [
            key
            for key in conversation
            if key.startswith("session_")
            and not key.endswith("date_time")
            and isinstance(conversation[key], list)
        ]
        session_keys.sort(key=session_number)

        memories = []
        dia_ids = set()
        for session_key in session_keys:
            for turn in conversation[session_key]:
                dia_id = turn["dia_id"]
                dia_ids.add(dia_id)
                speaker = turn.get("speaker", "speaker")
                text = turn.get("text", "")
                memories.append(
                    {
                        "id": dia_id,
                        "session_id": session_key,
                        "role": "speaker",
                        "text": f"{speaker}: {text}",
                    }
                )

        for qa_idx, qa in enumerate(sample["qa"]):
            evidence = [x for x in qa.get("evidence", []) if x in dia_ids]
            if not evidence:
                continue
            examples.append(
                {
                    "question_id": f"{sample['sample_id']}::qa{qa_idx}",
                    "question_type": f"category_{qa.get('category', 'unknown')}",
                    "query": qa["question"],
                    "answer": qa.get("answer", ""),
                    "memories": memories,
                    "gold_memory_ids": evidence,
                }
            )
            if limit and len(examples) >= limit:
                return examples
    return examples


def collect_texts(examples):
    texts = []
    for item in examples:
        texts.append(item["query"])
        texts.extend(m["text"] for m in item["memories"])
    return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/locomo10.json")
    parser.add_argument("--train-source", choices=["locomo", "longmemeval"], default="locomo")
    parser.add_argument("--train-data", default="data/longmemeval_s_cleaned.json")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--encoder", choices=["tfidf", "sbert"], default="sbert")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--embedding-cache", default=None)
    parser.add_argument("--window-size", type=int, default=9)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--pooling", choices=["mean", "query_attention"], default="query_attention")
    parser.add_argument("--multi-scale-kernels", default="3,5,9")
    parser.add_argument("--projection", choices=["none", "linear", "mlp"], default="linear")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--raw-weights", default="0.2,0.3,0.4,0.5,0.6")
    parser.add_argument("--rerank-top-ns", default="50,100")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="results/locomo/multiscale")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if args.device == "auto" and device != "cuda":
        device = "cpu"

    locomo_examples = load_locomo_examples(Path(__file__).parent / args.data, limit=args.limit)
    if args.train_source == "locomo":
        random.shuffle(locomo_examples)
        split = int(len(locomo_examples) * args.train_ratio)
        train_examples = locomo_examples[:split]
        test_examples = locomo_examples[split:]
    else:
        train_examples = load_chain_examples(
            Path(__file__).parent / args.train_data,
            limit=args.train_limit,
            seed=args.seed,
        )
        test_examples = locomo_examples

    texts = collect_texts(train_examples)

    if args.encoder == "tfidf":
        encoder = TfidfTextEncoder()
    else:
        encoder = SentenceTransformerTextEncoder(
            model_name=args.embedding_model,
            device=device,
            batch_size=args.embedding_batch_size,
            cache_path=args.embedding_cache,
        )
    encoder.fit(texts)

    encoded_train = [
        prepare_encoded_example(x, encoder, args.window_size, args.stride)
        for x in train_examples
    ]
    encoded_train = [x for x in encoded_train if x is not None]

    dim = encoded_train[0]["memory_embeddings"].shape[1]
    kernels = [int(x.strip()) for x in args.multi_scale_kernels.split(",") if x.strip()]
    model = ConvMemoryEncoder(
        dim,
        pooling=args.pooling,
        multi_scale_kernels=kernels,
        projection=args.projection,
    ).to(device)

    start = time.perf_counter()
    train_conv(model, encoded_train, epochs=args.epochs, device=device)
    train_time = time.perf_counter() - start

    raw_weights = [float(x.strip()) for x in args.raw_weights.split(",") if x.strip()]
    rerank_top_ns = [int(x.strip()) for x in args.rerank_top_ns.split(",") if x.strip()]

    rows = []
    test_count = 0
    memory_counts = []
    gold_counts = []
    start = time.perf_counter()
    for example in test_examples:
        item = prepare_encoded_example(example, encoder, args.window_size, args.stride)
        if item is None:
            continue
        test_count += 1
        memory_counts.append(len(item["memories"]))
        gold_counts.append(len(item["gold_memory_ids"]))
        rows.extend(
            evaluate_item(
                item,
                model,
                device=device,
                raw_weights=raw_weights,
                rerank_top_ns=rerank_top_ns,
            )
        )
    eval_time = time.perf_counter() - start

    out_dir = Path(__file__).parent / args.out
    summary = summarize(rows)
    write_csv(out_dir / "locomo_detailed_results.csv", rows)
    write_csv(out_dir / "locomo_summary_results.csv", summary)

    avg_memories = np.mean(memory_counts)
    avg_gold = np.mean(gold_counts)
    print("\nConvMem LoCoMo benchmark")
    print(f"train source: {args.train_source}")
    print(f"examples train/test: {len(encoded_train)}/{test_count}")
    print(f"device: {device}")
    print(f"kernels: {kernels}")
    print(f"projection: {args.projection}")
    print(f"avg turn memories: {avg_memories:.1f}")
    print(f"avg evidence turns: {avg_gold:.1f}")
    print(f"train time: {train_time:.1f}s")
    print(f"eval latency: {1000 * eval_time / max(1, test_count):.2f}ms/query")
    print("\nmethod                              questions  recall@10 hit@10 mrr")
    for row in summary:
        print(
            f"{row['method']:<35} "
            f"{row['questions']:<9} "
            f"{row['recall_at_10']:.3f}     "
            f"{row['hit_at_10']:.3f}  "
            f"{row['mrr']:.3f}"
        )
    print(f"\nSaved: {out_dir / 'locomo_summary_results.csv'}")


if __name__ == "__main__":
    main()

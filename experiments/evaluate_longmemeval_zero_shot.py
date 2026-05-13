import argparse
import csv
import random
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
from convmemory.scoring import lexical_signature
from convmem_longmemeval import (
    SentenceTransformerTextEncoder,
    load_longmemeval,
    resolve_local_model_path,
)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cosine_scores(query_embedding, memory_embeddings):
    query_embedding = np.asarray(query_embedding, dtype=np.float32)
    memory_embeddings = np.asarray(memory_embeddings, dtype=np.float32)
    return memory_embeddings @ query_embedding


def ranked_by_scores(memory_ids, scores):
    return [memory_ids[int(i)] for i in np.argsort(-np.asarray(scores, dtype=np.float32))]


def cross_encoder_rank(cross_encoder, query, memories, indices, batch_size):
    pairs = [(query, memories[int(i)]["text"]) for i in indices]
    scores = cross_encoder.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=False,
    )
    order = [int(indices[i]) for i in np.argsort(-np.asarray(scores, dtype=np.float32))]
    return order


def add_metrics(rows, method, item, ranked_ids, elapsed_s, candidate_count):
    rows.append(
        {
            "method": method,
            "question_id": item["question_id"],
            "question_type": item.get("question_type", "unknown"),
            "candidate_count": candidate_count,
            "latency_ms": elapsed_s * 1000.0,
            "recall_at_5": recall_at_k(ranked_ids, item["gold_memory_ids"], 5),
            "hit_at_5": hit_at_k(ranked_ids, item["gold_memory_ids"], 5),
            "recall_at_10": recall_at_k(ranked_ids, item["gold_memory_ids"], 10),
            "hit_at_10": hit_at_k(ranked_ids, item["gold_memory_ids"], 10),
            "mrr": mrr(ranked_ids, item["gold_memory_ids"]),
        }
    )


def summarize(rows, group_key=None):
    grouped = {}
    for row in rows:
        key = row["method"] if group_key is None else (row[group_key], row["method"])
        grouped.setdefault(key, []).append(row)

    summary = []
    for key, items in grouped.items():
        if group_key is None:
            prefix = {"method": key}
        else:
            prefix = {group_key: key[0], "method": key[1]}
        summary.append(
            {
                **prefix,
                "questions": len(items),
                "avg_candidates": float(np.mean([float(x["candidate_count"]) for x in items])),
                "recall_at_5": float(np.mean([float(x["recall_at_5"]) for x in items])),
                "hit_at_5": float(np.mean([float(x["hit_at_5"]) for x in items])),
                "recall_at_10": float(np.mean([float(x["recall_at_10"]) for x in items])),
                "hit_at_10": float(np.mean([float(x["hit_at_10"]) for x in items])),
                "mrr": float(np.mean([float(x["mrr"]) for x in items])),
                "ms_per_query": float(np.mean([float(x["latency_ms"]) for x in items])),
            }
        )
    return sorted(summary, key=lambda x: (x.get(group_key, ""), -x["recall_at_10"]))


def print_summary(summary):
    print("\nmethod                         questions  recall@5  hit@5  recall@10  hit@10  mrr    ms/query")
    for row in summary:
        print(
            f"{row['method']:<30} "
            f"{row['questions']:<9} "
            f"{row['recall_at_5']:.3f}     "
            f"{row['hit_at_5']:.3f}  "
            f"{row['recall_at_10']:.3f}      "
            f"{row['hit_at_10']:.3f}   "
            f"{row['mrr']:.3f}  "
            f"{row['ms_per_query']:.1f}"
        )


def add_distractor_sessions(examples, target_sessions, mode="append", seed=23):
    if target_sessions <= 0:
        return examples

    all_memories = []
    for item in examples:
        source_id = str(item["question_id"])
        for memory in item["memories"]:
            all_memories.append(
                {
                    "id": f"distractor::{source_id}::{memory['id']}",
                    "text": memory["text"],
                    "source_question_id": source_id,
                }
            )

    expanded = []
    for item in examples:
        memories = list(item["memories"])
        if len(memories) >= target_sessions:
            expanded.append(item)
            continue

        rng = random.Random(f"{seed}:{item['question_id']}")
        existing_texts = {memory["text"] for memory in memories}
        candidates = [
            memory
            for memory in all_memories
            if memory["source_question_id"] != str(item["question_id"])
            and memory["text"] not in existing_texts
        ]
        rng.shuffle(candidates)
        needed = target_sessions - len(memories)
        distractors = [
            {"id": memory["id"], "text": memory["text"]}
            for memory in candidates[:needed]
        ]

        if mode == "interleave" and distractors:
            mixed = memories + distractors
            rng.shuffle(mixed)
            memories = mixed
        else:
            memories = memories + distractors

        expanded.append({**item, "memories": memories})
    return expanded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/longmemeval_s_cleaned.json")
    parser.add_argument("--checkpoint", default="checkpoints/convmemory-locomo-mpnet")
    parser.add_argument("--encoder-model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--embedding-cache", default=None)
    parser.add_argument("--embedding-cache-key", default=None)
    parser.add_argument("--candidate-top-n", type=int, default=500)
    parser.add_argument("--window-mode", choices=["full", "candidate_local"], default="full")
    parser.add_argument("--distractor-sessions", type=int, default=0)
    parser.add_argument("--distractor-mode", choices=["append", "interleave"], default="append")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--stratified-per-type", type=int, default=None)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--no-preencode", action="store_true")
    parser.add_argument("--preencode-chunk-size", type=int, default=512)
    parser.add_argument("--precache-lexical", action="store_true")
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--eval-cross-encoder", action="store_true")
    parser.add_argument("--cross-encoder-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--cross-batch-size", type=int, default=64)
    parser.add_argument("--out", default="results/longmemeval_zero_shot")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    data_path = ROOT / args.data
    load_limit = None if (args.sample_size or args.stratified_per_type) else args.max_examples
    examples = load_longmemeval(data_path, limit=load_limit, skip_abstention=True)
    rng = random.Random(args.seed)
    if args.stratified_per_type:
        by_type = {}
        for item in examples:
            by_type.setdefault(item.get("question_type", "unknown"), []).append(item)
        sampled = []
        for question_type in sorted(by_type):
            items = list(by_type[question_type])
            rng.shuffle(items)
            sampled.extend(items[: args.stratified_per_type])
        rng.shuffle(sampled)
        examples = sampled
    elif args.sample_size:
        examples = list(examples)
        rng.shuffle(examples)
        examples = examples[: args.sample_size]
    if args.distractor_sessions:
        examples = add_distractor_sessions(
            examples,
            target_sessions=args.distractor_sessions,
            mode=args.distractor_mode,
            seed=args.seed,
        )

    encoder = SentenceTransformerTextEncoder(
        model_name=args.encoder_model,
        device=device,
        batch_size=args.embedding_batch_size,
        cache_path=args.embedding_cache,
        cache_key=args.embedding_cache_key or args.encoder_model,
    )
    model = ConvMemory.from_pretrained(
        ROOT / args.checkpoint,
        device=device,
        embedding_model=False,
    )
    cross_encoder = None
    if args.eval_cross_encoder:
        cross_encoder = CrossEncoder(resolve_local_model_path(args.cross_encoder_model), device=device)

    if not args.no_preencode:
        all_texts = []
        for item in examples:
            all_texts.append(item["query"])
            all_texts.extend(memory["text"] for memory in item["memories"])
        unique_texts = list(dict.fromkeys(all_texts))
        start = time.perf_counter()
        chunk_size = max(1, args.preencode_chunk_size)
        for offset in range(0, len(unique_texts), chunk_size):
            encoder.transform(unique_texts[offset : offset + chunk_size])
            print(
                f"preencoded {min(offset + chunk_size, len(unique_texts))}/{len(unique_texts)} unique texts",
                flush=True,
            )
        print(
            f"preencoded {len(unique_texts)} unique texts in {time.perf_counter() - start:.1f}s",
            flush=True,
        )
    if args.precache_lexical:
        unique_memory_texts = []
        seen_texts = set()
        for item in examples:
            for memory in item["memories"]:
                text = memory["text"]
                if text not in seen_texts:
                    seen_texts.add(text)
                    unique_memory_texts.append(text)
        start = time.perf_counter()
        for text in unique_memory_texts:
            lexical_signature(text)
        print(
            f"precached lexical signatures for {len(unique_memory_texts)} unique memory texts "
            f"in {time.perf_counter() - start:.1f}s",
            flush=True,
        )

    rows = []
    total_start = time.perf_counter()
    for index, item in enumerate(examples, start=1):
        memory_ids = [memory["id"] for memory in item["memories"]]
        memory_texts = [memory["text"] for memory in item["memories"]]
        memory_embeddings = encoder.transform(memory_texts)
        query_embedding = encoder.transform([item["query"]])[0]

        raw_scores = cosine_scores(query_embedding, memory_embeddings)
        raw_order_indices = np.argsort(-raw_scores)
        raw_ranked = [memory_ids[int(i)] for i in raw_order_indices]

        start = time.perf_counter()
        _ = ranked_by_scores(memory_ids, raw_scores)
        add_metrics(
            rows,
            "raw_mpnet",
            item,
            raw_ranked,
            time.perf_counter() - start,
            len(memory_ids),
        )

        candidate_indices = raw_order_indices[: min(args.candidate_top_n, len(raw_order_indices))]
        start = time.perf_counter()
        conv_ranked = model.rerank_embeddings(
            query_embedding=query_embedding,
            memory_embeddings=memory_embeddings,
            memory_ids=memory_ids,
            memory_texts=memory_texts,
            query=item["query"],
            candidate_indices=candidate_indices,
            window_mode=args.window_mode,
        )
        conv_method = f"convmemory_zero_shot_top{args.candidate_top_n}"
        if args.window_mode != "full":
            conv_method = f"{conv_method}_{args.window_mode}"
        add_metrics(
            rows,
            conv_method,
            item,
            [result.memory_id for result in conv_ranked],
            time.perf_counter() - start,
            len(candidate_indices),
        )

        if cross_encoder is not None:
            start = time.perf_counter()
            ce_order = cross_encoder_rank(
                cross_encoder,
                item["query"],
                item["memories"],
                candidate_indices,
                args.cross_batch_size,
            )
            ce_ranked = [memory_ids[i] for i in ce_order]
            ce_ranked_set = set(ce_ranked)
            ce_ranked.extend([memory_id for memory_id in raw_ranked if memory_id not in ce_ranked_set])
            add_metrics(
                rows,
                f"cross_encoder_top{args.candidate_top_n}",
                item,
                ce_ranked,
                time.perf_counter() - start,
                len(candidate_indices),
            )

        if index % 25 == 0 or index == len(examples):
            elapsed = time.perf_counter() - total_start
            print(f"processed {index}/{len(examples)} examples in {elapsed:.1f}s", flush=True)

    out_dir = ROOT / args.out
    detailed = out_dir / "longmemeval_zero_shot_detailed.csv"
    summary = out_dir / "longmemeval_zero_shot_summary.csv"
    by_type = out_dir / "longmemeval_zero_shot_by_type.csv"
    write_csv(detailed, rows)
    summary_rows = summarize(rows)
    write_csv(summary, summary_rows)
    write_csv(by_type, summarize(rows, group_key="question_type"))

    print("\nLongMemEval zero-shot ConvMemory evaluation")
    print(f"data: {data_path}")
    print(f"checkpoint: {ROOT / args.checkpoint}")
    print(f"examples: {len(examples)}")
    if args.stratified_per_type:
        print(f"stratified per type: {args.stratified_per_type}")
    elif args.sample_size:
        print(f"sample size: {args.sample_size}")
    print(f"device: {device}")
    print(f"encoder: {args.encoder_model}")
    print(f"candidate top-n: {args.candidate_top_n}")
    print(f"window mode: {args.window_mode}")
    if args.distractor_sessions:
        print(f"distractor sessions: {args.distractor_sessions}")
        print(f"distractor mode: {args.distractor_mode}")
    if args.precache_lexical:
        print("lexical cache: precached memory signatures")
    print_summary(summary_rows)
    print(f"\nSaved: {summary}")
    print(f"Saved: {by_type}")
    print(f"Saved: {detailed}")


if __name__ == "__main__":
    main()

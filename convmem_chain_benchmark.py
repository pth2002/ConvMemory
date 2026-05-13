import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from convmem_longmemeval import (
    ConvMemoryEncoder,
    SentenceTransformerTextEncoder,
    TfidfTextEncoder,
    add_distractors,
    convmem_rerank,
    expand_window_ranking,
    make_windows,
    train_conv,
)


def l2_normalize(x, eps=1e-8):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def cosine_scores(query, matrix):
    return matrix @ query


def recall_at_k(ranked_ids, gold_ids, k):
    return len(set(ranked_ids[:k]) & set(gold_ids)) / max(1, len(gold_ids))


def hit_at_k(ranked_ids, gold_ids, k):
    return float(bool(set(ranked_ids[:k]) & set(gold_ids)))


def mrr(ranked_ids, gold_ids):
    gold = set(gold_ids)
    for rank, item_id in enumerate(ranked_ids, start=1):
        if item_id in gold:
            return 1.0 / rank
    return 0.0


def session_turn_text(turn):
    role = turn.get("role", "speaker")
    content = turn.get("content", "")
    return f"{role}: {content}"


def load_chain_examples(path, limit=None, distractor_sessions=0, seed=7):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    examples = []
    for item in raw:
        gold_sessions = set(item.get("answer_session_ids") or [])
        if not gold_sessions:
            continue

        memories = []
        gold_turn_ids = []
        for session_id, session in zip(item["haystack_session_ids"], item["haystack_sessions"]):
            for turn_idx, turn in enumerate(session):
                turn_id = f"{session_id}::turn{turn_idx}"
                memories.append(
                    {
                        "id": turn_id,
                        "session_id": session_id,
                        "role": turn.get("role", "other"),
                        "text": session_turn_text(turn),
                    }
                )
                if session_id in gold_sessions:
                    gold_turn_ids.append(turn_id)

        if not gold_turn_ids:
            continue

        examples.append(
            {
                "question_id": item["question_id"],
                "question_type": item.get("question_type", "unknown"),
                "query": item["question"],
                "memories": memories,
                "gold_memory_ids": gold_turn_ids,
            }
        )
        if limit and len(examples) >= limit:
            break

    if distractor_sessions:
        examples = add_turn_distractors(examples, distractor_sessions, seed=seed)
    return examples


def add_turn_distractors(examples, target_sessions, seed=7):
    rng = random.Random(seed)
    session_to_turns = {}
    for item in examples:
        for memory in item["memories"]:
            session_to_turns.setdefault(memory["session_id"], []).append(memory)

    session_ids = list(session_to_turns)
    expanded = []
    for item in examples:
        existing_sessions = {m["session_id"] for m in item["memories"]}
        new_memories = list(item["memories"])
        current_sessions = set(existing_sessions)
        candidates = [s for s in session_ids if s not in current_sessions]
        rng.shuffle(candidates)

        for session_id in candidates:
            if len(current_sessions) >= target_sessions:
                break
            for memory in session_to_turns[session_id]:
                new_memories.append(
                    {
                        "id": f"distractor::{memory['id']}",
                        "session_id": f"distractor::{session_id}",
                        "role": memory.get("role", "other"),
                        "text": memory["text"],
                    }
                )
            current_sessions.add(session_id)

        expanded.append({**item, "memories": new_memories})
    return expanded


def prepare_encoded_example(example, encoder, window_size, stride):
    memory_ids = [m["id"] for m in example["memories"]]
    memory_texts = [m["text"] for m in example["memories"]]
    memory_embeddings = encoder.transform(memory_texts)
    query_embedding = encoder.transform([example["query"]])[0]
    windows = make_windows(len(memory_ids), window_size, stride)
    if not windows:
        return None
    max_len = max(len(w) for w in windows)
    dim = memory_embeddings.shape[1]
    batch = np.zeros((len(windows), max_len, dim), dtype=np.float32)
    role_to_id = {"user": 0, "assistant": 1, "system": 2, "other": 2}
    role_ids = [role_to_id.get(m.get("role", "other"), 2) for m in example["memories"]]
    type_batch = np.full((len(windows), max_len), 3, dtype=np.int64)
    for i, window in enumerate(windows):
        batch[i, : len(window)] = memory_embeddings[window]
        type_batch[i, : len(window)] = [role_ids[j] for j in window]
    return {
        **example,
        "memory_ids": memory_ids,
        "memory_embeddings": memory_embeddings,
        "query_embedding": query_embedding,
        "windows": windows,
        "window_tensor": torch.tensor(batch, dtype=torch.float32),
        "window_type_tensor": torch.tensor(type_batch, dtype=torch.long),
    }


def evaluate_item(item, model, device, top_k=10, raw_weights=(0.8,), rerank_top_ns=(50,)):
    q = item["query_embedding"]
    memory_ids = item["memory_ids"]
    memory_embeddings = item["memory_embeddings"]
    windows = item["windows"]

    raw_scores = cosine_scores(q, memory_embeddings)
    mean_blocks = l2_normalize(np.array([memory_embeddings[w].mean(axis=0) for w in windows]))

    with torch.no_grad():
        if getattr(model, "pooling", "mean") == "query_attention":
            query_tensor = torch.tensor(q[None, :], dtype=torch.float32, device=device)
            query_batch = query_tensor.expand(item["window_tensor"].shape[0], -1)
            kwargs = {}
            if "window_type_tensor" in item:
                kwargs["type_ids"] = item["window_type_tensor"].to(device)
            window_batch = item["window_tensor"].to(device)
            conv_blocks_tensor = model(window_batch, query=query_batch, **kwargs)
            conv_blocks = conv_blocks_tensor.cpu().numpy()
            if hasattr(model, "score_windows") and getattr(model, "score_mode", "cosine") != "cosine":
                conv_scores = model.score_windows(window_batch, query=query_batch, **kwargs).cpu().numpy()
            else:
                conv_scores = cosine_scores(q, conv_blocks)
        else:
            kwargs = {}
            if "window_type_tensor" in item:
                kwargs["type_ids"] = item["window_type_tensor"].to(device)
            conv_blocks = model(item["window_tensor"].to(device), **kwargs).cpu().numpy()
            conv_scores = cosine_scores(q, conv_blocks)

    mean_scores = cosine_scores(q, mean_blocks)

    rankings = {
        "raw_turn": [memory_ids[i] for i in np.argsort(-raw_scores)],
        "mean_window": expand_window_ranking(mean_scores, windows, memory_ids),
        "conv1d_window": expand_window_ranking(conv_scores, windows, memory_ids),
    }

    for weight in raw_weights:
        for top_n in rerank_top_ns:
            rankings[f"convmem_chain_rerank_top{top_n}_rw{weight:g}"] = convmem_rerank(
                raw_scores,
                conv_scores,
                windows,
                memory_ids,
                raw_top_n=top_n,
                raw_weight=weight,
            )

    for top_n in rerank_top_ns:
        policy_weight = adaptive_confidence_weight(raw_scores)
        rankings[f"adaptive_confidence_rerank_top{top_n}"] = convmem_rerank(
            raw_scores,
            conv_scores,
            windows,
            memory_ids,
            raw_top_n=top_n,
            raw_weight=policy_weight,
        )

    rows = []
    for method, ranked in rankings.items():
        rows.append(
            {
                "method": method,
                "question_id": item["question_id"],
                "question_type": item["question_type"],
                "gold_chain_len": len(item["gold_memory_ids"]),
                "recall_at_5": recall_at_k(ranked, item["gold_memory_ids"], 5),
                "recall_at_10": recall_at_k(ranked, item["gold_memory_ids"], top_k),
                "recall_at_20": recall_at_k(ranked, item["gold_memory_ids"], 20),
                "hit_at_5": hit_at_k(ranked, item["gold_memory_ids"], 5),
                "hit_at_10": hit_at_k(ranked, item["gold_memory_ids"], top_k),
                "hit_at_20": hit_at_k(ranked, item["gold_memory_ids"], 20),
                "mrr": mrr(ranked, item["gold_memory_ids"]),
                "top20_ids": "||".join(ranked[:20]),
                "gold_ids": "||".join(item["gold_memory_ids"]),
            }
        )
    return rows


def adaptive_confidence_weight(raw_scores):
    order = np.argsort(-raw_scores)
    top = raw_scores[order[:10]]
    gap = float(top[0] - top[4]) if len(top) >= 5 else 0.0
    spread = float(np.std(top)) if len(top) else 0.0

    if gap >= 0.08 or spread >= 0.04:
        return 0.8
    if gap >= 0.04 or spread >= 0.025:
        return 0.6
    return 0.3


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
                "avg_gold_chain_len": float(np.mean([float(x["gold_chain_len"]) for x in items])),
                "recall_at_5": float(np.mean([float(x["recall_at_5"]) for x in items])),
                "recall_at_10": float(np.mean([float(x["recall_at_10"]) for x in items])),
                "recall_at_20": float(np.mean([float(x["recall_at_20"]) for x in items])),
                "hit_at_5": float(np.mean([float(x["hit_at_5"]) for x in items])),
                "hit_at_10": float(np.mean([float(x["hit_at_10"]) for x in items])),
                "hit_at_20": float(np.mean([float(x["hit_at_20"]) for x in items])),
                "mrr": float(np.mean([float(x["mrr"]) for x in items])),
            }
        )
    return sorted(out, key=lambda x: x["recall_at_10"], reverse=True)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/longmemeval_s_cleaned.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--encoder", choices=["tfidf", "sbert"], default="sbert")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--embedding-cache", default=None)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--conv-layers", type=int, default=1)
    parser.add_argument("--pooling", choices=["mean", "query_attention"], default="mean")
    parser.add_argument("--multi-scale-kernels", default=None)
    parser.add_argument("--use-role-embeddings", action="store_true")
    parser.add_argument("--projection", choices=["none", "linear", "mlp"], default="none")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--raw-weights", default="0.7,0.8,0.9")
    parser.add_argument("--rerank-top-ns", default="50,100")
    parser.add_argument("--distractor-sessions", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="results/chain")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if args.device == "auto" and device != "cuda":
        device = "cpu"

    examples = load_chain_examples(
        Path(__file__).parent / args.data,
        limit=args.limit,
        distractor_sessions=args.distractor_sessions,
        seed=args.seed,
    )
    random.shuffle(examples)
    split = int(len(examples) * args.train_ratio)
    train_examples = examples[:split]
    test_examples = examples[split:]

    texts = []
    for item in train_examples:
        texts.append(item["query"])
        texts.extend(m["text"] for m in item["memories"])

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
    encoded_test = [
        prepare_encoded_example(x, encoder, args.window_size, args.stride)
        for x in test_examples
    ]
    encoded_train = [x for x in encoded_train if x is not None]
    encoded_test = [x for x in encoded_test if x is not None]

    dim = encoded_train[0]["memory_embeddings"].shape[1]
    multi_scale_kernels = None
    if args.multi_scale_kernels:
        multi_scale_kernels = [
            int(x.strip()) for x in args.multi_scale_kernels.split(",") if x.strip()
        ]
    model = ConvMemoryEncoder(
        dim,
        args.kernel_size,
        args.conv_layers,
        pooling=args.pooling,
        multi_scale_kernels=multi_scale_kernels,
        type_vocab_size=4 if args.use_role_embeddings else 0,
        projection=args.projection,
    ).to(device)
    start = time.perf_counter()
    train_conv(model, encoded_train, epochs=args.epochs, device=device)
    train_time = time.perf_counter() - start

    raw_weights = [float(x.strip()) for x in args.raw_weights.split(",") if x.strip()]
    rerank_top_ns = [int(x.strip()) for x in args.rerank_top_ns.split(",") if x.strip()]

    rows = []
    start = time.perf_counter()
    for item in encoded_test:
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
    write_csv(out_dir / "chain_detailed_results.csv", rows)
    write_csv(out_dir / "chain_summary_results.csv", summary)

    avg_memories = np.mean([len(x["memories"]) for x in encoded_test])
    avg_windows = np.mean([len(x["windows"]) for x in encoded_test])
    avg_gold = np.mean([len(x["gold_memory_ids"]) for x in encoded_test])

    print("\nConvMem chain retrieval experiment")
    print(f"train questions: {len(encoded_train)}")
    print(f"test questions: {len(encoded_test)}")
    print(f"device: {device}")
    print(f"encoder: {args.encoder}")
    print(f"pooling: {args.pooling}")
    if multi_scale_kernels:
        print(f"multi-scale kernels: {multi_scale_kernels}")
    print(f"role embeddings: {args.use_role_embeddings}")
    print(f"projection: {args.projection}")
    print(f"avg turn memories per question: {avg_memories:.1f}")
    print(f"avg windows per question: {avg_windows:.1f}")
    print(f"avg gold chain len: {avg_gold:.1f}")
    print(f"train time: {train_time:.1f}s")
    print(f"eval latency: {1000 * eval_time / max(1, len(encoded_test)):.2f}ms/query")
    print("\nmethod                              questions  recall@10 hit@10 mrr")
    for row in summary:
        print(
            f"{row['method']:<35} "
            f"{row['questions']:<9} "
            f"{row['recall_at_10']:.3f}     "
            f"{row['hit_at_10']:.3f}  "
            f"{row['mrr']:.3f}"
        )
    print(f"\nSaved: {out_dir / 'chain_summary_results.csv'}")
    print(f"Saved: {out_dir / 'chain_detailed_results.csv'}")


if __name__ == "__main__":
    main()

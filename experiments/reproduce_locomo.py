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

from convmemory import ConvMemoryReranker, RerankConfig
from convmemory.api import ConvMemory
from convmemory.metrics import hit_at_k, mrr, recall_at_k
from experiments.support.convmem_ce_lite import (
    CELiteScorer,
    cached_teacher_turn_scores,
    load_teacher_cache,
    save_teacher_cache,
    train_one_item,
)
from experiments.support.convmem_chain_benchmark import prepare_encoded_example
from experiments.support.convmem_locomo_benchmark import load_locomo_examples
from experiments.support.convmem_longmemeval import (
    MixerConvMemoryEncoder,
    SentenceTransformerTextEncoder,
    resolve_local_model_path,
)
from experiments.support.locomo_crossencoder_baseline import choose_split


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    by_method = {}
    for row in rows:
        by_method.setdefault(row["method"], []).append(row)
    out = []
    for method, items in by_method.items():
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/locomo10.json")
    parser.add_argument("--encoder-model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--embedding-cache", default=None)
    parser.add_argument("--embedding-cache-key", default=None)
    parser.add_argument("--cross-encoder-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--teacher-cache", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--kernel", type=int, default=3)
    parser.add_argument("--raw-top-n", type=int, default=500)
    parser.add_argument("--candidate-top-n", type=int, default=500)
    parser.add_argument("--cross-top-n", type=int, default=500)
    parser.add_argument("--eval-cross-encoder", action="store_true")
    parser.add_argument("--save-pretrained", default=None)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--raw-weight", type=float, default=0.025)
    parser.add_argument("--gold-weight", type=float, default=0.35)
    parser.add_argument("--first-rank-weight", type=float, default=0.05)
    parser.add_argument("--first-rank-target", choices=["teacher", "best_gold"], default="teacher")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--encoder-batch-size", type=int, default=128)
    parser.add_argument("--cross-batch-size", type=int, default=128)
    parser.add_argument("--out", default="results/reproduce_locomo")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if args.device == "auto" and device != "cuda":
        device = "cpu"

    examples = load_locomo_examples(Path(args.data))
    train_examples = choose_split(examples, "dev", 0.5, args.seed)
    test_examples = choose_split(examples, "test", 0.5, args.seed)
    if args.train_limit:
        train_examples = train_examples[: args.train_limit]
    if args.test_limit:
        test_examples = test_examples[: args.test_limit]

    encoder = SentenceTransformerTextEncoder(
        model_name=args.encoder_model,
        device=device,
        batch_size=args.encoder_batch_size,
        cache_path=args.embedding_cache,
        cache_key=args.embedding_cache_key,
    )
    cross_encoder = CrossEncoder(resolve_local_model_path(args.cross_encoder_model), device=device)

    first = prepare_encoded_example(train_examples[0], encoder, args.window_size, args.stride)
    dim = first["memory_embeddings"].shape[1]
    model = MixerConvMemoryEncoder(
        dim,
        window_size=args.window_size,
        kernel_size=args.kernel,
        hidden_dim=256,
        token_mlp_dim=32,
        channel_mlp_dim=512,
        output_mode="residual",
        output_gate_init=0.1,
        score_mode="cosine",
    ).to(device)
    scorer = CELiteScorer(dim, hidden_dim=256, extra_scalar_features=5).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(scorer.parameters()),
        lr=2e-4,
        weight_decay=1e-4,
    )

    teacher_cache_path = Path(args.teacher_cache) if args.teacher_cache else None
    teacher_cache = load_teacher_cache(teacher_cache_path)
    start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        scorer.train()
        losses = []
        for idx, example in enumerate(train_examples, start=1):
            item = first if epoch == 0 and idx == 1 else prepare_encoded_example(
                example,
                encoder,
                args.window_size,
                args.stride,
            )
            teacher_indices, teacher_scores, _ = cached_teacher_turn_scores(
                item,
                cross_encoder,
                raw_top_n=args.raw_top_n,
                batch_size=args.cross_batch_size,
                cache=teacher_cache,
            )
            loss = train_one_item(
                model,
                scorer,
                optimizer,
                item,
                teacher_indices,
                teacher_scores,
                device=device,
                teacher_temperature=0.1,
                pairwise_top_k=16,
                pairwise_bottom_k=64,
                negative_mode="bottom",
                hard_negative_pool=128,
                gold_weight=args.gold_weight,
                score_mse_weight=0.0,
                first_rank_weight=args.first_rank_weight,
                first_rank_target=args.first_rank_target,
                dca_router_block_size=32,
                lexical_features=True,
            )
            losses.append(loss)
            if idx % 100 == 0:
                print(
                    f"epoch {epoch + 1}/{args.epochs} "
                    f"processed {idx}/{len(train_examples)} "
                    f"loss={np.mean(losses[-100:]):.4f}",
                    flush=True,
                )
    save_teacher_cache(teacher_cache_path, teacher_cache)
    train_time = time.perf_counter() - start

    reranker = ConvMemoryReranker(
        model,
        scorer,
        config=RerankConfig(
            window_size=args.window_size,
            stride=args.stride,
            candidate_top_n=args.candidate_top_n,
            raw_weight=args.raw_weight,
            dca_router_block_size=32,
            lexical_features=True,
        ),
        device=device,
    )

    if args.save_pretrained:
        package = ConvMemory(
            conv_model=model,
            scorer=scorer,
            config=reranker.config,
            device=device,
            embedding_model_name=args.encoder_model,
            model_config={
                "embedding_dim": dim,
                "window_size": args.window_size,
                "kernel_size": args.kernel,
                "hidden_dim": 256,
                "token_mlp_dim": 32,
                "channel_mlp_dim": 512,
                "extra_scalar_features": 5,
            },
        )
        package.save_pretrained(args.save_pretrained)
        print(f"saved pretrained ConvMemory: {args.save_pretrained}", flush=True)

    if args.skip_eval:
        print("skip eval requested; exiting after training/export.", flush=True)
        return

    rows = []
    eval_start = time.perf_counter()
    model.eval()
    scorer.eval()
    for eval_idx, example in enumerate(test_examples, start=1):
        item = prepare_encoded_example(example, encoder, args.window_size, args.stride)
        raw_scores = item["memory_embeddings"] @ item["query_embedding"]
        raw_order = np.argsort(-raw_scores)
        raw_ranked = [item["memory_ids"][int(i)] for i in raw_order]
        conv_ranked = [x.memory_id for x in reranker.rerank_item(item)]
        method_rankings = [
            ("raw_turn", raw_ranked),
            ("convmemory", conv_ranked),
        ]
        if args.eval_cross_encoder:
            cross_top_n = min(args.cross_top_n, len(raw_order))
            cross_candidate_indices = raw_order[:cross_top_n]
            pairs = [
                (item["query"], item["memories"][int(i)]["text"])
                for i in cross_candidate_indices
            ]
            cross_scores = cross_encoder.predict(
                pairs,
                batch_size=args.cross_batch_size,
                show_progress_bar=False,
            )
            cross_order = [
                int(cross_candidate_indices[i])
                for i in np.argsort(-np.asarray(cross_scores, dtype=np.float32))
            ]
            cross_ranked = [item["memory_ids"][i] for i in cross_order]
            cross_seen = set(cross_ranked)
            cross_ranked.extend([x for x in raw_ranked if x not in cross_seen])
            method_rankings.append((f"cross_encoder_top{args.cross_top_n}", cross_ranked))

        for method, ranked in method_rankings:
            rows.append(
                {
                    "method": method,
                    "question_id": item["question_id"],
                    "question_type": item["question_type"],
                    "gold_chain_len": len(item["gold_memory_ids"]),
                    "recall_at_10": recall_at_k(ranked, item["gold_memory_ids"], 10),
                    "hit_at_10": hit_at_k(ranked, item["gold_memory_ids"], 10),
                    "mrr": mrr(ranked, item["gold_memory_ids"]),
                    "top20_ids": "||".join(ranked[:20]),
                    "gold_ids": "||".join(item["gold_memory_ids"]),
                }
            )
        if eval_idx % 100 == 0:
            print(f"evaluated {eval_idx}/{len(test_examples)}", flush=True)
    eval_time = time.perf_counter() - eval_start

    out_dir = Path(args.out)
    summary = summarize(rows)
    write_csv(out_dir / "detailed_results.csv", rows)
    write_csv(out_dir / "summary_results.csv", summary)

    print("\nConvMemory reproducibility run")
    print(f"device: {device}")
    print(f"train questions: {len(train_examples)}")
    print(f"test questions: {len(test_examples)}")
    print(f"train time: {train_time:.1f}s")
    print(f"eval latency: {1000 * eval_time / max(1, len(test_examples)):.2f}ms/query")
    print("method       questions recall@10 hit@10 mrr")
    for row in summary:
        print(
            f"{row['method']:<12} "
            f"{row['questions']:<9} "
            f"{row['recall_at_10']:.3f}     "
            f"{row['hit_at_10']:.3f}  "
            f"{row['mrr']:.3f}"
        )


if __name__ == "__main__":
    main()

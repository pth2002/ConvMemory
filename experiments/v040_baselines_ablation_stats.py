import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from convmemory import ConvMemory
from convmemory.metrics import hit_at_k, mrr, recall_at_k
from convmemory.reranker import candidate_local_windows, window_tensor
from convmemory.scoring import (
    TOKEN_RE,
    STOPWORDS,
    build_memory_to_windows,
    candidate_features,
    cosine_scores,
    lexical_signature,
    normalize_scores,
    rerank_candidates,
    window_scores,
)
from experiments.support.convmem_chain_benchmark import prepare_encoded_example
from experiments.support.convmem_locomo_benchmark import load_locomo_examples
from experiments.support.convmem_longmemeval import SentenceTransformerTextEncoder, resolve_local_model_path
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


def tokenize(text):
    return [t for t in TOKEN_RE.findall(str(text).lower()) if len(t) > 1 and t not in STOPWORDS]


def bm25_scores(query, texts, k1=1.5, b=0.75):
    query_terms = tokenize(query)
    docs = [tokenize(text) for text in texts]
    lengths = np.asarray([len(doc) for doc in docs], dtype=np.float32)
    avg_len = float(lengths.mean()) if len(lengths) else 0.0
    df = Counter()
    for doc in docs:
        df.update(set(doc))
    n_docs = max(1, len(docs))
    scores = np.zeros(len(docs), dtype=np.float32)
    for term in query_terms:
        if term not in df:
            continue
        idf = np.log(1.0 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
        for idx, doc in enumerate(docs):
            tf = doc.count(term)
            if tf == 0:
                continue
            denom = tf + k1 * (1.0 - b + b * lengths[idx] / max(avg_len, 1e-6))
            scores[idx] += idf * (tf * (k1 + 1.0)) / max(denom, 1e-6)
    return scores


def order_from_scores(scores):
    return [int(i) for i in np.argsort(-np.asarray(scores, dtype=np.float32))]


def ids_from_order(memory_ids, order):
    return [memory_ids[int(i)] for i in order]


def rank_positions(order, n_items):
    ranks = np.full(n_items, fill_value=n_items + 1, dtype=np.float32)
    for rank, idx in enumerate(order, start=1):
        ranks[int(idx)] = rank
    return ranks


def rrf_rank(orders, n_items, k=60):
    scores = np.zeros(n_items, dtype=np.float32)
    for order in orders:
        ranks = rank_positions(order, n_items)
        scores += 1.0 / (k + ranks)
    return order_from_scores(scores)


def temporal_neighbor_scores(scores, window_size):
    scores = np.asarray(scores, dtype=np.float32)
    half = max(1, int(window_size) // 2)
    out = np.zeros_like(scores)
    for idx in range(len(scores)):
        start = max(0, idx - half)
        end = min(len(scores), idx + half + 1)
        out[idx] = float(scores[start:end].max())
    return out


def recency_dense_scores(raw_scores, lam):
    raw = normalize_scores(raw_scores)
    n_items = len(raw)
    if n_items <= 1:
        return raw
    recency = np.linspace(0.0, 1.0, n_items, dtype=np.float32)
    return raw + float(lam) * recency


def lexical_dense_scores(raw_scores, bm25):
    return normalize_scores(raw_scores) + normalize_scores(bm25)


def add_metric_row(rows, seed, split, method, item, ranked_ids, extra=None):
    row = {
        "seed": int(seed),
        "split": split,
        "method": method,
        "question_id": item["question_id"],
        "question_type": item["question_type"],
        "gold_chain_len": len(item["gold_memory_ids"]),
        "recall_at_5": recall_at_k(ranked_ids, item["gold_memory_ids"], 5),
        "hit_at_5": hit_at_k(ranked_ids, item["gold_memory_ids"], 5),
        "recall_at_10": recall_at_k(ranked_ids, item["gold_memory_ids"], 10),
        "hit_at_10": hit_at_k(ranked_ids, item["gold_memory_ids"], 10),
        "recall_at_20": recall_at_k(ranked_ids, item["gold_memory_ids"], 20),
        "hit_at_20": hit_at_k(ranked_ids, item["gold_memory_ids"], 20),
        "mrr": mrr(ranked_ids, item["gold_memory_ids"]),
        "top20_ids": "||".join(ranked_ids[:20]),
        "gold_ids": "||".join(item["gold_memory_ids"]),
    }
    if extra:
        row.update(extra)
    rows.append(row)


def feature_masked_rank(convmemory, item, mask_name, candidate_top_n=500, window_mode="candidate_local"):
    raw_scores = cosine_scores(item["query_embedding"], item["memory_embeddings"])
    candidate_indices = np.argsort(-raw_scores)[: min(candidate_top_n, len(raw_scores))]
    scoring_item = item
    if window_mode == "candidate_local":
        windows = candidate_local_windows(
            len(item["memory_ids"]),
            candidate_indices,
            convmemory.config.window_size,
        )
        scoring_item = {
            **item,
            "windows": windows,
            "window_tensor": window_tensor(item["memory_embeddings"], windows),
        }
    elif window_mode == "full":
        scoring_item = {
            **item,
            "window_tensor": window_tensor(item["memory_embeddings"], item["windows"]),
        }
    else:
        raise ValueError("window_mode must be 'candidate_local' or 'full'")

    with torch.no_grad():
        window_logits = window_scores(convmemory.reranker.conv_model, scoring_item, convmemory.device)
        memory_to_windows = build_memory_to_windows(scoring_item["windows"])
        features, _, _ = candidate_features(
            convmemory.reranker.conv_model,
            scoring_item,
            candidate_indices,
            convmemory.device,
            raw_scores_all=raw_scores,
            window_logits=window_logits,
            memory_to_windows=memory_to_windows,
            dca_router_block_size=convmemory.config.dca_router_block_size,
            lexical_features=convmemory.config.lexical_features,
        )

        dim = item["memory_embeddings"].shape[1]
        scalar_start = dim * 4
        raw_col = scalar_start
        window_col = scalar_start + 1
        router_col = scalar_start + 4
        lexical_start = scalar_start + 5

        masked = features.clone()
        if mask_name == "full":
            pass
        elif mask_name == "no_temporal_window":
            masked[:, window_col] = 0.0
        elif mask_name == "no_lexical":
            masked[:, lexical_start : lexical_start + 4] = 0.0
        elif mask_name == "no_router":
            masked[:, router_col] = 0.0
        elif mask_name == "no_raw_feature":
            masked[:, raw_col] = 0.0
        elif mask_name == "temporal_only":
            kept = masked[:, window_col].clone()
            masked.zero_()
            masked[:, window_col] = kept
        else:
            raise ValueError(f"Unknown mask: {mask_name}")

        scores = convmemory.reranker.scorer(masked).detach().cpu().numpy()

    raw_weight = convmemory.config.raw_weight
    if mask_name in {"no_raw_feature", "temporal_only"}:
        raw_weight = 0.0
    return rerank_candidates(
        raw_scores,
        candidate_indices,
        scores,
        item["memory_ids"],
        raw_weight=raw_weight,
    )


def baseline_rankings(item, window_size, recency_lambdas):
    memory_ids = item["memory_ids"]
    texts = [memory["text"] for memory in item["memories"]]
    raw_scores = cosine_scores(item["query_embedding"], item["memory_embeddings"])
    bm25 = bm25_scores(item["query"], texts)
    raw_order = order_from_scores(raw_scores)
    bm25_order = order_from_scores(bm25)
    temporal_order = order_from_scores(temporal_neighbor_scores(raw_scores, window_size))
    lexical_dense_order = order_from_scores(lexical_dense_scores(raw_scores, bm25))
    rrf_dense_lexical = rrf_rank([raw_order, bm25_order], len(memory_ids))
    rrf_dense_lexical_temporal = rrf_rank([raw_order, bm25_order, temporal_order], len(memory_ids))

    rankings = {
        "raw_dense": ids_from_order(memory_ids, raw_order),
        "bm25": ids_from_order(memory_ids, bm25_order),
        "dense_plus_lexical_score": ids_from_order(memory_ids, lexical_dense_order),
        "dense_lexical_rrf": ids_from_order(memory_ids, rrf_dense_lexical),
        "dense_lexical_temporal_rrf": ids_from_order(memory_ids, rrf_dense_lexical_temporal),
    }
    for lam in recency_lambdas:
        rankings[f"recency_dense_lambda{lam:g}"] = ids_from_order(
            memory_ids,
            order_from_scores(recency_dense_scores(raw_scores, lam)),
        )
    return rankings


def cross_encoder_rankings(item, raw_top_n, cross_encoders, batch_size):
    if not cross_encoders:
        return {}
    raw_scores = cosine_scores(item["query_embedding"], item["memory_embeddings"])
    raw_order = order_from_scores(raw_scores)
    candidate_indices = raw_order[: min(raw_top_n, len(raw_order))]
    rankings = {}
    for name, cross_encoder in cross_encoders.items():
        pairs = [(item["query"], item["memories"][int(idx)]["text"]) for idx in candidate_indices]
        scores = cross_encoder.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        ce_order = [candidate_indices[int(i)] for i in np.argsort(-np.asarray(scores, dtype=np.float32))]
        ce_ids = ids_from_order(item["memory_ids"], ce_order)
        seen = set(ce_ids)
        ce_ids.extend([item["memory_ids"][idx] for idx in raw_order if item["memory_ids"][idx] not in seen])
        rankings[f"cross_encoder::{name}::top{raw_top_n}"] = ce_ids
    return rankings


def summarize_by_seed(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], int(row["seed"]))].append(row)

    per_seed = []
    for (method, seed), items in sorted(grouped.items()):
        out = {"method": method, "seed": seed, "questions": len(items)}
        for metric in ["recall_at_5", "hit_at_5", "recall_at_10", "hit_at_10", "recall_at_20", "hit_at_20", "mrr"]:
            out[metric] = float(np.mean([float(item[metric]) for item in items]))
        per_seed.append(out)

    by_method = defaultdict(list)
    for row in per_seed:
        by_method[row["method"]].append(row)

    summary = []
    for method, items in sorted(by_method.items()):
        out = {"method": method, "seeds": len(items), "questions_total": sum(int(x["questions"]) for x in items)}
        for metric in ["recall_at_5", "hit_at_5", "recall_at_10", "hit_at_10", "recall_at_20", "hit_at_20", "mrr"]:
            values = np.asarray([float(item[metric]) for item in items], dtype=np.float32)
            out[f"{metric}_mean"] = float(values.mean())
            out[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary.append(out)
    return per_seed, sorted(summary, key=lambda x: x["recall_at_10_mean"], reverse=True)


def paired_bootstrap(rows, baseline, contender, metrics, samples, seed):
    by_unit = defaultdict(dict)
    for row in rows:
        unit = f"{row['seed']}::{row['split']}::{row['question_id']}"
        by_unit[unit][row["method"]] = row
    units = [unit for unit, values in by_unit.items() if baseline in values and contender in values]
    rng = np.random.default_rng(seed)
    out = []
    if not units:
        return out
    for metric in metrics:
        diffs = np.asarray(
            [
                float(by_unit[unit][contender][metric]) - float(by_unit[unit][baseline][metric])
                for unit in units
            ],
            dtype=np.float32,
        )
        observed = float(diffs.mean())
        boot = np.zeros(samples, dtype=np.float32)
        for i in range(samples):
            idx = rng.integers(0, len(diffs), size=len(diffs))
            boot[i] = float(diffs[idx].mean())
        lo, hi = np.percentile(boot, [2.5, 97.5])
        if observed >= 0:
            p_value = 2.0 * min(float(np.mean(boot <= 0.0)), float(np.mean(boot >= 0.0)))
        else:
            p_value = 2.0 * min(float(np.mean(boot >= 0.0)), float(np.mean(boot <= 0.0)))
        out.append(
            {
                "baseline": baseline,
                "contender": contender,
                "metric": metric,
                "paired_units": len(units),
                "mean_delta": observed,
                "ci95_low": float(lo),
                "ci95_high": float(hi),
                "paired_bootstrap_p": min(1.0, p_value),
            }
        )
    return out


def write_report(path, summary, bootstrap_rows, args):
    lines = [
        "# v0.40 Baselines, Ablations, And Statistics",
        "",
        "This report addresses the main evaluation gaps: simple baselines, feature ablations, multi-seed reporting, and paired bootstrap intervals.",
        "",
        f"- Seeds: `{', '.join(str(x) for x in args.seeds)}`",
        f"- Split: `{args.split}` with dev ratio `{args.dev_ratio}`",
        f"- Encoder: `{args.encoder_model}`",
        f"- Checkpoint: `{args.checkpoint}`",
        "",
        "## Summary",
        "",
        "| Method | Seeds | Recall@10 mean | Recall@10 std | Hit@10 mean | MRR mean | MRR std |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| `{row['method']}` | {row['seeds']} | {row['recall_at_10_mean']:.4f} | "
            f"{row['recall_at_10_std']:.4f} | {row['hit_at_10_mean']:.4f} | "
            f"{row['mrr_mean']:.4f} | {row['mrr_std']:.4f} |"
        )

    if bootstrap_rows:
        lines.extend(
            [
                "",
                "## Paired Bootstrap",
                "",
                "| Baseline | Contender | Metric | Delta | 95% CI | p-value | Units |",
                "|---|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in bootstrap_rows:
            lines.append(
                f"| `{row['baseline']}` | `{row['contender']}` | `{row['metric']}` | "
                f"{row['mean_delta']:+.4f} | [{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}] | "
                f"{row['paired_bootstrap_p']:.4f} | {row['paired_units']} |"
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
    parser.add_argument("--recency-lambdas", default="0.01,0.03,0.1,0.3,1.0")
    parser.add_argument("--cross-encoder-models", default="")
    parser.add_argument("--cross-top-n", type=int, default=500)
    parser.add_argument("--cross-batch-size", type=int, default=128)
    parser.add_argument("--encoder-batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--out", default="results/v040/baselines_ablation_stats")
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
    recency_lambdas = [float(x.strip()) for x in args.recency_lambdas.split(",") if x.strip()]

    cross_encoders = {}
    if args.cross_encoder_models.strip():
        from sentence_transformers import CrossEncoder

        for model_name in [x.strip() for x in args.cross_encoder_models.split(",") if x.strip()]:
            cross_encoders[model_name] = CrossEncoder(resolve_local_model_path(model_name), device=device)

    rows = []
    masks = [
        "full",
        "no_temporal_window",
        "no_lexical",
        "no_router",
        "no_raw_feature",
        "temporal_only",
    ]
    checkpoint_dim = int(convmemory.model_config.get("embedding_dim", -1))
    warned_dim_mismatch = False
    for seed in args.seeds:
        split_examples = choose_split(examples, args.split, args.dev_ratio, seed)
        if args.limit:
            split_examples = split_examples[: args.limit]
        for idx, example in enumerate(split_examples, start=1):
            item = prepare_encoded_example(example, encoder, convmemory.config.window_size, convmemory.config.stride)
            baseline_ranked = baseline_rankings(item, convmemory.config.window_size, recency_lambdas)
            for method, ranked in baseline_ranked.items():
                add_metric_row(rows, seed, args.split, method, item, ranked)

            embedding_dim = int(item["memory_embeddings"].shape[1])
            if embedding_dim == checkpoint_dim:
                for mask in masks:
                    ranked = feature_masked_rank(
                        convmemory,
                        item,
                        mask,
                        candidate_top_n=args.candidate_top_n,
                        window_mode="candidate_local",
                    )
                    add_metric_row(rows, seed, args.split, f"convmemory_mask::{mask}", item, ranked)

                full_window_ranked = feature_masked_rank(
                    convmemory,
                    item,
                    "full",
                    candidate_top_n=args.candidate_top_n,
                    window_mode="full",
                )
                add_metric_row(rows, seed, args.split, "convmemory_mask::full_global_windows", item, full_window_ranked)
            elif not warned_dim_mismatch:
                print(
                    "Skipping ConvMemory checkpoint evaluation because encoder dimension "
                    f"{embedding_dim} != checkpoint dimension {checkpoint_dim}. "
                    "Raw/BM25/lexical/CE baselines will still run.",
                    flush=True,
                )
                warned_dim_mismatch = True

            ce_ranked = cross_encoder_rankings(item, args.cross_top_n, cross_encoders, args.cross_batch_size)
            for method, ranked in ce_ranked.items():
                add_metric_row(rows, seed, args.split, method, item, ranked, extra={"cross_top_n": args.cross_top_n})

            if idx % 100 == 0 or idx == len(split_examples):
                print(f"seed {seed}: {idx}/{len(split_examples)}", flush=True)

    per_seed, summary = summarize_by_seed(rows)
    bootstrap_rows = []
    contenders = [
        "convmemory_mask::full",
        "convmemory_mask::no_temporal_window",
        "dense_lexical_rrf",
        "dense_lexical_temporal_rrf",
    ]
    for contender in contenders:
        bootstrap_rows.extend(
            paired_bootstrap(
                rows,
                baseline="raw_dense",
                contender=contender,
                metrics=["recall_at_10", "hit_at_10", "mrr"],
                samples=args.bootstrap_samples,
                seed=13,
            )
        )
    bootstrap_rows.extend(
        paired_bootstrap(
            rows,
            baseline="convmemory_mask::full",
            contender="convmemory_mask::no_temporal_window",
            metrics=["recall_at_10", "hit_at_10", "mrr"],
            samples=args.bootstrap_samples,
            seed=17,
        )
    )

    out_dir = ROOT / args.out
    write_csv(out_dir / "detailed.csv", rows)
    write_csv(out_dir / "summary_by_seed.csv", per_seed)
    write_csv(out_dir / "summary.csv", summary)
    write_csv(out_dir / "paired_bootstrap.csv", bootstrap_rows)
    write_report(out_dir / "REPORT.md", summary, bootstrap_rows, args)
    print(f"Saved v0.40 results to {out_dir}", flush=True)


if __name__ == "__main__":
    main()

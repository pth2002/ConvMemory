import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import CrossEncoder

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from convmemory import ConvMemory, CompressionRouteConfig, CompressionRouter
from convmemory.metrics import hit_at_k, mrr, recall_at_k
from experiments.support.convmem_chain_benchmark import prepare_encoded_example
from experiments.support.convmem_locomo_benchmark import load_locomo_examples
from experiments.support.convmem_longmemeval import SentenceTransformerTextEncoder, resolve_local_model_path
from experiments.support.locomo_notes import encode_memory_bank, load_locomo_memory_sets
from experiments.support.locomo_crossencoder_baseline import choose_split


ROUTE_PRESETS = {
    "fast": CompressionRouteConfig(
        note_depth=240,
        max_sources_per_note=5,
        max_candidates=450,
        raw_anchor=80,
    ),
    "balanced": CompressionRouteConfig(
        note_depth=240,
        max_sources_per_note=8,
        max_candidates=450,
        raw_anchor=50,
    ),
}


def build_note_banks(banks):
    obs = [dict(item, note_type="observation") for item in banks["observation_notes"]]
    events = [dict(item, note_type="event") for item in banks["event_notes"]]
    sessions = [dict(item, note_type="session") for item in banks["session_summaries"]]
    return {"all": obs + events + sessions}


FAMILIES = ["full_top50", "full_top100", "balanced_top50", "balanced_top100"]


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_cache(path):
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_cache(path, cache):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(cache), encoding="utf-8")
    tmp.replace(path)


def ranked_ids(results):
    return [result.memory_id for result in results]


def result_score_map(results):
    return {result.memory_id: float(result.score) for result in results}


def append_remaining(primary_ids, fallback_ids):
    seen = set(primary_ids)
    return list(primary_ids) + [memory_id for memory_id in fallback_ids if memory_id not in seen]


def zscore(values):
    values = np.asarray(values, dtype=np.float32)
    std = float(values.std())
    if std < 1e-8:
        return values - float(values.mean())
    return (values - float(values.mean())) / std


def metric_row(seed, split_name, method, family, alpha, item, ranked, ce_pairs):
    return {
        "seed": seed,
        "split": split_name,
        "method": method,
        "family": family,
        "alpha": float(alpha),
        "question_id": item["question_id"],
        "question_type": item["question_type"],
        "ce_pairs": int(ce_pairs),
        "recall_at_10": recall_at_k(ranked, item["gold_memory_ids"], 10),
        "hit_at_10": hit_at_k(ranked, item["gold_memory_ids"], 10),
        "mrr": mrr(ranked[:10], item["gold_memory_ids"]),
    }


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["seed"], row["split"], row["method"], row["family"], row["alpha"]), []).append(row)
    out = []
    for (seed, split_name, method, family, alpha), items in sorted(groups.items()):
        out.append(
            {
                "seed": seed,
                "split": split_name,
                "method": method,
                "family": family,
                "alpha": alpha,
                "questions": len(items),
                "avg_ce_pairs": float(np.mean([float(x["ce_pairs"]) for x in items])),
                "recall_at_10": float(np.mean([float(x["recall_at_10"]) for x in items])),
                "hit_at_10": float(np.mean([float(x["hit_at_10"]) for x in items])),
                "mrr": float(np.mean([float(x["mrr"]) for x in items])),
            }
        )
    return out


def route_indices(route_name, item, sample_id, note_banks, encoded_notes):
    route_config = ROUTE_PRESETS[route_name]
    routed = CompressionRouter(route_config).route(
        query_embedding=item["query_embedding"],
        memory_embeddings=item["memory_embeddings"],
        memory_ids=item["memory_ids"],
        compressed_embeddings=encoded_notes[sample_id]["all"],
        compressed_memories=note_banks[sample_id]["all"],
    )
    indices = np.asarray(routed.candidate_indices, dtype=np.int64)
    if len(indices) == 0:
        raw_scores = item["memory_embeddings"] @ item["query_embedding"]
        raw_order = np.argsort(-raw_scores)
        indices = np.asarray(raw_order[: route_config.raw_anchor], dtype=np.int64)
    return indices


def ensure_ce_scores(item, indices, cross_encoder, batch_size, cache, cache_path, counter, save_every):
    qcache = cache.setdefault(item["question_id"], {})
    missing = [int(idx) for idx in indices if str(int(idx)) not in qcache]
    if missing:
        pairs = [(item["query"], item["memories"][int(idx)]["text"]) for idx in missing]
        scores = cross_encoder.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        for idx, score in zip(missing, scores):
            qcache[str(int(idx))] = float(score)
        counter["misses"] += len(missing)
        if cache_path and counter["questions"] % save_every == 0:
            save_cache(cache_path, cache)
    return qcache


def fuse_rank(candidate_ids, conv_scores, ce_scores, alpha):
    conv_values = np.asarray([conv_scores[memory_id] for memory_id in candidate_ids], dtype=np.float32)
    ce_values = np.asarray([ce_scores[memory_id] for memory_id in candidate_ids], dtype=np.float32)
    fused = (1.0 - alpha) * zscore(conv_values) + alpha * zscore(ce_values)
    order = np.argsort(-fused)
    return [candidate_ids[int(i)] for i in order]


def ce_only_rank(candidate_ids, ce_scores):
    return sorted(candidate_ids, key=lambda memory_id: -float(ce_scores[memory_id]))


def evaluate_example(seed, split_name, example, samples, note_banks, encoded_notes, encoder, convmemory, cross_encoder, args, cache, counter):
    sample_id = example["question_id"].split("::", 1)[0]
    item = prepare_encoded_example(example, encoder, convmemory.config.window_size, convmemory.config.stride)
    memory_ids = item["memory_ids"]
    id_to_idx = {memory_id: idx for idx, memory_id in enumerate(memory_ids)}

    conv_full = convmemory.reranker.rerank_item(
        item,
        candidate_top_n=args.raw_top_n,
        window_mode="candidate_local",
    )
    balanced = route_indices("balanced", item, sample_id, note_banks, encoded_notes)
    conv_balanced = convmemory.reranker.rerank_item(
        item,
        candidate_indices=balanced,
        window_mode="candidate_local",
    )
    conv_full_ids = ranked_ids(conv_full)
    conv_bal_ids = ranked_ids(conv_balanced)
    conv_full_scores = result_score_map(conv_full)
    conv_bal_scores = result_score_map(conv_balanced)

    families = {
        "full_top50": (conv_full_ids[:50], conv_full_ids, conv_full_scores),
        "full_top100": (conv_full_ids[:100], conv_full_ids, conv_full_scores),
        "balanced_top50": (conv_bal_ids[:50], conv_bal_ids, conv_bal_scores),
        "balanced_top100": (conv_bal_ids[:100], conv_bal_ids, conv_bal_scores),
    }
    all_candidate_indices = []
    seen = set()
    for candidate_ids, _, _ in families.values():
        for memory_id in candidate_ids:
            idx = int(id_to_idx[memory_id])
            if idx not in seen:
                seen.add(idx)
                all_candidate_indices.append(idx)
    counter["questions"] += 1
    qcache = ensure_ce_scores(
        item,
        all_candidate_indices,
        cross_encoder,
        args.cross_batch_size,
        cache,
        Path(args.teacher_cache) if args.teacher_cache else None,
        counter,
        args.cache_save_every,
    )
    ce_by_id = {memory_ids[int(idx)]: float(score) for idx, score in qcache.items()}

    rows = [
        metric_row(seed, split_name, "convmemory_full", "baseline", 0.0, item, conv_full_ids, 0),
        metric_row(seed, split_name, "convmemory_balanced", "baseline", 0.0, item, conv_bal_ids, 0),
    ]
    for family, (candidate_ids, fallback_ids, conv_scores) in families.items():
        ce_ranked = append_remaining(ce_only_rank(candidate_ids, ce_by_id), fallback_ids)
        rows.append(metric_row(seed, split_name, f"ce_only_{family}", family, 1.0, item, ce_ranked, len(candidate_ids)))
        for alpha in args.alphas:
            fused = append_remaining(fuse_rank(candidate_ids, conv_scores, ce_by_id, alpha), fallback_ids)
            rows.append(metric_row(seed, split_name, f"fusion_{family}", family, alpha, item, fused, len(candidate_ids)))
    return rows


def select_policies(summary_rows, tolerance):
    selected = []
    for seed in sorted({int(row["seed"]) for row in summary_rows}):
        dev_rows = [row for row in summary_rows if int(row["seed"]) == seed and row["split"] == "dev"]
        test_rows = [row for row in summary_rows if int(row["seed"]) == seed and row["split"] == "test"]
        dev_full = next(row for row in dev_rows if row["method"] == "convmemory_full")
        dev_bal = next(row for row in dev_rows if row["method"] == "convmemory_balanced")
        test_by_key = {(row["method"], row["family"], row["alpha"]): row for row in test_rows}
        for baseline_name, dev_base in [("full", dev_full), ("balanced", dev_bal)]:
            candidates = [
                row for row in dev_rows
                if row["method"].startswith("fusion_") or row["method"].startswith("ce_only_")
            ]
            feasible = [
                row for row in candidates
                if float(row["recall_at_10"]) >= float(dev_base["recall_at_10"]) - tolerance
            ]
            pool = feasible if feasible else candidates
            best = sorted(
                pool,
                key=lambda row: (
                    -float(row["mrr"]),
                    -float(row["recall_at_10"]),
                    float(row["avg_ce_pairs"]),
                ),
            )[0]
            test = test_by_key[(best["method"], best["family"], best["alpha"])]
            base_test = next(row for row in test_rows if row["method"] == f"convmemory_{baseline_name}")
            selected.append(
                {
                    "seed": seed,
                    "baseline": baseline_name,
                    "tolerance": tolerance,
                    "selected_method": best["method"],
                    "selected_family": best["family"],
                    "selected_alpha": best["alpha"],
                    "dev_recall": best["recall_at_10"],
                    "dev_mrr": best["mrr"],
                    "test_recall": test["recall_at_10"],
                    "test_hit": test["hit_at_10"],
                    "test_mrr": test["mrr"],
                    "test_ce_pairs": test["avg_ce_pairs"],
                    "baseline_test_recall": base_test["recall_at_10"],
                    "baseline_test_mrr": base_test["mrr"],
                    "delta_recall_vs_baseline": float(test["recall_at_10"]) - float(base_test["recall_at_10"]),
                    "delta_mrr_vs_baseline": float(test["mrr"]) - float(base_test["mrr"]),
                }
            )
    return selected


def aggregate_selected(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["baseline"], row["tolerance"]), []).append(row)
    out = []
    for (baseline, tolerance), items in sorted(groups.items()):
        out.append(
            {
                "baseline": baseline,
                "tolerance": tolerance,
                "seeds": len(items),
                "test_recall": float(np.mean([float(x["test_recall"]) for x in items])),
                "delta_recall_vs_baseline": float(np.mean([float(x["delta_recall_vs_baseline"]) for x in items])),
                "test_hit": float(np.mean([float(x["test_hit"]) for x in items])),
                "test_mrr": float(np.mean([float(x["test_mrr"]) for x in items])),
                "delta_mrr_vs_baseline": float(np.mean([float(x["delta_mrr_vs_baseline"]) for x in items])),
                "test_ce_pairs": float(np.mean([float(x["test_ce_pairs"]) for x in items])),
            }
        )
    return out


def write_report(path, selected_summary):
    lines = [
        "# v0.35 ConvMemory + Cross-Encoder Score Fusion",
        "",
        "The dev split selects a fusion family and alpha, then reports test quality.",
        "",
        "| Baseline | Tolerance | Recall@10 | Delta Recall | Hit@10 | MRR@10 | Delta MRR | CE pairs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected_summary:
        lines.append(
            f"| `{row['baseline']}` | {float(row['tolerance']):.3f} | "
            f"{row['test_recall']:.4f} | {row['delta_recall_vs_baseline']:+.4f} | "
            f"{row['test_hit']:.4f} | {row['test_mrr']:.4f} | "
            f"{row['delta_mrr_vs_baseline']:+.4f} | {row['test_ce_pairs']:.1f} |"
        )
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/locomo10.json")
    parser.add_argument("--checkpoint", default="checkpoints/convmemory-locomo-mpnet")
    parser.add_argument("--encoder-model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--embedding-cache", default=None)
    parser.add_argument("--embedding-cache-key", default=None)
    parser.add_argument("--cross-encoder-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--teacher-cache", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--dev-ratio", type=float, default=0.5)
    parser.add_argument("--raw-top-n", type=int, default=500)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--encoder-batch-size", type=int, default=128)
    parser.add_argument("--cross-batch-size", type=int, default=512)
    parser.add_argument("--cache-save-every", type=int, default=50)
    parser.add_argument("--alphas", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8")
    parser.add_argument("--out", default="results/v035/ce_fusion")
    args = parser.parse_args()
    args.alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    samples, _ = load_locomo_memory_sets(ROOT / args.data)
    examples = load_locomo_examples(ROOT / args.data)
    convmemory = ConvMemory.from_pretrained(ROOT / args.checkpoint, device=device, embedding_model=False)
    encoder = SentenceTransformerTextEncoder(
        model_name=args.encoder_model,
        device=device,
        batch_size=args.encoder_batch_size,
        cache_path=args.embedding_cache,
        cache_key=args.embedding_cache_key or args.encoder_model,
    )
    cross_encoder = CrossEncoder(resolve_local_model_path(args.cross_encoder_model), device=device)
    note_banks = {}
    encoded_notes = {}
    for sample_id, banks in samples.items():
        note_banks[sample_id] = build_note_banks(banks)
        encoded_notes[sample_id] = {
            name: encode_memory_bank(encoder, notes)
            for name, notes in note_banks[sample_id].items()
        }

    cache = load_cache(ROOT / args.teacher_cache) if args.teacher_cache else {}
    cache_path = ROOT / args.teacher_cache if args.teacher_cache else None
    counter = {"questions": 0, "misses": 0}
    rows = []
    for seed in args.seeds:
        for split_name in ["dev", "test"]:
            split_examples = choose_split(examples, split_name, args.dev_ratio, seed)
            if args.limit:
                split_examples = split_examples[: args.limit]
            for idx, example in enumerate(split_examples, start=1):
                rows.extend(
                    evaluate_example(
                        seed,
                        split_name,
                        example,
                        samples,
                        note_banks,
                        encoded_notes,
                        encoder,
                        convmemory,
                        cross_encoder,
                        args,
                        cache,
                        counter,
                    )
                )
                if idx % 100 == 0 or idx == len(split_examples):
                    print(
                        f"seed {seed} {split_name}: {idx}/{len(split_examples)} "
                        f"cache_misses={counter['misses']}",
                        flush=True,
                    )
                    if cache_path:
                        save_cache(cache_path, cache)

    if cache_path:
        save_cache(cache_path, cache)
    summary = summarize(rows)
    selected = []
    for tolerance in [0.0, 0.0025, 0.005, 0.01]:
        selected.extend(select_policies(summary, tolerance))
    selected_summary = aggregate_selected(selected)
    write_csv(out_dir / "detailed.csv", rows)
    write_csv(out_dir / "summary.csv", summary)
    write_csv(out_dir / "selected_by_seed.csv", selected)
    write_csv(out_dir / "selected_summary.csv", selected_summary)
    write_report(out_dir / "REPORT.md", selected_summary)


if __name__ == "__main__":
    main()

import argparse
import csv
import json
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
from experiments.support.locomo_notes import encode_memory_bank, load_locomo_memory_sets, simple_token_count
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


def add_halo(candidate_indices, num_items, radius):
    if radius <= 0:
        return np.asarray(candidate_indices, dtype=np.int64)
    selected = []
    seen = set()
    for idx in candidate_indices:
        center = int(idx)
        for new_idx in range(max(0, center - radius), min(num_items, center + radius + 1)):
            if new_idx in seen:
                continue
            selected.append(new_idx)
            seen.add(new_idx)
    return np.asarray(selected, dtype=np.int64)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os_getpid()}.tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(path)


def os_getpid():
    try:
        import os

        return os.getpid()
    except Exception:
        return 0


def token_count(items):
    return sum(simple_token_count(item.get("text", "")) for item in items)


def candidate_tokens(memories, indices):
    return token_count([memories[int(idx)] for idx in indices])


def ranked_ids(results):
    return [result.memory_id for result in results]


def ids_from_indices(memory_ids, indices):
    return [memory_ids[int(idx)] for idx in indices]


def append_remaining(primary_ids, fallback_ids):
    seen = set(primary_ids)
    return list(primary_ids) + [memory_id for memory_id in fallback_ids if memory_id not in seen]


def rank_by_scores(indices, score_map):
    return sorted([int(idx) for idx in indices], key=lambda idx: -float(score_map.get(str(idx), -1e9)))


def metric_row(seed, split_name, method, item, ranked, ce_pairs, pool_count, pool_tokens, score_time):
    return {
        "seed": seed,
        "split": split_name,
        "method": method,
        "question_id": item["question_id"],
        "question_type": item["question_type"],
        "gold_count": len(item["gold_memory_ids"]),
        "ce_pairs": int(ce_pairs),
        "pool_count": int(pool_count),
        "pool_tokens": int(pool_tokens),
        "ce_score_time_ms": float(score_time) * 1000.0,
        "recall_at_10": recall_at_k(ranked, item["gold_memory_ids"], 10),
        "hit_at_10": hit_at_k(ranked, item["gold_memory_ids"], 10),
        "mrr": mrr(ranked[:10], item["gold_memory_ids"]),
    }


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["seed"], row["split"], row["method"]), []).append(row)
    by_seed = []
    for (seed, split_name, method), items in sorted(groups.items()):
        by_seed.append(
            {
                "seed": seed,
                "split": split_name,
                "method": method,
                "questions": len(items),
                "avg_ce_pairs": float(np.mean([float(x["ce_pairs"]) for x in items])),
                "avg_pool_count": float(np.mean([float(x["pool_count"]) for x in items])),
                "avg_pool_tokens": float(np.mean([float(x["pool_tokens"]) for x in items])),
                "avg_ce_score_time_ms": float(np.mean([float(x["ce_score_time_ms"]) for x in items])),
                "recall_at_10": float(np.mean([float(x["recall_at_10"]) for x in items])),
                "hit_at_10": float(np.mean([float(x["hit_at_10"]) for x in items])),
                "mrr": float(np.mean([float(x["mrr"]) for x in items])),
            }
        )

    test_rows = [row for row in by_seed if row["split"] == "test"]
    mean_groups = {}
    for row in test_rows:
        mean_groups.setdefault(row["method"], []).append(row)
    mean_rows = []
    base = None
    for method, items in sorted(mean_groups.items()):
        row = {
            "method": method,
            "seeds": len(items),
            "questions": int(np.sum([int(x["questions"]) for x in items])),
            "avg_ce_pairs": float(np.mean([float(x["avg_ce_pairs"]) for x in items])),
            "avg_pool_count": float(np.mean([float(x["avg_pool_count"]) for x in items])),
            "avg_pool_tokens": float(np.mean([float(x["avg_pool_tokens"]) for x in items])),
            "avg_ce_score_time_ms": float(np.mean([float(x["avg_ce_score_time_ms"]) for x in items])),
            "recall_at_10": float(np.mean([float(x["recall_at_10"]) for x in items])),
            "hit_at_10": float(np.mean([float(x["hit_at_10"]) for x in items])),
            "mrr": float(np.mean([float(x["mrr"]) for x in items])),
        }
        mean_rows.append(row)
        if method == "ce_raw_top500":
            base = row
    if base is not None:
        for row in mean_rows:
            row["delta_recall_vs_ce_raw_top500"] = row["recall_at_10"] - base["recall_at_10"]
            row["delta_mrr_vs_ce_raw_top500"] = row["mrr"] - base["mrr"]
            row["ce_pair_ratio_vs_top500"] = row["avg_ce_pairs"] / max(1e-9, base["avg_ce_pairs"])
    return by_seed, sorted(mean_rows, key=lambda row: (-row["mrr"], -row["recall_at_10"], row["avg_ce_pairs"]))


def score_union(item, candidate_indices, cross_encoder, batch_size, cache, cache_path, save_every, counter):
    qid = item["question_id"]
    qcache = cache.setdefault(qid, {})
    missing = [int(idx) for idx in candidate_indices if str(int(idx)) not in qcache]
    elapsed = 0.0
    if missing:
        memories = item["memories"]
        pairs = [(item["query"], memories[int(idx)]["text"]) for idx in missing]
        start = time.perf_counter()
        scores = cross_encoder.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        elapsed = time.perf_counter() - start
        for idx, score in zip(missing, scores):
            qcache[str(int(idx))] = float(score)
        counter["misses"] += len(missing)
        if cache_path and counter["questions"] % save_every == 0:
            save_json(cache_path, cache)
    return qcache, elapsed, len(missing)


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


def evaluate_example(seed, split_name, example, samples, note_banks, encoded_notes, encoder, convmemory, cross_encoder, args, cache, counter):
    sample_id = example["question_id"].split("::", 1)[0]
    item = prepare_encoded_example(example, encoder, convmemory.config.window_size, convmemory.config.stride)
    memory_ids = item["memory_ids"]
    raw_scores = item["memory_embeddings"] @ item["query_embedding"]
    raw_order = np.argsort(-raw_scores)
    raw_ranked = ids_from_indices(memory_ids, raw_order)

    full_top = raw_order[: min(args.raw_top_n, len(raw_order))]
    raw100 = raw_order[: min(100, len(raw_order))]
    raw300 = raw_order[: min(300, len(raw_order))]
    fast = route_indices("fast", item, sample_id, note_banks, encoded_notes)
    balanced = route_indices("balanced", item, sample_id, note_banks, encoded_notes)
    balanced_halo1 = add_halo(balanced, len(memory_ids), 1)

    conv_full = convmemory.reranker.rerank_item(
        item,
        candidate_top_n=args.raw_top_n,
        window_mode="candidate_local",
    )
    conv_balanced = convmemory.reranker.rerank_item(
        item,
        candidate_indices=balanced,
        window_mode="candidate_local",
    )
    id_to_idx = {memory_id: idx for idx, memory_id in enumerate(memory_ids)}
    conv_full_top50 = np.asarray([id_to_idx[memory_id] for memory_id in ranked_ids(conv_full)[:50]], dtype=np.int64)
    conv_full_top100 = np.asarray([id_to_idx[memory_id] for memory_id in ranked_ids(conv_full)[:100]], dtype=np.int64)
    conv_bal_top50 = np.asarray([id_to_idx[memory_id] for memory_id in ranked_ids(conv_balanced)[:50]], dtype=np.int64)
    conv_bal_top100 = np.asarray([id_to_idx[memory_id] for memory_id in ranked_ids(conv_balanced)[:100]], dtype=np.int64)

    union = []
    seen = set()
    for group in [full_top, fast, balanced, balanced_halo1, conv_full_top100, conv_bal_top100]:
        for idx in group:
            idx = int(idx)
            if idx not in seen:
                seen.add(idx)
                union.append(idx)
    counter["questions"] += 1
    score_map, elapsed, misses = score_union(
        item,
        union,
        cross_encoder,
        args.cross_batch_size,
        cache,
        Path(args.teacher_cache) if args.teacher_cache else None,
        args.cache_save_every,
        counter,
    )

    rows = []
    rows.append(
        metric_row(
            seed,
            split_name,
            "raw_vector",
            item,
            raw_ranked,
            0,
            len(memory_ids),
            candidate_tokens(item["memories"], np.arange(len(memory_ids))),
            0.0,
        )
    )

    conv_full_ids = ranked_ids(conv_full)
    rows.append(
        metric_row(
            seed,
            split_name,
            f"convmemory_raw_top{args.raw_top_n}",
            item,
            conv_full_ids,
            0,
            len(full_top),
            candidate_tokens(item["memories"], full_top),
            0.0,
        )
    )
    conv_balanced_ids = ranked_ids(conv_balanced)
    rows.append(
        metric_row(
            seed,
            split_name,
            "convmemory_balanced_halo0",
            item,
            conv_balanced_ids,
            0,
            len(balanced),
            candidate_tokens(item["memories"], balanced),
            0.0,
        )
    )

    ce_sets = {
        "ce_raw_top100": raw100,
        "ce_raw_top300": raw300,
        f"ce_raw_top{args.raw_top_n}": full_top,
        "ce_fast_halo0": fast,
        "ce_balanced_halo0": balanced,
        "ce_balanced_halo1": balanced_halo1,
        "ce_convmemory_full_top50": conv_full_top50,
        "ce_convmemory_full_top100": conv_full_top100,
        "ce_convmemory_balanced_top50": conv_bal_top50,
        "ce_convmemory_balanced_top100": conv_bal_top100,
    }
    for method, indices in ce_sets.items():
        ordered = rank_by_scores(indices, score_map)
        ranked = ids_from_indices(memory_ids, ordered)
        if method.startswith("ce_convmemory_full"):
            ranked = append_remaining(ranked, conv_full_ids)
        elif method.startswith("ce_convmemory_balanced"):
            ranked = append_remaining(ranked, conv_balanced_ids)
        else:
            ranked = append_remaining(ranked, raw_ranked)
        rows.append(
            metric_row(
                seed,
                split_name,
                method,
                item,
                ranked,
                len(indices),
                len(indices),
                candidate_tokens(item["memories"], indices),
                elapsed * (len(indices) / max(1, len(union))) if misses else 0.0,
            )
        )
    return rows


def write_report(path, mean_rows):
    focus = [
        "ce_raw_top500",
        "ce_raw_top300",
        "ce_balanced_halo0",
        "ce_fast_halo0",
        "ce_convmemory_full_top50",
        "ce_convmemory_balanced_top50",
        "convmemory_balanced_halo0",
    ]
    by_method = {row["method"]: row for row in mean_rows}
    lines = [
        "# v0.34 Cross-Encoder Cascade Candidate Pools",
        "",
        "Question: can ConvMemory/compression routing reduce the number of candidates a cross-encoder must score while preserving quality?",
        "",
        "| Method | Recall@10 | Delta Recall | Hit@10 | MRR@10 | CE pairs | Pair ratio |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in focus:
        row = by_method.get(method)
        if not row:
            continue
        lines.append(
            f"| `{method}` | {row['recall_at_10']:.4f} | "
            f"{row.get('delta_recall_vs_ce_raw_top500', 0.0):+.4f} | "
            f"{row['hit_at_10']:.4f} | {row['mrr']:.4f} | "
            f"{row['avg_ce_pairs']:.1f} | {row.get('ce_pair_ratio_vs_top500', 0.0):.3f} |"
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
    parser.add_argument("--splits", default="test")
    parser.add_argument("--dev-ratio", type=float, default=0.5)
    parser.add_argument("--raw-top-n", type=int, default=500)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--encoder-batch-size", type=int, default=128)
    parser.add_argument("--cross-batch-size", type=int, default=256)
    parser.add_argument("--cache-save-every", type=int, default=50)
    parser.add_argument("--out", default="results/v034/ce_cascade_candidate_pools")
    args = parser.parse_args()

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

    cache = load_json(ROOT / args.teacher_cache) if args.teacher_cache else {}
    counter = {"questions": 0, "misses": 0}
    rows = []
    split_names = [x.strip() for x in args.splits.split(",") if x.strip()]
    for seed in args.seeds:
        for split_name in split_names:
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
                if idx % 50 == 0 or idx == len(split_examples):
                    print(
                        f"seed {seed} {split_name}: {idx}/{len(split_examples)} "
                        f"cache_misses={counter['misses']}",
                        flush=True,
                    )
                    if args.teacher_cache:
                        save_json(ROOT / args.teacher_cache, cache)

    if args.teacher_cache:
        save_json(ROOT / args.teacher_cache, cache)
    by_seed, mean_rows = summarize(rows)
    write_csv(out_dir / "detailed.csv", rows)
    write_csv(out_dir / "summary_by_seed.csv", by_seed)
    write_csv(out_dir / "summary.csv", mean_rows)
    write_report(out_dir / "REPORT.md", mean_rows)


if __name__ == "__main__":
    main()

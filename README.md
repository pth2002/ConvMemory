# ConvMemory

[![CI](https://github.com/pth2002/ConvMemory/actions/workflows/ci.yml/badge.svg)](https://github.com/pth2002/ConvMemory/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

ConvMemory is a lightweight temporal reranker for long-term conversational and
agent memory.

It is designed to sit after vector search and before prompt construction:

```text
user query -> vector search top-k -> ConvMemory -> selected memory context
```

ConvMemory models local temporal structure in ordered memories, such as
multi-session conversations, user histories, agent traces, and event logs. It
can be used as a standalone memory reranker, a context expander, or a candidate
stage before a smaller cross-encoder pass.

Current version: `v0.3.0`

## Highlights

- Temporal memory reranking over text or precomputed embeddings.
- Drop-in API for agent memory pipelines: `rerank`, `retrieve`, and `expand`.
- Candidate-local scoring for large memory pools.
- Compression-aware routing utilities for session notes or block summaries.
- Reproducible retrieval-stage evaluation scripts for LoCoMo and LongMemEval-S.
- Research-preview cascade experiments with a small cross-encoder pass.

## Installation

From source:

```bash
git clone https://github.com/pth2002/ConvMemory.git
cd ConvMemory
pip install -e .
```

Install the basic dependencies:

```bash
pip install -r requirements.txt
```

ConvMemory requires Python 3.10 or later.

## Quick Start

```python
from convmemory import ConvMemory

model = ConvMemory.from_pretrained(
    "checkpoints/convmemory-locomo-mpnet",
    device="cuda",
)

memories = [
    {"id": "m1", "text": "The user said their hiking trip moved to Sunday."},
    {"id": "m2", "text": "The assistant recommended bringing extra water."},
    {"id": "m3", "text": "The user has an exam next Friday."},
]

results = model.rerank(
    query="When is the hiking trip?",
    memories=memories,
    top_k=2,
)

for item in results:
    print(item.rank, item.memory_id, item.score, item.text)
```

Memory order matters. Pass memories in chronological order whenever that order
is available.

## Checkpoint

The public LoCoMo MPNet checkpoint is distributed as a GitHub release asset:

[Download `convmemory-locomo-mpnet.zip`](https://github.com/pth2002/ConvMemory/releases/download/v0.1.0/convmemory-locomo-mpnet.zip)

Extract it from the repository root:

```bash
mkdir -p checkpoints
unzip convmemory-locomo-mpnet.zip -d checkpoints
```

On Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force -Path checkpoints
Expand-Archive .\convmemory-locomo-mpnet.zip -DestinationPath .\checkpoints -Force
```

Expected layout:

```text
checkpoints/convmemory-locomo-mpnet/
  config.json
  model.pt
```

Verify the checkpoint:

```bash
python examples/load_pretrained.py
```

The same checkpoint is used for v0.1, v0.2, and v0.3. The v0.3 release adds
routing utilities and cascade-fusion experiments; it does not require a new
ConvMemory weight file.

See [docs/MODEL_CARD.md](docs/MODEL_CARD.md) for checkpoint details and known
limitations.

## Core API

### Rerank

Use `rerank` when you want the strongest ordered top-k memories.

```python
ranked = model.rerank(
    query="What changed about the meeting plan?",
    memories=memories,
    top_k=10,
)
```

### Retrieve

Use `retrieve` when integrating ConvMemory into an application-level memory
pipeline.

```python
ranked = model.retrieve(
    query="What changed about the meeting plan?",
    memories=memories,
    mode="rerank",
    top_k=10,
)
```

### Expand Context

Use `mode="expand"` when the downstream agent can read a slightly wider context
and missing evidence is more costly than including a few additional candidates.

```python
context = model.retrieve(
    query="What changed about the meeting plan?",
    memories=memories,
    mode="expand",
    protected_k=10,
    top_k=15,
)
```

The first `protected_k` results come from the ConvMemory ranking. The remaining
slots are filled from complementary rankings such as raw dense retrieval and
candidate-local temporal scoring.

### Precomputed Embeddings

Production memory systems often already store memory embeddings. Use
`rerank_embeddings` to avoid re-encoding the memory store.

```python
ranked = model.rerank_embeddings(
    query_embedding=query_embedding,
    memory_embeddings=memory_embeddings,
    memory_ids=memory_ids,
    memory_texts=memory_texts,
    candidate_indices=candidate_indices,
    query=query,
    top_k=20,
)
```

The checkpoint and embeddings must use the same embedding model family and
embedding dimension.

## Agent Memory Integration

ConvMemory is usually called after vector search:

```python
from convmemory import ConvMemory

memory_reranker = ConvMemory.from_pretrained(
    "checkpoints/convmemory-locomo-mpnet",
    device="cuda",
)

def retrieve_agent_memory(query, memory_store, top_k=15):
    candidates = memory_store.vector_search(query, top_k=500)

    ranked = memory_reranker.retrieve(
        query=query,
        memories=candidates,
        mode="expand",
        protected_k=10,
        top_k=top_k,
    )

    return [
        {"id": item.memory_id, "text": item.text, "score": item.score}
        for item in ranked
    ]
```

If your vector database returns ids and the full memory stream is stored
elsewhere, pass `candidate_ids`:

```python
memory_reranker.prewarm_lexical(all_user_memories)

candidate_ids = [
    item["id"]
    for item in memory_store.vector_search(query, top_k=500)
]

ranked = memory_reranker.rerank(
    query=query,
    memories=all_user_memories,
    candidate_ids=candidate_ids,
    top_k=20,
    window_mode="candidate_local",
)
```

## Compression-Aware Routing

v0.3 adds utilities for routing from compressed notes back to raw memories. This
is useful when the full memory store is large, but the system maintains
session-level notes, summaries, or fixed-size memory blocks.

```python
from convmemory import (
    CompressedNoteConfig,
    CompressionRouteConfig,
    CompressionRouter,
    build_compressed_notes,
)

notes = build_compressed_notes(
    memories=all_user_memories,
    memory_embeddings=memory_embeddings,
    config=CompressedNoteConfig(mode="session", representatives=3),
)

note_embeddings = embedding_model.encode(
    [note["text"] for note in notes],
    normalize_embeddings=True,
)

router = CompressionRouter(
    CompressionRouteConfig(
        note_depth=240,
        max_sources_per_note=5,
        max_candidates=450,
        raw_anchor=80,
    )
)

route = router.route(
    query_embedding=query_embedding,
    memory_embeddings=memory_embeddings,
    memory_ids=memory_ids,
    compressed_embeddings=note_embeddings,
    compressed_memories=notes,
)

ranked = model.rerank_embeddings(
    query_embedding=query_embedding,
    memory_embeddings=memory_embeddings,
    memory_ids=memory_ids,
    memory_texts=memory_texts,
    candidate_indices=route.candidate_indices,
    query=query,
    top_k=20,
)
```

The router is deliberately separate from the neural reranker. It only returns
candidate ids or indices, so it can be used with existing memory stores.

## Preliminary Benchmarks

These numbers are intended to make the current checkpoint inspectable, not to
claim state-of-the-art reranking.

Scope:

- All numbers are retrieval-stage evaluations, not end-to-end answer generation.
- The public checkpoint is trained on LoCoMo, so LoCoMo results are in-domain.
- LongMemEval-S is an out-of-domain check within the same long-memory task family.
- Reported cross-encoder baselines use `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Latency measures online reranking after memory embeddings and memory-side
  indexes are available.
- Statistical confidence intervals, stronger rerankers, more embedding
  backbones, and full ablations are still future work.

### LoCoMo

In-domain LoCoMo test split, MPNet embeddings, top500 candidate pool.

| Method | Questions | Recall@10 | Hit@10 | MRR |
|---|---:|---:|---:|---:|
| Raw MPNet retrieval | 937 | 0.553 | 0.607 | 0.334 |
| Cross-encoder top500 | 937 | 0.749 | 0.790 | 0.614 |
| ConvMemory top500 | 937 | 0.768 | 0.821 | 0.554 |

In this setting, ConvMemory improves recall-oriented memory selection over raw
dense retrieval. The tested cross-encoder remains stronger at top-rank
precision, reflected by MRR. This table should not be read as a general claim
that ConvMemory outperforms cross-encoders.

### LongMemEval-S OOD Check

LoCoMo-trained ConvMemory checkpoint evaluated on LongMemEval-S without
LongMemEval training. This is a same-family out-of-domain retrieval check, not
a broad generalization claim.

| Method | Questions | Recall@5 | Hit@5 | Recall@10 | Hit@10 | MRR | ms/query |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw MPNet retrieval | 500 | 0.823 | 0.902 | 0.905 | 0.946 | 0.783 | 0.0 |
| Cross-encoder top500 | 500 | 0.882 | 0.934 | 0.933 | 0.956 | 0.890 | 115.6 |
| ConvMemory LoCoMo checkpoint | 500 | 0.920 | 0.968 | 0.959 | 0.988 | 0.897 | 48.7 |

### Large-Pool Stress Test

LongMemEval-S with distractor sessions added to each question memory pool.

| Memory pool | Raw R@10 / MRR | Cross-encoder R@10 / MRR / ms | ConvMemory R@10 / MRR / ms |
|---|---:|---:|---:|
| Original, avg 48 sessions | 0.905 / 0.783 | 0.933 / 0.890 / 115.6 | 0.959 / 0.897 / 48.7 |
| 500 sessions | 0.641 / 0.524 | 0.791 / 0.730 / 528.3 | 0.809 / 0.711 / 357.8 |
| 1000 sessions, top500 rerank | 0.548 / 0.442 | 0.730 / 0.671 / 519.9 | 0.739 / 0.623 / 862.9 |
| 1000 sessions, candidate-local + cached lexical | 0.548 / 0.442 | 0.730 / 0.671 / 519.9 | 0.729 / 0.618 / 89.1 |

The optimized 1000-session setting uses candidate-local temporal windows and
cached lexical signatures. Memory-side lexical signatures are built once when
the memory store is indexed, similar to cached embeddings.

The latency comparison in this stress test should be treated as an engineering
measurement for this implementation, not as an apples-to-apples benchmark
against optimized production cross-encoder serving.

## Cascade Fusion Preview

The current cascade experiment uses ConvMemory as a candidate stage before a
small cross-encoder pass:

```text
vector top500 -> ConvMemory top100 -> cross-encoder -> normalized score fusion
```

The goal is cost-quality tradeoff: use ConvMemory to build a smaller candidate
set, then apply a cross-encoder only to that reduced set.

LoCoMo test splits, seeds 7/11/23, MPNet embeddings,
`cross-encoder/ms-marco-MiniLM-L-6-v2`.

| Method | Recall@10 | Hit@10 | MRR@10 | CE pairs/query |
|---|---:|---:|---:|---:|
| Cross-encoder raw top500 | 0.7371 | 0.7842 | 0.5978 | 483.8 |
| ConvMemory balanced | 0.7798 | 0.8361 | 0.5698 | 0.0 |
| ConvMemory + CE fusion top100 | 0.7908 | 0.8425 | 0.6124 | 100.0 |

`ConvMemory balanced` refers to the current balanced candidate-routing preset:
raw dense anchors plus compressed-note routing, followed by ConvMemory reranking
and no cross-encoder scoring.

RTX 4090 latency benchmark, 600 LoCoMo test queries, 10 warmup queries per seed.

| Method | Recall@10 | Hit@10 | MRR@10 | Mean ms/query | P95 ms/query | Speedup vs CE top500 |
|---|---:|---:|---:|---:|---:|---:|
| ConvMemory balanced | 0.7408 | 0.8083 | 0.5214 | 10.2 | 13.3 | 10.26x |
| Cross-encoder raw top500 | 0.7290 | 0.7750 | 0.5813 | 104.5 | 125.4 | 1.00x |
| ConvMemory + CE fusion top100 | 0.7487 | 0.8033 | 0.5887 | 38.2 | 43.9 | 2.74x |

Cascade fusion is not yet a stable public API. The scripts are included so the
results can be inspected and reproduced before the interface is finalized. The
current results should be interpreted as evidence that ConvMemory can provide a
useful candidate stage for a small cross-encoder pass, not as proof that the
model is a universal cross-encoder replacement.

## Reproducibility

For the full evaluation protocol, including simple baselines, feature ablations,
multi-seed reporting, order robustness, and calibration checks, see
[docs/EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md).

Main LoCoMo evaluation:

```bash
python experiments/reproduce_locomo.py \
  --device cuda \
  --encoder-model sentence-transformers/all-mpnet-base-v2 \
  --embedding-cache results/cache/mpnet_embeddings.sqlite \
  --embedding-cache-key sentence-transformers/all-mpnet-base-v2 \
  --teacher-cache results/cache/repro_teacher_mpnet_top500_seed23.json \
  --raw-top-n 500 \
  --candidate-top-n 500 \
  --cross-top-n 500 \
  --eval-cross-encoder \
  --out results/reproduce_mpnet_full_top500_seed23
```

Cascade-fusion preview:

```bash
python experiments/v035_ce_fusion.py \
  --device cuda \
  --checkpoint checkpoints/convmemory-locomo-mpnet \
  --encoder-model sentence-transformers/all-mpnet-base-v2 \
  --embedding-cache results/cache/mpnet_embeddings.sqlite \
  --embedding-cache-key sentence-transformers/all-mpnet-base-v2 \
  --cross-encoder-model cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --seeds 7 11 23 \
  --out results/v035/ce_fusion
```

Latency benchmark:

```bash
python experiments/v036_latency_benchmark.py \
  --device cuda \
  --checkpoint checkpoints/convmemory-locomo-mpnet \
  --encoder-model sentence-transformers/all-mpnet-base-v2 \
  --embedding-cache results/cache/mpnet_embeddings.sqlite \
  --embedding-cache-key sentence-transformers/all-mpnet-base-v2 \
  --cross-encoder-model cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --seeds 7 11 23 \
  --limit 200 \
  --warmup 10 \
  --out results/v036/latency_benchmark
```

LongMemEval-S OOD evaluation:

```bash
python experiments/evaluate_longmemeval_zero_shot.py \
  --device cuda \
  --checkpoint checkpoints/convmemory-locomo-mpnet \
  --encoder-model sentence-transformers/all-mpnet-base-v2 \
  --embedding-cache results/cache/longmemeval_mpnet_embeddings.sqlite \
  --embedding-cache-key sentence-transformers/all-mpnet-base-v2 \
  --candidate-top-n 500 \
  --eval-cross-encoder \
  --out results/longmemeval_zero_shot_full_gpu
```

For the large-pool optimized path, add:

```bash
  --distractor-sessions 1000 \
  --window-mode candidate_local \
  --precache-lexical
```

Baselines, feature ablations, and paired bootstrap statistics:

```bash
python experiments/v040_baselines_ablation_stats.py \
  --device cuda \
  --seeds 7 11 23 31 47 \
  --bootstrap-samples 10000 \
  --out results/v040/baselines_ablation_stats
```

Order robustness:

```bash
python experiments/v041_order_robustness.py \
  --device cuda \
  --seeds 7 11 23 31 47 \
  --out results/v041/order_robustness
```

Error analysis and calibration:

```bash
python experiments/v042_error_calibration.py \
  --device cuda \
  --seeds 7 11 23 31 47 \
  --out results/v042/error_calibration
```

Generic converted-dataset evaluation:

```bash
python experiments/v043_generic_retrieval_eval.py \
  --dataset-name my_dataset \
  --jsonl data/my_dataset_memory_eval.jsonl \
  --device cuda \
  --out results/v043/generic_retrieval_eval
```

Training and checkpoint export are documented in
[docs/TRAINING.md](docs/TRAINING.md).

## Project Status

ConvMemory is an early research library. The stable public surface in v0.3 is:

- `ConvMemory.from_pretrained`
- `ConvMemory.rerank`
- `ConvMemory.retrieve`
- `ConvMemory.expand_context`
- `ConvMemory.rerank_embeddings`
- `CompressionRouter`
- `build_compressed_notes`

The cascade-fusion path is currently provided as reproducible research code in
`experiments/`. A dedicated public cascade API is planned for a later release.

Important open items before making stronger research claims:

- report mean/std and paired significance tests across more seeds;
- evaluate stronger rerankers and additional embedding backbones;
- add simple baselines such as BM25, recency-weighted dense retrieval, and
  dense-lexical fusion;
- publish ablations for temporal windows, lexical features, routing, raw-score
  fusion, and candidate-local scoring;
- add a training script, model card, and training-data details;
- test robustness to missing or noisy memory order;
- add qualitative error analysis and calibration checks.

## License

MIT

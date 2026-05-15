# ConvMemory

[![CI](https://github.com/pth2002/ConvMemory/actions/workflows/ci.yml/badge.svg)](https://github.com/pth2002/ConvMemory/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

ConvMemory is a lightweight temporal memory reranker for long-term
conversational and agent memory.

It is designed to run after vector search and before prompt construction:

```text
user query -> vector search top-k -> ConvMemory -> memory context
```

The model scores ordered memory candidates with local temporal windows,
query-memory interaction features, lexical anchors, and dense retrieval scores.
It is intended for systems where memories form an event stream: conversations,
user histories, agent traces, task logs, or session-level notes.

Current package version: `0.3.0`

## When To Use It

Use ConvMemory when:

- your memory store has chronological or session structure;
- raw vector search misses important neighboring evidence;
- you need a cheaper recall-oriented stage before a full cross-encoder pass;
- the downstream agent can benefit from a compact, reranked memory context.

Do not use ConvMemory as:

- a vector database;
- a general web/document reranker without temporal structure;
- an end-to-end QA model;
- a universal replacement for cross-encoders.

For maximum top-rank precision, the strongest current path is a cascade:

```text
vector top500 -> ConvMemory top100 -> small cross-encoder -> fused ranking
```

## Installation

```bash
git clone https://github.com/pth2002/ConvMemory.git
cd ConvMemory
pip install -e .
pip install -r requirements.txt
```

ConvMemory requires Python 3.10 or later.

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

The same checkpoint is used by the current package. Newer package versions add
library and evaluation utilities; they do not require a new weight file.

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

Pass memories in chronological order when that order is available.

## Agent Memory Integration

Most applications call ConvMemory after vector search:

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
        mode="rerank",
        top_k=top_k,
    )

    return [
        {"id": item.memory_id, "text": item.text, "score": item.score}
        for item in ranked
    ]
```

If the downstream agent can read a slightly wider context, use `expand` mode.
It preserves the strongest ConvMemory prefix and fills the remaining budget
with complementary candidates:

```python
context = memory_reranker.retrieve(
    query=query,
    memories=candidates,
    mode="expand",
    protected_k=10,
    top_k=15,
)
```

For systems that already store embeddings, use `rerank_embeddings` to avoid
re-encoding the memory store:

```python
ranked = memory_reranker.rerank_embeddings(
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

## Compression-Aware Routing

ConvMemory also includes optional routing utilities for large memory stores
that maintain session notes, summaries, or fixed-size memory blocks:

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

ranked = memory_reranker.rerank_embeddings(
    query_embedding=query_embedding,
    memory_embeddings=memory_embeddings,
    memory_ids=memory_ids,
    memory_texts=memory_texts,
    candidate_indices=route.candidate_indices,
    query=query,
    top_k=20,
)
```

The router is separate from the neural reranker. It returns candidate ids or
indices and can be used with existing memory stores.

## Results

These are retrieval-stage evaluations. They measure whether annotated evidence
memories are retrieved into the top-k list; they do not measure final answer
generation.

Important scope notes:

- The public checkpoint is trained on LoCoMo-style data; LoCoMo is an in-domain
  evaluation for this checkpoint.
- The cross-encoder baseline reported here is
  `cross-encoder/ms-marco-MiniLM-L-6-v2`, a small reranker baseline.
- Latency numbers are measured after memory embeddings and memory-side indexes
  are available. Cross-encoder timing includes `CrossEncoder.predict`
  tokenization.
- Results should be read as evidence for a lightweight memory module, not as a
  state-of-the-art reranking claim.

### LoCoMo Memory Retrieval

Five split seeds, MPNet embeddings, top500 candidate pool.

| Method | Recall@10 | Hit@10 | MRR |
|---|---:|---:|---:|
| Raw dense retrieval | 0.5345 +/- 0.0210 | 0.5894 | 0.3254 +/- 0.0105 |
| BM25 | 0.5889 | 0.6361 | 0.4305 |
| Dense + lexical + temporal RRF | 0.6375 | 0.6910 | 0.3997 |
| Dense + lexical score fusion | 0.6579 | 0.7148 | 0.4682 |
| MiniLM cross-encoder top500 | 0.7294 +/- 0.0151 | 0.7765 | 0.5968 +/- 0.0132 |
| ConvMemory | 0.7798 +/- 0.0074 | 0.8350 | 0.5824 +/- 0.0189 |

Paired bootstrap over 4,955 test questions:

- ConvMemory vs raw dense: +0.2465 Recall@10, 95% CI [+0.2338, +0.2594], p < 0.001.
- ConvMemory vs MiniLM cross-encoder top500: +0.0508 Recall@10, 95% CI [+0.0407, +0.0609], p < 0.001.
- ConvMemory vs MiniLM cross-encoder top500: -0.0139 MRR, 95% CI [-0.0238, -0.0043], p = 0.0038.

Interpretation: ConvMemory is stronger on recall coverage in this memory
setting, while the tested cross-encoder keeps a small but significant advantage
in top-rank precision.

### Feature Ablations

Inference-time feature masking over five seeds. These are diagnostic ablations,
not separately retrained checkpoints.

| Variant | Recall@10 | Delta vs full | MRR |
|---|---:|---:|---:|
| Full ConvMemory | 0.7798 | +0.0000 | 0.5824 |
| No temporal-window feature | 0.7317 | -0.0481 | 0.5556 |
| No lexical features | 0.6978 | -0.0821 | 0.4790 |
| No raw dense feature | 0.7650 | -0.0148 | 0.5707 |
| No router feature | 0.7810 | +0.0012 | 0.5832 |
| Temporal-window only | 0.5604 | -0.2195 | 0.2776 |

The current checkpoint depends on a combination of dense, lexical, and temporal
signals. The router feature is not a strong standalone contributor in this
configuration and should be treated as auxiliary.

### Cross-Encoder And Latency

RTX 4090, three seeds, 200 measured queries per seed, 10 warmup queries per
seed.

`ConvMemory balanced` refers to the built-in compression-routing preset with no
cross-encoder scoring.

| Method | Recall@10 | Hit@10 | MRR | Mean ms/query | P95 ms/query | Speedup vs CE top500 |
|---|---:|---:|---:|---:|---:|---:|
| Raw vector search | 0.5253 | 0.5867 | 0.3112 | 0.9 | 5.7 | 158.21x |
| ConvMemory balanced | 0.7408 | 0.8083 | 0.5214 | 16.4 | 57.7 | 8.20x |
| ConvMemory full top500 | 0.7342 | 0.7967 | 0.5222 | 21.1 | 71.4 | 6.37x |
| Cross-encoder raw top500 | 0.7290 | 0.7750 | 0.5813 | 134.5 | 245.3 | 1.00x |
| Cross-encoder raw top100 | 0.6633 | 0.7167 | 0.5343 | 22.1 | 24.8 | 6.08x |
| ConvMemory + CE fusion top100 | 0.7487 | 0.8033 | 0.5887 | 44.0 | 94.9 | 3.05x |

Interpretation: ConvMemory is competitive on Recall@10 at lower latency, while
the cross-encoder remains stronger at top-rank precision. The best current
cost-quality tradeoff is using ConvMemory to build a smaller candidate pool for
a small cross-encoder pass.

### Order Robustness

Five seeds, LoCoMo memory order perturbations.

| Memory order | ConvMemory Recall@10 | Delta vs original | MRR |
|---|---:|---:|---:|
| Original | 0.7798 | +0.0000 | 0.5824 |
| Block shuffle | 0.7740 | -0.0058 | 0.5762 |
| Shuffle 10% | 0.7681 | -0.0117 | 0.5781 |
| Shuffle 50% | 0.7351 | -0.0448 | 0.5450 |
| Shuffle 100% | 0.7158 | -0.0641 | 0.5362 |
| Reverse | 0.7855 | +0.0056 | 0.5855 |

This suggests ConvMemory benefits from local memory-neighborhood coherence, not
from a fragile assumption that absolute chronological order must always be
perfect.

If timestamps are unreliable, use ConvMemory more conservatively:

- prefer `mode="expand"` so the downstream agent receives a slightly wider
  context;
- keep raw dense candidates in the final context budget;
- avoid interpreting scores as calibrated confidence when memory order is known
  to be corrupted;
- consider disabling temporal features or retraining with order noise before
  using ConvMemory as a strict top-k gate.

### Same-Family OOD Check

LoCoMo-trained checkpoint evaluated on LongMemEval-S without LongMemEval
training. This is a same-family out-of-domain check, not a broad generalization
claim.

| Method | Questions | Recall@10 | Hit@10 | MRR |
|---|---:|---:|---:|---:|
| Raw MPNet retrieval | 500 | 0.905 | 0.946 | 0.783 |
| MiniLM cross-encoder top500 | 500 | 0.933 | 0.956 | 0.890 |
| ConvMemory LoCoMo checkpoint | 500 | 0.959 | 0.988 | 0.897 |

Paired bootstrap on the fixed 500-question LongMemEval-S set:

- ConvMemory vs raw MPNet: +0.0544 Recall@10, 95% CI [+0.0351, +0.0742], p < 0.001.
- ConvMemory vs MiniLM cross-encoder: +0.0261 Recall@10, 95% CI [+0.0065, +0.0461], p = 0.0088.
- ConvMemory vs MiniLM cross-encoder on MRR: +0.0069, 95% CI [-0.0179, +0.0320], not significant.

Stress setting: LongMemEval-S with each question expanded to a 1000-session
memory pool. Five distractor seeds, candidate-local ConvMemory, top500
reranking.

| Method | Recall@10 | Hit@10 | MRR | Mean ms/query |
|---|---:|---:|---:|---:|
| Raw MPNet retrieval | 0.5408 +/- 0.0054 | 0.6704 | 0.4400 +/- 0.0062 | 0.1 |
| ConvMemory candidate-local | 0.7258 +/- 0.0041 | 0.8280 | 0.6060 +/- 0.0077 | 76.6 |
| MiniLM cross-encoder top500 | 0.7312 +/- 0.0021 | 0.8440 | 0.6722 +/- 0.0052 | 954.6 |

In the 1000-session stress setting, ConvMemory improves over raw MPNet by
+0.1850 Recall@10, 95% CI [+0.1708, +0.1995]. The Recall@10 gap between
ConvMemory and MiniLM cross-encoder is not significant, but the cross-encoder
has significantly better MRR.

The repository also includes `v043_generic_retrieval_eval.py` for converted
external datasets. A synthetic agent-scratchpad sanity check confirms the JSONL
adapter works, but it is not a public OOD benchmark.

## What The Results Show

Supported by the current experiments:

- ConvMemory substantially improves recall-oriented memory retrieval over raw
  dense retrieval on LoCoMo-style memory streams.
- The gain is not explained by a simple BM25, recency, or dense-lexical fusion
  baseline alone.
- Temporal neighborhood features provide a measurable contribution.
- ConvMemory is cheaper than reranking the full top500 pool with the tested
  MiniLM cross-encoder.
- ConvMemory can improve a small cross-encoder cascade by providing a better
  candidate pool.
- Same-family OOD checks on LongMemEval-S show transfer beyond LoCoMo, including
  a 1000-session stress setting.

Not yet shown:

- State-of-the-art reranking performance.
- Robustness across many unrelated datasets.
- Superiority over stronger rerankers such as BGE, Jina, or mxbai rerankers.
- End-to-end answer-generation improvement.
- Production-grade calibrated cross-query thresholding.

## Reproducibility

Main reproduction commands:

```bash
python experiments/v040_baselines_ablation_stats.py \
  --device cuda \
  --seeds 7 11 23 31 47 \
  --bootstrap-samples 10000 \
  --out results/v040/baselines_ablation_stats

python experiments/v041_order_robustness.py \
  --device cuda \
  --seeds 7 11 23 31 47 \
  --out results/v041/order_robustness

python experiments/v042_error_calibration.py \
  --device cuda \
  --seeds 7 11 23 31 47 \
  --out results/v042/error_calibration

python experiments/v036_latency_benchmark.py \
  --device cuda \
  --checkpoint checkpoints/convmemory-locomo-mpnet \
  --encoder-model sentence-transformers/all-mpnet-base-v2 \
  --cross-encoder-model cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --seeds 7 11 23 \
  --limit 200 \
  --warmup 10 \
  --cross-batch-size 512 \
  --out results/v036/latency_benchmark

python experiments/evaluate_longmemeval_zero_shot.py \
  --device cuda \
  --checkpoint checkpoints/convmemory-locomo-mpnet \
  --encoder-model sentence-transformers/all-mpnet-base-v2 \
  --candidate-top-n 500 \
  --eval-cross-encoder \
  --cross-encoder-model cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --cross-batch-size 512 \
  --out results/v044/longmemeval_clean_fixed500

python experiments/v046_calibrate_confidence.py \
  --cases results/v042/error_calibration_mpnet/cases.csv \
  --out results/v046/confidence_calibration_mpnet
```

For the full evaluation plan, see
[docs/EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md). For checkpoint and
training details, see [docs/MODEL_CARD.md](docs/MODEL_CARD.md) and
[docs/TRAINING.md](docs/TRAINING.md).

## Project Status

Stable public API:

- `ConvMemory.from_pretrained`
- `ConvMemory.rerank`
- `ConvMemory.retrieve`
- `ConvMemory.expand_context`
- `ConvMemory.rerank_embeddings`
- `CompressionRouter`
- `build_compressed_notes`

Research-preview code:

- cascade fusion with cross-encoder scoring;
- stronger cross-encoder comparison scripts;
- generic JSONL adapters for external memory-retrieval datasets.

## License

MIT

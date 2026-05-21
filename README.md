# ConvMemory

[![CI](https://github.com/pth2002/ConvMemory/actions/workflows/ci.yml/badge.svg)](https://github.com/pth2002/ConvMemory/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

ConvMemory is a lightweight learned memory reranker for long-term
conversational and agent memory.

It runs after vector search and before prompt construction:

```text
user query -> vector search top-k -> ConvMemory -> memory context
```

ConvMemory is not a vector database, a full QA system, or a general document
reranker. Its intended use is recall-oriented memory selection for structured
memory streams: conversations, user histories, agent traces, task logs, and
session-level notes.

Current package version: `0.3.0`

## When To Use It

Use ConvMemory when:

- your memory store has session or user-history structure;
- raw vector search misses important neighboring evidence;
- you need a cheaper high-recall stage before an optional cross-encoder pass;
- the downstream agent can benefit from a compact, reranked memory context.

Do not use ConvMemory as:

- a vector database;
- a general web/document reranker;
- an end-to-end answer-generation model;
- a universal replacement for modern cross-encoders.

The strongest current deployment pattern is a cascade:

```text
vector top500 -> ConvMemory candidate stage -> optional small cross-encoder -> memory context
```

Public alpha: the CCGE-LA conflict editor can be inserted after ConvMemory to
repair stale/current memory conflicts. See [CCGE-LA](docs/CCGE_LA.md) and the
broader [research trajectory](docs/RESEARCH_TRAJECTORY.md). The API is
available in `convmemory.ccge`; the alpha editor checkpoint is published on
Hugging Face Hub for opt-in use.

## Installation

```bash
git clone https://github.com/pth2002/ConvMemory.git
cd ConvMemory
pip install -e .
pip install -r requirements.txt
```

ConvMemory requires Python 3.10 or later.

## Checkpoints

The public LoCoMo MPNet checkpoint is available from Hugging Face Hub:

[Purdy0228/ConvMemory-LoCoMo-MPNet](https://huggingface.co/Purdy0228/ConvMemory-LoCoMo-MPNet)

```python
from convmemory import ConvMemory

model = ConvMemory.from_pretrained("Purdy0228/ConvMemory-LoCoMo-MPNet")
```

The same checkpoint is also distributed as a GitHub release asset:

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

### Optional CCGE-LA Alpha Checkpoint

The CCGE-LA conflict editor has a separate alpha checkpoint on Hugging Face Hub:

[Purdy0228/ConvMemory-CCGE-LA](https://huggingface.co/Purdy0228/ConvMemory-CCGE-LA)

Attach it after loading the base ConvMemory checkpoint:

```python
model = ConvMemory.from_pretrained("Purdy0228/ConvMemory-LoCoMo-MPNet")
model.load_ccge_editor("Purdy0228/ConvMemory-CCGE-LA")
```

The same editor is also distributed as a GitHub release asset:

[Download `convmemory-ccge-la-locomo-mpnet-seed23-alpha.zip`](https://github.com/pth2002/ConvMemory/releases/download/ccge-la-alpha-v0.1/convmemory-ccge-la-locomo-mpnet-seed23-alpha.zip)

Extract it from the repository root:

```bash
unzip convmemory-ccge-la-locomo-mpnet-seed23-alpha.zip -d checkpoints
```

If using local files, attach it after loading the base ConvMemory checkpoint:

```python
model = ConvMemory.from_pretrained("checkpoints/convmemory-locomo-mpnet")
model.load_ccge_editor("checkpoints/convmemory-ccge-la-locomo-mpnet-seed23-alpha")
```

This is an alpha LoCoMo/MPNet editor trained on the seed-23 split. It is useful
for trying the public CCGE-LA API, but it is not yet the default checkpoint.
SHA256:
`459ecfb2b4c35887f1d8f2cdd87dab402c37bd8dee86628655eff08f314b2e7c`.

## Quick Start

```python
from convmemory import ConvMemory

model = ConvMemory.from_pretrained(
    "Purdy0228/ConvMemory-LoCoMo-MPNet",
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

Pass memories in a stable application order when that order is available.

## Agent Memory Integration

Most applications call ConvMemory after vector search:

```python
from convmemory import ConvMemory

memory_reranker = ConvMemory.from_pretrained(
    "Purdy0228/ConvMemory-LoCoMo-MPNet",
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

To enable the CCGE-LA conflict editor after ConvMemory, attach a trained editor
checkpoint and pass `editor="ccge_la"`:

```python
memory_reranker.load_ccge_editor("Purdy0228/ConvMemory-CCGE-LA")

ranked = memory_reranker.retrieve(
    query=query,
    memories=candidates,
    mode="rerank",
    editor="ccge_la",
    top_k=15,
)
```

The checkpoint and embeddings must use the same embedding model family and
embedding dimension.

## Results

These are retrieval-stage evaluations. They measure whether annotated evidence
memories are retrieved into the top-k list; they do not measure final answer
generation.

The tables below are summarized from internal evaluation artifacts. Large
per-question CSV files, embedding caches, teacher caches, and checkpoints are
intentionally excluded from the public Git history. See the `docs/` directory
for the public evaluation protocol, training notes, model card, and negative
results write-up.

Note: v0.40-v0.51 are internal evaluation-iteration identifiers for hardening
experiments, not packaged PyPI releases. The installable package version remains
0.3.0 and the public checkpoint is unchanged.

Important scope notes:

- The public checkpoint is trained on LoCoMo-style data; LoCoMo is in-domain
  for this checkpoint.
- The headline value proposition is cost-effective learned reranking for memory
  tasks, plus a rigorous negative result about mechanism attribution.
- Stronger rerankers matter. ConvMemory beats BGE-reranker-base/large on
  LoCoMo Recall@10, but it loses to `mxbai-rerank-large-v1` on both Recall@10
  and MRR.
- v0.50/v0.51 refute the stronger claim that temporal structure is the reason
  ConvMemory works. The learned temporal window contributes statistically, but
  its benefit is not temporally specific.
- External OOD results are mixed. The MuSiQue negative result is reported below
  because ConvMemory is not intended as a broad multi-hop document reranker.
- Latency numbers assume memory embeddings and memory-side indexes are already
  available. Cross-encoder timing includes pairwise scoring through the tested
  `CrossEncoder` path.

### LongMemEval Cost Advantage

This is the strongest practical story for ConvMemory: on memory-family tasks it
offers a much cheaper reranking stage while remaining recall-competitive.

| Setting | Method | Recall@10 | MRR | ms/query |
|---|---|---:|---:|---:|
| Clean500, BGE-large CE | Raw MPNet | 0.9049 | 0.7829 | 0.01 |
| Clean500, BGE-large CE | BGE-large CE top500 | 0.8807 | 0.8574 | 555.69 |
| Clean500, BGE-large CE | ConvMemory top500 | 0.9593 | 0.8973 | 44.00 |
| Clean500, mxbai CE | Raw MPNet | 0.9049 | 0.7829 | 0.01 |
| Clean500, mxbai CE | mxbai CE top500 | 0.9835 | 0.9317 | 1129.14 |
| Clean500, mxbai CE | ConvMemory top500 | 0.9593 | 0.8973 | 40.80 |
| Stress1000 seed23, BGE-large CE | Raw MPNet | 0.5452 | 0.4561 | 0.13 |
| Stress1000 seed23, BGE-large CE | BGE-large CE top500 | 0.6913 | 0.6651 | 5231.77 |
| Stress1000 seed23, BGE-large CE | ConvMemory candidate-local | 0.7386 | 0.6125 | 110.71 |
| Stress1000 seed23, mxbai CE | Raw MPNet | 0.5452 | 0.4561 | 0.12 |
| Stress1000 seed23, mxbai CE | mxbai CE top500 | 0.8195 | 0.7044 | 11211.63 |
| Stress1000 seed23, mxbai CE | ConvMemory candidate-local | 0.7386 | 0.6125 | 95.57 |

LongMemEval numbers are not seed-averaged: Clean500 is a single run and Stress1000 is reported for a single seed (23). Read these as indicative single-run retrieval-stage checks, not benchmark-grade comparisons.

Reading: ConvMemory reranks above BGE-large CE on these memory-family Recall@10
checks while being about 12-47x faster. It remains below mxbai accuracy, but is
about 28-117x lower latency in the tested settings.

### LoCoMo Strong Cross-Encoder Baselines

Five split seeds: 7, 11, 23, 31, 47. Candidate pool: raw dense top500.

| Reranker | Recall@10 | Hit@10 | MRR |
|---|---:|---:|---:|
| ConvMemory (v0.40 5-seed) | 0.7798 +/- 0.0074 | not reported | 0.5824 |
| BGE-reranker-base | 0.6967 +/- 0.0126 | 0.7469 +/- 0.0144 | 0.5469 +/- 0.0140 |
| Jina-reranker-v2-base-multilingual | 0.7411 +/- 0.0103 | 0.7924 +/- 0.0083 | 0.5754 +/- 0.0074 |
| BGE-reranker-large | 0.7621 +/- 0.0155 | 0.8124 +/- 0.0135 | 0.6120 +/- 0.0144 |
| mxbai-rerank-large-v1 | 0.8080 +/- 0.0153 | 0.8486 +/- 0.0108 | 0.6687 +/- 0.0093 |

Reading: ConvMemory is competitive on recall, but it should not be given an
overall cross-encoder superiority claim. `mxbai-rerank-large-v1` is stronger on
LoCoMo Recall@10 and MRR.

### Retrained Ablation

Three split seeds, MPNet. These are retrained ablations, not inference-time
feature masks.

| Variant | Recall@10 | MRR | Delta R@10 vs full |
|---|---:|---:|---:|
| full_control | 0.7474 +/- 0.0229 | 0.5343 +/- 0.0160 | 0.0000 |
| no_router | 0.7491 +/- 0.0213 | 0.5391 +/- 0.0137 | +0.0017 +/- 0.0020 |
| no_temporal_w1 | 0.7121 +/- 0.0232 | 0.5305 +/- 0.0148 | -0.0353 +/- 0.0052 |
| no_lexical | 0.6584 +/- 0.0185 | 0.4367 +/- 0.0129 | -0.0890 +/- 0.0061 |
| no_lexical_no_router | 0.6574 +/- 0.0163 | 0.4342 +/- 0.0127 | -0.0899 +/- 0.0087 |

Reading: lexical interaction features are the largest contributor. The
no-temporal variant is weaker than the full model in this three-seed ablation,
but this table alone does not prove that the gain is temporally specific. The
router/DCA scalar contributes approximately zero; removing it is neutral to
slightly positive, so it is treated as an experimental negative result rather
than a model feature.

### Attribution / Negative Result

The v0.50/v0.51 follow-up was designed to test whether the temporal window is
the load-bearing reason ConvMemory works. This section uses the retrained
attribution pipeline, not the v0.40 headline pipeline above.

Five split seeds: 7, 11, 23, 31, 47.

| Method | Recall@10 |
|---|---:|
| full_control_retrained | 0.7432 +/- 0.0207 |
| no_temporal_w1_retrained | 0.7054 +/- 0.0221 |
| tuned_heuristic | 0.7234 +/- 0.0227 |
| raw_dense | 0.5345 +/- 0.0210 |

Paired bootstrap, `full_control_retrained - no_temporal_w1_retrained`,
Recall@10:

| Slice | Delta | 95% CI | Reading |
|---|---:|---:|---|
| ALL | +0.0376 | [+0.0306, +0.0451] | significant |
| T_SUP_auto | +0.0407 | [+0.0219, +0.0603] | significant, open question |
| T_REQUIRED_auto | +0.0252 | [+0.0139, +0.0363] | significant |
| T_HOP_auto | +0.0096 | [-0.0037, +0.0230] | not significant |
| OTHER | +0.0868 | [+0.0672, +0.1045] | significant |
| HARD_NON_TEMPORAL_auto | +0.0838 | [+0.0650, +0.1040] | significant |

The honest reading is negative for the original temporal-mechanism thesis: the
learned temporal window contributes on aggregate, but its benefit is largest on
hard non-temporal controls (`OTHER` and `HARD_NON_TEMPORAL_auto`) and is not
statistically significant on the most temporal multi-hop proxy (`T_HOP_auto`).
This looks more like generic neighborhood/capacity smoothing than proven
temporal-structure exploitation. `T_SUP_auto` remains the only notable open
question, but it is still smaller than the hard non-temporal control effect and
should not be used as a load-bearing temporal claim.

Against the tuned heuristic, the same retrained attribution pipeline gives
`full_control_retrained` a Recall@10 delta of +0.0199 with 95% CI
[+0.0105, +0.0283], and an MRR delta of +0.0566. So the learned reranker still
adds value; the negative result is about why it works.

See [docs/NEGATIVE_RESULTS.md](docs/NEGATIVE_RESULTS.md) for the full
v0.50/v0.51 interpretation.

For the later current/stale and conflict-editor research trajectory, see
[docs/RESEARCH_TRAJECTORY.md](docs/RESEARCH_TRAJECTORY.md). That document
summarizes the internal v0.60+ research line in a public-safe form without raw
logs, private paths, caches, or exploratory scripts.

### Strong-Backbone Retraining

Three split seeds. ConvMemory is retrained in each embedding space.

| Backbone | Raw Recall@10 | ConvMemory Recall@10 | Gain | ConvMemory MRR |
|---|---:|---:|---:|---:|
| BGE-large | 0.6680 +/- 0.0237 | 0.7726 +/- 0.0100 | +0.1046 +/- 0.0137 | 0.5639 +/- 0.0066 |
| E5-large | 0.7010 +/- 0.0216 | 0.7902 +/- 0.0171 | +0.0892 +/- 0.0052 | 0.5941 +/- 0.0103 |

Reading: ConvMemory gains are not just an artifact of a weak MPNet retriever.
Retraining on stronger embeddings still gives about +9 to +10 Recall@10 points.

### External OOD Results

Single run per dataset. These are intentionally reported as mixed evidence.

| Dataset | Questions | ConvMemory R@10 | Raw dense | Dense + lexical | BM25 |
|---|---:|---:|---:|---:|---:|
| QMSum | 272 | 0.5882 | 0.4724 | 0.5423 | 0.5294 |
| MSC persona | 6155 | 0.9632 | 0.8375 | 0.9765 | 0.9920 |
| HotpotQA | 1000 | 0.7983 | 0.7682 | 0.8621 | 0.8280 |
| MuSiQue | 1000 | 0.7635 | 0.8640 | 0.8175 | 0.7245 |

These external OOD results are single runs without seed averaging or confidence intervals; treat them as indicative scope checks, not benchmark-grade comparisons.

Reading: ConvMemory wins on QMSum and improves strongly over raw dense on MSC,
but lexical/BM25 baselines dominate MSC's weak persona-overlap labels. On
HotpotQA, a trivial dense+lexical baseline is stronger. On MuSiQue, ConvMemory
regresses below raw dense. This is a scope boundary: ConvMemory is a memory
reranker, not a general multi-hop document reranker.

### Where ConvMemory Fails

- Non-temporal multi-hop retrieval: MuSiQue is negative against raw dense.
- Lexically anchored document retrieval: HotpotQA favors dense+lexical scoring.
- Maximum top-rank precision: mxbai-rerank-large remains stronger on LoCoMo MRR.
- Cross-query score calibration: scores should not be treated as calibrated
  confidence without application-specific validation.
- Mechanism attribution: v0.51 does not support temporal structure as the
  load-bearing explanation for ConvMemory's gain.

## Reproducibility

The current documentation reports the hardened v0.47/v0.51 audit. See:

- [docs/EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md)
- [docs/MODEL_CARD.md](docs/MODEL_CARD.md)
- [docs/TRAINING.md](docs/TRAINING.md)

Main evaluation artifacts are kept outside the repository history. Large
per-question CSV files, teacher caches, and checkpoints are intentionally not
committed.

## Project Status

Stable public API:

- `ConvMemory.from_pretrained`
- `ConvMemory.rerank`
- `ConvMemory.retrieve`
- `ConvMemory.expand_context`
- `ConvMemory.rerank_embeddings`

Public alpha API:

- `ConvMemory.attach_ccge_editor`
- `ConvMemory.load_ccge_editor`
- `CCGELowAmplitudeEditor`
- `build_ccge_features`

Research-preview code:

- context expansion policies for wider agent memory budgets;
- cascade fusion with cross-encoder scoring;
- stronger cross-encoder comparison scripts;
- generic JSONL adapters for external memory-retrieval datasets.

Not included in the public package:

- raw datasets, checkpoints, embedding caches, teacher caches, and full
  per-question result CSVs;
- local experiment logs and remote execution archives;
- exploratory numbered experiment prototypes unless they are explicitly promoted
  into the documented public API.

CCGE-LA is packaged as a public alpha API for conflict-aware ConvMemory editing.
It should still be treated as experimental until a public training recipe and
same-split public evaluation command are released.

## License

MIT

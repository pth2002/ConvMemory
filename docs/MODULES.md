# Architecture Notes

This note summarizes the main components behind ConvMemory. Most users only need
the public `ConvMemory` API.

Runtime code lives in the installable `convmemory/` package:

- `convmemory/api.py`: user-facing `ConvMemory` wrapper.
- `convmemory/reranker.py`: embedding-level reranking and candidate-local windowing.
- `convmemory/encoder.py`: Conv/Mixer candidate-window encoder.
- `convmemory/scoring.py`: CE-lite scorer, lexical cache, and score fusion helpers.
- `convmemory/ccge.py`: CCGE-LA conflict-aware candidate-set editor.
- `convmemory/metrics.py`: small retrieval metrics used by examples and experiments.

The `experiments/support/` helpers are used by the reproduction scripts. They
are kept outside the installable wheel so the library API stays small.

## Public API Modes

The public API intentionally exposes two stable modes:

- `rerank`: score and reorder candidate memories.
- `expand`: keep a protected reranked prefix, then add complementary candidates
  for a wider agent memory context.

Both modes can optionally run the public alpha CCGE-LA editor after ConvMemory
by passing `editor="ccge_la"` to `retrieve`, `rerank`, or `rerank_embeddings`.
The editor must first be attached with `attach_ccge_editor(...)` or loaded with
`load_ccge_editor(...)`.

The expansion mode is a context-construction layer, not a separate neural
architecture. It is designed for agent systems where the downstream LLM can read
more than the strict top-k and missing a key memory is worse than including a
few additional candidates.

## 1. Conv/Mixer Candidate-Window Encoder

The core memory encoder reads a short candidate-local window of memory
embeddings:

```text
[candidate_i-2, candidate_i-1, candidate_i, candidate_i+1, candidate_i+2]
```

It uses lightweight convolution plus Mixer-style token/channel mixing over this
candidate window.

The older v0.48 ablation showed that `window_size=1` is weaker than the full
model. The newer v0.51 attribution check is stricter: the window contributes on
aggregate, but its gain is largest on hard non-temporal controls and is not
significant on the T_HOP temporal proxy. Therefore this encoder should be
described as a learned candidate-neighborhood module, not as proven temporal
structure exploitation.

## 2. Lexical Features

ConvMemory includes small lexical overlap features:

- token overlap;
- token recall against the query;
- bigram overlap;
- bigram recall against the query.

Retrained ablation shows these are the largest contributor in the current
checkpoint. Removing lexical features reduces Recall@10 by 0.0890 relative to
the full model.

## 3. CE-lite Scorer

The CE-lite scorer fuses:

- query embedding;
- candidate memory embedding;
- query-memory interaction features;
- raw dense score;
- ConvMemory window score;
- rank and position features;
- lexical features.

It is not a token-level cross-encoder. It operates over precomputed embeddings
plus lightweight side features.

The current module description is:

```text
Conv/Mixer candidate-window + lexical/query interaction CE-lite reranking
```

## 4. Retired Experimental Router Scalar

Earlier experiments included a router/DCA-style scalar side feature. Hardened
v0.48 retrained ablation found no measurable benefit:

```text
full_control Recall@10: 0.7474
no_router    Recall@10: 0.7491
delta:       +0.0017
```

This scalar is therefore treated as an experimental negative result, not a
feature or selling point.

## 5. Raw Dense Score Fusion

The final ranking can retain a small amount of raw retriever score:

```text
final_score = raw_weight * raw_score + (1 - raw_weight) * convmemory_score
```

The current best setting usually keeps `raw_weight` near `0` to `0.025`.

## 6. Context Expansion

`ConvMemory.expand_context(...)` and `ConvMemory.retrieve(..., mode="expand")`
protect the strongest ConvMemory results, then fill the remaining context budget
from complementary rankings. The default policy uses the main ConvMemory
ranking, raw dense retrieval, and candidate-local window scoring.

Expansion is useful when the downstream agent can read a wider context and the
cost of missing a key memory is higher than the cost of including a few extra
candidates.

## 7. Optional Compressed-Note Candidate Selection

Some applications maintain session summaries or compressed notes. In that case,
compressed-note search can be used as a candidate-selection layer before
ConvMemory reranking:

```text
query embedding -> compressed-note search -> raw memory candidate ids -> ConvMemory rerank
```

This is separate from the neural reranker and is not part of the core v0.51
mechanism claim.

## 8. Cascade Fusion Research Path

The public package does not yet expose a stable cascade API. Current experiments
support a practical optional path:

```text
ConvMemory candidate stage -> small cross-encoder pass -> normalized score fusion
```

Treat this as a research-preview path rather than the default library interface.

## 9. CCGE-LA Conflict Editor

CCGE-LA is a public alpha candidate-set editor for stale/current memory
conflicts:

```text
vector search -> ConvMemory -> CCGE-LA low-amplitude edit -> context
```

The editor reads the ConvMemory candidate set, builds conflict-state features,
and applies a small gated residual update to ConvMemory scores. It is intended
to repair cases where old and current memories are semantically similar and the
gold memory is already in the candidate pool.

The public API is:

- `CCGELowAmplitudeEditor`
- `build_ccge_features`
- `ConvMemory.attach_ccge_editor`
- `ConvMemory.load_ccge_editor`
- `editor="ccge_la"` in `retrieve`, `rerank`, and `rerank_embeddings`

The current repository exposes the API but does not ship trained CCGE-LA
weights. Randomly initialized editors are useful only for smoke tests.

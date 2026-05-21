# CCGE-LA: Conflict-Aware ConvMemory Editor

CCGE-LA is the current research-preview structure for stale/current memory
conflicts.

The name means:

> Low-Amplitude Counterfactual Conflict Graph Editor

It is not a replacement for ConvMemory. It is a lightweight editor that runs
after ConvMemory and before agent context construction.

## Placement

```text
query
  -> vector search top-k
  -> ConvMemory rerank
  -> CCGE-LA conflict-aware score edit
  -> memory context for the downstream agent
```

ConvMemory remains the semantic memory retriever. CCGE-LA reads ConvMemory's
candidate set and applies a small residual correction when the candidate set
looks conflict-prone.

## Why This Structure Exists

Long-term memory systems often retrieve both stale and current facts. A normal
semantic retriever may rank an older but very similar memory above the current
one.

CCGE-LA targets this failure mode by reading the whole candidate set. It looks
for signs such as:

- semantically similar candidates at different time/order positions;
- narrow score margins at the top of the list;
- high candidate-set entropy;
- dense clusters of memories around the same topic;
- ConvMemory top-1 failures where the gold memory is already in the pool.

The editor does not decide the final answer. It only changes the memory ranking
before the LLM sees the context.

## Score Update

The core update is residual:

```text
final_score_i = convmemory_score_i + gate(candidate_set) * residual_i
```

The gate is query/candidate-set level and intentionally low-amplitude. Internal
experiments found that making the gate more granular at the candidate level was
less stable than a small query-level editor.

## Public Prototype

A clean research-preview implementation is provided in:

```text
experiments/ccge_la_research_preview.py
```

It exposes:

- `build_conflict_features`
- `CCGELowAmplitudeEditor`
- `multi_positive_retrieval_loss`
- `rank_candidates`

This file is an architecture and training scaffold, not a pretrained public
checkpoint.

## Training Signal

The intended training objective is retrieval cross-entropy over the candidate
set.

Do not train it with:

- current/stale labels;
- is-gold or is-current feature indicators;
- gold-defined "latest" inputs;
- auxiliary state supervision;
- offline distillation as the defining mechanism.

Those shortcuts would undermine the structural claim.

## Internal Evidence Summary

The strongest internal real-LoCoMo line used ConvMemory as the candidate source
and trained a conflict-state residual editor over the candidate set.

Representative five-seed internal results:

| Setting | ConvMemory MRR | Best editor MRR | Delta |
|---|---:|---:|---:|
| FULL | 0.5708 | 0.5878 | +0.0170 |
| T_SUP_auto | 0.5518 | 0.5803 | +0.0285 |
| ConvMemory top-1 wrong, gold in pool | 0.2586 | 0.3285 | +0.0699 |
| Rescuable stale top-1 | 0.2636 | 0.3655 | +0.1019 |

Reading:

- The full-set improvement is modest but positive.
- The improvement is larger on supersession/stale-conflict slices.
- More complex candidate-level gates, directional graphs, and dual-head variants
  did not become the best mainline.

## Relation To Cross-Encoders

CCGE-LA is still a reranking layer. It does not remove the need for reranking.

Compared with a cross-encoder:

- cross-encoders usually have stronger top-rank precision;
- CCGE-LA is cheaper and candidate-set structured;
- CCGE-LA is designed for stale/current memory conflict repair, not generic
  document ranking.

The intended deployment pattern is:

```text
vector top-k -> ConvMemory -> CCGE-LA -> optional cross-encoder -> context
```

For cost-sensitive agents, CCGE-LA can be used as a lightweight conflict-aware
reranker. For maximum accuracy, a cross-encoder can still be used after it.

## Current Status

CCGE-LA is a research-preview structure. Before it becomes a stable public API,
it still needs:

- a public training recipe;
- a packaged checkpoint;
- deterministic feature extraction integrated with `ConvMemory.retrieve`;
- same-split comparisons against raw dense retrieval, ConvMemory, and a strong
  cross-encoder.

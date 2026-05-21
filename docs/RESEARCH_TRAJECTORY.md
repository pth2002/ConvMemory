# Research Trajectory

This document summarizes the internal research line that followed the public
ConvMemory LoCoMo/LongMemEval evaluation work.

The version numbers below are internal experiment identifiers, not package
versions. They are included to make the research path auditable without
publishing raw logs, remote execution records, private datasets, caches, or
exploratory scripts.

## Scope

The public package remains the documented ConvMemory reranker API:

- `ConvMemory.from_pretrained`
- `ConvMemory.rerank`
- `ConvMemory.retrieve`
- `ConvMemory.expand_context`
- `ConvMemory.rerank_embeddings`

The conflict-aware editor work described here is a research direction, not yet a
stable public API.

## Motivation

ConvMemory is strong as a memory reranker, but later attribution work showed
that its original "temporal convolution" story was too strong. The learned
window/context features help, but the gain is not cleanly explained by temporal
reasoning alone.

The follow-up research asked a narrower question:

> Can we add a structure that specifically helps when a memory system retrieves
> stale or conflicting facts?

This led to two parallel tracks:

1. Clean, controlled benchmarks for current/stale or as-of fact retrieval.
2. Real LoCoMo post-retrieval editors over ConvMemory candidate sets.

## V0.60-V0.92: Real LoCoMo Version-State Prototypes

The first real-data attempts tested slot/state, raw-delta, version-chain,
router, temporal-contrast, dual-expert, and cached setwise variants on LoCoMo.

High-level finding:

- Raw dense and ConvMemory were already strong candidate generators.
- Simple global recency or event-time priors did not transfer reliably.
- Many architectures helped diagnostic slices but did not produce a stable
  mainline improvement.
- Candidate availability was a recurring bottleneck: when the gold memory was
  not in the candidate pool, no reranker-style structure could recover it.

Public interpretation:

These experiments are useful negative evidence. They should not be promoted as
final methods. They motivated the later split between candidate generation and
candidate-set conflict resolution.

## V0.93-V1.09: State And Compiler Diagnostics

The next stage tested whether explicit version-state structures could learn a
clean current/stale rule under more controlled conditions.

Important lessons:

- Several synthetic tasks were accidentally degenerate: the gold answer could be
  recovered from an input shortcut such as recency or position.
- Oracle and diagnostic variants were useful for identifying whether a structure
  had enough information, but they were not deployable methods.
- Clean benchmark design mattered more than adding model capacity.

Public interpretation:

This stage should be described as benchmark hardening and mechanism discovery,
not as evidence for a final architecture.

## V1.10-V1.42: Fair Temporal Memory Benchmarks

The synthetic path was redesigned into harder narrative benchmarks with:

- current queries, where the answer is the latest actual value;
- as-of queries, where the answer is the value at or before a target date;
- paraphrasing and adversarial event phrasing;
- external verifier checks on sampled examples.

The cleanest result is the V142-style instructional narrative benchmark.

### Current Queries

Current queries do not contain an as-of date. On this subset, the defensible
mechanism is:

> learned support extraction + event-time ordering

Representative V142 decomposition:

| Arm | R@1 | MRR | stale@1 |
|---|---:|---:|---:|
| support only | 0.4660 | 0.6909 | 0.3990 |
| event-time ordering after support | 0.7360 | 0.8539 | 0.0700 |

This is the cleanest mechanism evidence because it does not rely on a query
date. It shows that support extraction plus event-time ordering can strongly
reduce stale top-1 errors in a controlled benchmark.

### As-Of Queries

As-of queries contain a target date. On this subset, a hard as-of filter using
the query date is highly effective:

| Arm | R@1 | MRR | stale@1 |
|---|---:|---:|---:|
| support only | 0.6675 | 0.8025 | 0.3325 |
| event-time latest without as-of filter | 0.1450 | 0.5625 | 0.8550 |
| support + event-time + as-of filter | 0.9475 | 0.9738 | 0.0525 |

Caveat:

This is a useful point-in-time retrieval operation, but it uses the date
embedded in the query. It should not be presented as the same mechanism as
current-query stale-fact resolution.

### Honest Synthetic Claim

The synthetic claim should be split:

- Current queries: learned support extraction plus event-time ordering is the
  real mechanism.
- As-of queries: learned support extraction plus explicit query-date filtering
  is a strong engineering composition.

Do not collapse these into a single aggregate "bitemporal" gain.

## V1.43-V1.51: Real LoCoMo Conflict Editors

After the synthetic path was clarified, the idea was moved back to real LoCoMo
as a post-ConvMemory editor.

The final research direction is:

> ConvMemory retrieves semantically relevant memories; a lightweight
> conflict-state editor reads the candidate set and applies a low-amplitude
> residual correction to ConvMemory scores.

This module is best described as a candidate-set conflict editor, not a
standalone retriever and not an answer generator.

### Best Real-Data Findings

Representative five-seed LoCoMo results:

| Setting | ConvMemory MRR | Best editor MRR | Delta |
|---|---:|---:|---:|
| FULL | 0.5708 | 0.5878 | +0.0170 |
| T_SUP_auto | 0.5518 | 0.5803 | +0.0285 |
| ConvMemory top-1 wrong, gold in pool | 0.2586 | 0.3285 | +0.0699 |
| Rescuable stale top-1 | 0.2636 | 0.3655 | +0.1019 |

Interpretation:

- The editor gives a modest but consistent lift over ConvMemory on the full
  real-data evaluation.
- The lift is larger on supersession/conflict slices.
- The strongest practical form is a low-amplitude query-level residual editor.
- More complex candidate-level gates, directional graph variants, and dual-head
  variants did not become the best mainline.

### Working Name

The research working name is:

> CCGE-LA: Low-Amplitude Counterfactual Conflict Graph Editor

However, the exact public method name should only be finalized once the editor
is promoted into a stable API. In current public documentation, it is safer to
call it a conflict-aware ConvMemory editor.

### Current Structure

The editor sits between retrieval and context construction:

```text
query
  -> vector search / dense candidate generation
  -> ConvMemory candidate reranking
  -> conflict-state residual editor
  -> final memory context for the downstream agent
```

The editor reads candidate-set features such as:

- ConvMemory score and rank;
- dense score and rank;
- candidate time/order position;
- score margin and entropy of the candidate set;
- semantic density among top candidates;
- whether similar candidates appear at different temporal positions.

The score update is deliberately small:

```text
final_score_i = convmemory_score_i + gate(candidate_set) * residual_i
```

Training uses retrieval cross-entropy only. It does not use:

- gold-defining input features;
- current/stale labels;
- auxiliary supervision for entity states;
- offline distillation as the defining mechanism.

### Relation To Strong Rerankers

The editor is still a reranker/editor. It does not remove the need for reranking.

Compared with cross-encoders:

- cross-encoders are usually stronger at top-rank precision;
- ConvMemory and ConvMemory-style editors can be cheaper and more recall
  oriented;
- the conflict editor specifically targets stale/current candidate-set errors.

The right positioning is not "replace all cross-encoders." It is:

> a lightweight conflict-aware reranking layer for memory systems, optionally
> used before a stronger but more expensive cross-encoder.

## What To Claim

Reasonable public claims:

- ConvMemory is a lightweight learned memory reranker for long-term memory
  streams.
- ConvMemory's public checkpoint is useful as a recall-oriented memory reranker.
- The original temporal-mechanism explanation was too strong; later audits
  support a more conservative explanation.
- Controlled benchmarks show that support extraction plus event-time ordering
  can reduce stale-fact retrieval errors.
- Real LoCoMo conflict-editor prototypes improve ConvMemory most strongly on
  supersession and stale-top1 diagnostic slices.

Claims to avoid:

- ConvMemory is a general replacement for cross-encoders.
- Temporal convolution is proven to be the load-bearing mechanism.
- As-of benchmark gains prove current-query reasoning without caveats.
- The conflict editor is already a stable public API.
- Internal version IDs are package releases.

## Next Public Engineering Step

Before publishing the conflict editor as a feature, it should be cleaned into:

- a small documented module;
- deterministic feature extraction code;
- a stable checkpoint/export format;
- a reproducible public command;
- an explicit comparison against ConvMemory, raw dense retrieval, and a strong
  cross-encoder under the same split and candidate pool.

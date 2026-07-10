# ConvMemory v3 Validity Context Model Card

This document separates the three source-of-truth layers for ConvMemory v3:

1. method-level evaluation;
2. the exported representative checkpoint;
3. package-level API measurement with that checkpoint.

These are intentionally different provenance layers.  Method-level numbers
estimate the v3 approach across seeds.  The checkpoint is a representative
implementation of that approach.  The package-level benchmark is the number a
user should expect when loading that checkpoint through the public API on the
fixed dense Memora retrieval benchmark.

## Scope

ConvMemory v3 adds validity evidence to the existing v1/v2 retrieval path.
The default v3 use is `validity_mode="context"`: it attaches a structured
`validity` field to returned memories and preserves the candidate set and
ranking order.

`validity_mode="demote"` is explicit opt-in. It is intended for dense
current-state/update workloads where a top-1 source evidence policy is
available. It preserves the candidate set and may reorder by applying a validity
penalty.

ConvMemory v3 does not make full automatic dependency-graph propagation the
default retrieval path. Multi-hop graph propagation is used as an evidence-path
and analysis capability unless the caller supplies a workload where graph
construction has been validated.

## Checkpoint

The representative v3 validity checkpoint is exported by the v557 recipe:

| Field | Value |
|---|---|
| Module | ConvMemory v3 Validity Context Layer |
| Backbone | `nli-deberta-v3-base` |
| Parameters | `184,423,682` |
| Export seed | `7` |
| Training rows | `5,520` |
| Dev rows | `1,400` |
| Threshold | `0.5` |
| Max length | `192` |
| Source policy | `top1` |
| Default mode | `context` |
| Hub repository | `Purdy0228/ConvMemory-v3-Validity-Context` |
| Checkpoint upload commit | `0883a43fe6df608030ebe9ec29286280e83c857c` |
| `cross_encoder/model.safetensors` SHA256 | `446ee0cf6df4a8967e1a78c46d2ff3a2d777de65efbf475d2278d99468faa8d9` |
| `validity_config.json` SHA256 | `81eddb5f2ff4545dcf4b7655fedd1f7cf846248ad8962394195e6960a2e07849` |

The checkpoint implements the v511 query-conditioned validity method.  It
should not be used as a replacement for the v511 multi-seed method-level
estimate when reporting method quality.

## Input Format

The validity scorer uses the v506/v511 query/source/target format:

```text
USER_QUERY:
...

SOURCE_EVIDENCE:
...

TASK: Decide whether the target memory should be demoted for this user query.
```

paired with:

```text
TARGET_MEMORY:
...
```

The package exposes two scoring paths:

- `ValidityEvidenceModule.apply(...)`: annotate or demote `RerankResult`
  objects while preserving the mode contracts.
- `ValidityEvidenceModule.score_evidence_pairs(...)`: batch explicit
  query/source/target pairs after source evidence has already been selected.

The second path is the preferred dense-workload path because it avoids per-pair
CrossEncoder calls.

## Method-Level Evaluation

The v511 5-seed Memora-retrieval benchmark is the method-level estimate. It
scores `69,200` source-query rows across seeds `[7, 11, 23, 31, 47]`.

Top-1 retrieved source, max aggregation:

| Metric | v511 method-level |
|---|---:|
| Pair accuracy | `98.6% +/- 0.2%` |
| Demote recall | `92.9% +/- 1.1%` |
| Protect recall | `99.4% +/- 0.1%` |
| Old-target all-type consistency | `92.8% +/- 1.1%` |
| Event all-type consistency | `89.1% +/- 1.3%` |
| Current active H@1 | `95.7% +/- 1.2%` |
| Scoring cost | `1.9291` ms/source-query pair |

This table is the right citation for method-level claims.

## Package-Level Check

The v558 public API benchmark loads the exported v557 checkpoint through
`ValidityEvidenceModule.from_pretrained(...)` and scores the same top-1 source
policy through the package API.

Top-1 retrieved source, max aggregation:

| Metric | v558 package/API check |
|---|---:|
| Source-query rows | `6,920` |
| Target predictions | `20,760` |
| Pair accuracy | `98.7%` |
| Demote recall | `93.6%` |
| Protect recall | `99.4%` |
| Old-target all-type consistency | `93.1%` |
| Event all-type consistency | `89.6%` |
| Current active H@1 | `96.5%` |
| API scoring batch size | `512` |
| Scoring cost | `1.5844` ms/source-query pair |
| Module load time | `2.16` s |

The v558 number is the package-level reproducibility check for this checkpoint.
It is a single-checkpoint measurement, not a replacement for the v511
multi-seed method-level estimate.

## Safety Contracts

The package-level safety checks from v558 all pass:

| Contract | Status |
|---|---:|
| `context` mode preserves order | `pass` |
| `context` mode preserves ranks | `pass` |
| `context` mode attaches validity metadata | `pass` |
| `demote` mode preserves candidate set | `pass` |
| `demote` mode preserves result count | `pass` |

The test suite also covers off-mode byte identity, context-mode rank
preservation, demote candidate-set preservation, explicit opt-in semantics,
forbidden-field rejection, safe evidence output, checkpoint round-trip, and
batched CrossEncoder scoring.

## Operating Policy

| Workload | Recommended mode | Source policy | Ranking mutation |
|---|---|---|---|
| General ConvMemory retrieval | `context` | top-1 evidence when available | no |
| Dense current-state/update retrieval | `demote` opt-in | lexical top-1 source | yes, candidate set preserved |
| Multi-hop graph explanation | `context` | conservative graph evidence | no |

Top-3/top-5 source aggregation is not the default policy because earlier v499,
v502, and v503 runs showed that adding more sources can introduce false
positive demotions. Full top-500 graph construction is also not the default
path because learned graph errors can be amplified by propagation. The
integrated API protects the requested result prefix and selects at
most one later source per target with a cheap lexical rule. Applications with a
dedicated update index should pass that result through `validity_source_map`.

The package latency below measures the validity scorer on already selected
source/query pairs. It does not include application-specific evidence-index
lookup. The bounded in-package fallback avoids unbounded all-pairs CrossEncoder
scoring, but its cheap source-selection work remains part of end-to-end latency.

## Source-Of-Truth Ledger

| Claim or artifact | Value or role | Source file | Provenance layer | Availability |
|---|---|---|---|---|
| v3 method-level dense benchmark | v511 5-seed top1: old-target all-type `92.8% +/- 1.1%`, current active H@1 `95.7% +/- 1.2%` | `results/v511_memora_retrieval_demotion_benchmark_5seed/REPORT.md` | method-level evaluation | author-retained results |
| v3 frozen configuration policy | default context mode; demote opt-in for dense current-state/update workloads; top1 source | `results/v514_v3_freeze_config/final_config.json` | configuration freeze | author-retained results |
| exported checkpoint manifest | seed-7 representative checkpoint; `184,423,682` params; threshold `0.5`; Hub repo `Purdy0228/ConvMemory-v3-Validity-Context` | `results/v557_v3_validity_checkpoint/seed_7/MANIFEST.json` | checkpoint export | checkpoint artifact / author-retained manifest |
| checkpoint scorer config | `mode_default="context"`, `source_policy="top1"`, `cross_encoder_num_labels=2` | `results/v557_v3_validity_checkpoint/seed_7/validity_config.json` | checkpoint export | checkpoint artifact / author-retained config |
| Hub checkpoint commit | `0883a43fe6df608030ebe9ec29286280e83c857c` | `Purdy0228/ConvMemory-v3-Validity-Context` | checkpoint hosting | Hugging Face Hub |
| package API benchmark | v558 top1 package check: old-target all-type `93.1%`, current active H@1 `96.5%` | `results/v558_v3_public_api_benchmark_batch/REPORT.md` | package-level measurement | author-retained results |
| package API latency | `1.5844` ms/source-query pair, API batch size `512` | `results/v558_v3_public_api_benchmark_batch/summary.json` | package-level measurement | author-retained results |
| validity module code | `ValidityEvidenceModule`, `ValidityEvidenceConfig`, `score_evidence_pairs` | `convmemory/validity.py` | package code | public package when tagged `v0.6.0` |
| public API integration | `load_validity_module`, `validity_mode`, retrieve/rerank integration | `convmemory/api.py` | package code | public package when tagged `v0.6.0` |
| result payload | `RerankResult.validity` | `convmemory/reranker.py` | package code | public package when tagged `v0.6.0` |
| safety tests | `41 passed` after v558 batch update | `tests/test_validity_context.py` and existing package tests | machine-checkable tests | public package when tagged `v0.6.0` |
| user documentation | mode semantics, safety contracts, scorer format | `docs/VALIDITY_CONTEXT.md` | package documentation | public package when tagged `v0.6.0` |
| release sanity checklist | package contents, tests, HF cold-start load path | `docs/V3_RELEASE_SANITY.md` | release verification | public package when tagged `v0.6.0` |

The `results/...` packets are source-of-truth evaluation artifacts kept with
the author workspace unless explicitly packaged with a release. The package
code, tests, and documentation are the public reproducibility surface once tag
`v0.6.0` is cut.

## Known Boundaries

- The v3 checkpoint is trained for query-conditioned validity decisions with
  source evidence. It is not a generic factuality judge.
- Automatic demotion is intended for dense current-state/update workloads.
  General sparse retrieval should use context annotation by default.
- Broad learned source retrieval and automatic strict dependency graph
  construction are not part of the default v3 retrieval contract.
- The v511 method-level estimate and v558 package-level benchmark use different
  but connected provenance layers; report them with their layer names.

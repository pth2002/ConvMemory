# Benchmarks

Full evaluation tables for ConvMemory. The README carries the summary; this file
carries everything, including the results that do not favour ConvMemory.

These are **retrieval-stage** evaluations. They measure whether annotated
evidence memories are retrieved into the top-k list; they do not measure final
answer generation.

The tables are summarized from internal evaluation artifacts. Large per-question
CSV files, embedding caches, teacher caches, and checkpoints are intentionally
excluded from the public Git history. See [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md)
for the protocol, [MODEL_CARD.md](MODEL_CARD.md) for the checkpoint, and
[TRAINING.md](TRAINING.md) for training notes.

Note: v0.40-v0.51 are internal evaluation-iteration identifiers for hardening
experiments, not packaged PyPI releases.

## Scope notes (read before quoting any number)

- The public checkpoint is trained on LoCoMo-style data; **LoCoMo is in-domain**
  for this checkpoint. Splits are conversation-level, so the reported numbers are
  on held-out conversations, but not on a different data distribution.
- The headline value proposition is cost-effective learned reranking for memory
  tasks, plus a rigorous negative result about mechanism attribution.
- Stronger rerankers matter. ConvMemory beats BGE-reranker-base/large on LoCoMo
  Recall@10, but it loses to `mxbai-rerank-large-v1` on both Recall@10 and MRR.
- v0.50/v0.51 refute the stronger claim that temporal structure is the reason
  ConvMemory works. The learned temporal window contributes statistically, but
  its benefit is not temporally specific.
- External OOD results are mixed. The MuSiQue negative result is reported below
  because ConvMemory is not intended as a broad multi-hop document reranker.
- Latency numbers assume memory embeddings and memory-side indexes are already
  available. Cross-encoder timing includes pairwise scoring through the tested
  `CrossEncoder` path.

## LoCoMo: the headline comparison

Five split seeds: 7, 11, 23, 31, 47. Candidate pool: raw dense top-500.
This is the locked v363 table, and the source of the README figure.

| Method | R@10 | MRR | H@1 |
| --- | ---: | ---: | ---: |
| raw_dense | 0.5345 | 0.3254 | 0.1937 |
| ConvMemory v1 | 0.7798 | 0.5824 | 0.4440 |
| ConvMemory v1 + v2 | 0.7798 | 0.6560 | 0.5474 |
| mxbai CE top500 | 0.8080 | 0.6688 | 0.5646 |

Cost for the same paths, measured on the v362 RTX 4080 SUPER run with memory
embeddings precomputed:

| Path | ms/query |
| --- | ---: |
| ConvMemory v1 top500 | 16.8 |
| v1 + v2 evidence reranker | 28.6 |
| mxbai top500 CE | 1960.2 |

Raw dense latency was not measured in that run. It is cosine scoring over
precomputed embeddings, measured at 0.01-0.13 ms/query in the LongMemEval
harness below.

## LoCoMo: cross-encoder baselines

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

## LoCoMo: is a tuned non-neural baseline enough?

Same five seeds. The tuned heuristic combines dense retrieval, BM25/lexical
signals, temporal-neighbor propagation, and time decay, tuned on the dev split.

| Method | Recall@10 | Hit@10 | MRR |
|---|---:|---:|---:|
| convmemory_v040_full | 0.7798 +/- 0.0074 | 0.8350 | 0.5824 |
| tuned_dense_lexical_temporal_decay | 0.7234 +/- 0.0227 | 0.7757 | 0.4741 |
| dense_plus_bm25_lexical | 0.6473 +/- 0.0103 | 0.7034 | 0.4739 |
| raw_dense | 0.5345 +/- 0.0210 | 0.5894 | 0.3254 |

Reading: the learned reranker beats a carefully tuned heuristic, so the gain is
not just "add BM25 and time decay". See [NEGATIVE_RESULTS.md](NEGATIVE_RESULTS.md)
for the paired-bootstrap intervals.

## LongMemEval: the cost advantage

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

LongMemEval numbers are not seed-averaged: Clean500 is a single run and
Stress1000 is reported for a single seed (23). Read these as indicative
single-run retrieval-stage checks, not benchmark-grade comparisons.

Reading: ConvMemory reranks above BGE-large CE on these memory-family Recall@10
checks while being about 12-47x faster. It remains below mxbai accuracy, but is
about 28-117x lower latency in the tested settings.

Note the Stress1000 rows: as the candidate pool grows, raw dense recall drops
from 0.9049 to 0.5452 and the reranking stage becomes load-bearing. That is the
regime ConvMemory is designed for.

## Retrained ablation

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

## Attribution: the negative result

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

See [NEGATIVE_RESULTS.md](NEGATIVE_RESULTS.md) for the full v0.50/v0.51
interpretation, and [RESEARCH_TRAJECTORY.md](RESEARCH_TRAJECTORY.md) for what
was done afterwards.

## v2 load-bearing ablation

The v364 audit retrains the v2-style full-text arm in the same harness as the
ablation arms:

| Arm | FULL MRR |
| --- | ---: |
| v361 full text | 0.6677 |
| no_memory_text | 0.2966 |
| random_other_query_text | 0.2506 |
| shuffled_memory_text | 0.2731 |
| scalar_only | 0.5792 |

All three text perturbations fall below raw_dense MRR 0.3254. Token interaction
on candidate-specific memory text is doing the work, not scalar/rank/time
shortcuts. Full detail in [EVIDENCE_RERANKER.md](EVIDENCE_RERANKER.md).

## Strong-backbone retraining

Three split seeds. ConvMemory is retrained in each embedding space.

| Backbone | Raw Recall@10 | ConvMemory Recall@10 | Gain | ConvMemory MRR |
|---|---:|---:|---:|---:|
| BGE-large | 0.6680 +/- 0.0237 | 0.7726 +/- 0.0100 | +0.1046 +/- 0.0137 | 0.5639 +/- 0.0066 |
| E5-large | 0.7010 +/- 0.0216 | 0.7902 +/- 0.0171 | +0.0892 +/- 0.0052 | 0.5941 +/- 0.0103 |

Reading: ConvMemory gains are not just an artifact of a weak MPNet retriever.
Retraining on stronger embeddings still gives about +9 to +10 Recall@10 points.

## External OOD results

Single run per dataset. These are intentionally reported as mixed evidence.

| Dataset | Questions | ConvMemory R@10 | Raw dense | Dense + lexical | BM25 |
|---|---:|---:|---:|---:|---:|
| QMSum | 272 | 0.5882 | 0.4724 | 0.5423 | 0.5294 |
| MSC persona | 6155 | 0.9632 | 0.8375 | 0.9765 | 0.9920 |
| HotpotQA | 1000 | 0.7983 | 0.7682 | 0.8621 | 0.8280 |
| MuSiQue | 1000 | 0.7635 | 0.8640 | 0.8175 | 0.7245 |

These external OOD results are single runs without seed averaging or confidence
intervals; treat them as indicative scope checks, not benchmark-grade
comparisons.

Reading: ConvMemory wins on QMSum and improves strongly over raw dense on MSC,
but lexical/BM25 baselines dominate MSC's weak persona-overlap labels. On
HotpotQA, a trivial dense+lexical baseline is stronger. On MuSiQue, ConvMemory
regresses below raw dense. This is a scope boundary: ConvMemory is a memory
reranker, not a general multi-hop document reranker.

## Where ConvMemory fails

- Non-temporal multi-hop retrieval: MuSiQue is negative against raw dense.
- Lexically anchored document retrieval: HotpotQA favors dense+lexical scoring.
- Maximum top-rank precision: mxbai-rerank-large remains stronger on LoCoMo MRR.
- Small candidate pools: when a plain dense search over ~100 memories already
  puts the right memory in the top 5, a reranking stage has nothing to recover.
- Cross-query score calibration: scores should not be treated as calibrated
  confidence without application-specific validation.
- Mechanism attribution: v0.51 does not support temporal structure as the
  load-bearing explanation for ConvMemory's gain.

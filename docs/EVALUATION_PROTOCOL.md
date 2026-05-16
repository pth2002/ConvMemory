# Evaluation Protocol

This protocol records which evaluation items have been completed and how claims
should be reported.

Canonical audited summary:

- `remote_results_archive/2026-05-16_v047_v048/results/v047/V047_SUMMARY_REGENERATED.md`
- The old remote `results/v047/V047_SUMMARY.md` is deprecated because it was a
  broken `tabulate` import stub.

## Completion Status

A command appearing later in this document is a reproducibility recipe, not a
claim that every optional experiment has been completed.

| Item | Status | Evidence path | Notes |
|---|---|---|---|
| MPNet + LoCoMo simple baselines / feature masks, 5 seeds | Done | `results/v040/baselines_ablation_stats_mpnet` | In-domain checkpoint evaluation. |
| MPNet + MiniLM-L6 LoCoMo CE top500, 5 seeds | Done | `results/v040/minilm_ce_top500_5seed_full` | Includes paired bootstrap for ConvMemory vs MiniLM CE. |
| LoCoMo stronger rerankers, 5 seeds | Done | `remote_results_archive/.../results/v047/strong_rerankers` | Includes BGE-base, BGE-large, Jina trust, and mxbai. |
| Jina reranker trust-mode audit | Done | `remote_results_archive/.../results/v047/strong_rerankers/jina_reranker_v2_base_multilingual_trust` | Non-trust Jina failed; trust-enabled runs completed. |
| MPNet + LoCoMo order robustness, 5 seeds | Done | `results/v041/order_robustness_mpnet` | Synthetic order perturbations. |
| MPNet + LoCoMo error analysis / calibration bins, 5 seeds | Done | `results/v042/error_calibration_mpnet` | Diagnostic confidence features. |
| Post-hoc confidence calibration | Done | `results/v046/confidence_calibration_mpnet` | Calibration is measured, not production-certified. |
| MPNet + LongMemEval-S clean fixed 500 questions | Done | `remote_results_archive/.../results/v047/longmemeval_strong_ce` | Includes BGE-large and mxbai CE checks. |
| MPNet + LongMemEval-S 1000-session stress | Partially done | `remote_results_archive/.../results/v047/longmemeval_strong_ce` | Strong-CE stress checks are seed23 only. Earlier MiniLM stress was 5-seed. |
| External OOD: QMSum, MSC, HotpotQA, MuSiQue | Done | `remote_results_archive/.../results/v047/external_ood` | Single run each; report as mixed evidence. |
| Retrained feature ablation | Done | `remote_results_archive/.../results/v048/retrained_ablation_3seed` | 3 seeds; includes no-temporal, no-lexical, no-router. |
| BGE-large and E5-large backbone retraining | Done | `remote_results_archive/.../results/v048/backbone_3seed_summary` | 3 seeds each; model retrained in each embedding space. |
| End-to-end answer generation evaluation | Pending | none | Retrieval-stage results do not prove QA improvements. |
| Production calibration / abstention thresholding | Pending | none | Needs application-specific validation. |

## 1. Reporting Scope

Do not present LoCoMo as an out-of-domain result when using the public
LoCoMo-trained checkpoint.

Recommended table layout:

| Dataset | Split type | Training overlap | Purpose |
|---|---|---|---|
| LoCoMo | conversation-level test split | in-domain | checkpoint sanity and main memory-retrieval check |
| LongMemEval-S | same-family OOD | no LongMemEval training | long-memory transfer and cost comparison |
| QMSum / MSC | conversation-style external checks | no task-specific training | memory-like OOD signal |
| HotpotQA / MuSiQue | non-temporal document-style checks | no task-specific training | scope-boundary evidence |

External OOD results should be described as mixed. The MuSiQue negative result
must be shown when discussing generalization.

## 2. Multi-Seed Statistics

Use multi-seed mean/std for headline in-domain claims:

```bash
python experiments/v040_baselines_ablation_stats.py \
  --device cuda \
  --seeds 7 11 23 31 47 \
  --bootstrap-samples 10000 \
  --out results/v040/baselines_ablation_stats
```

This writes:

- `detailed.csv`
- `summary_by_seed.csv`
- `summary.csv`
- `paired_bootstrap.csv`
- `REPORT.md`

Headline claims should use mean/std and paired bootstrap intervals when
available, not single numbers.

## 3. Simple Baselines

v0.40 includes:

- raw dense retrieval;
- BM25;
- recency-weighted dense retrieval;
- dense + lexical score fusion;
- dense + lexical RRF;
- dense + lexical + temporal-window RRF.

These are the first baselines to check before claiming that neural temporal
reranking is necessary.

## 4. Retrained Ablations

The v0.48 retrained ablation matrix is now the preferred ablation evidence.
Inference-time feature masks from v0.40 remain diagnostic, but should not be
used as the primary architecture claim.

Completed v0.48 variants:

- `full_control`
- `no_router`
- `no_temporal_w1`
- `no_lexical`
- `no_lexical_no_router`

Interpretation:

- lexical features are the largest contributor;
- temporal windowing is real but secondary;
- the router/DCA scalar contributes approximately zero and should not be
  presented as a feature.

## 5. Stronger Cross-Encoder Baselines

The original MiniLM comparison is useful but insufficient for a modern reranker
claim. v0.47 completed stronger LoCoMo baselines:

- `BAAI/bge-reranker-base`
- `jinaai/jina-reranker-v2-base-multilingual` with `trust_remote_code=True`
- `BAAI/bge-reranker-large`
- `mixedbread-ai/mxbai-rerank-large-v1`

Claim boundary:

- ConvMemory beats BGE-reranker-base/large on LoCoMo Recall@10.
- ConvMemory loses to mxbai on LoCoMo Recall@10 and MRR.
- Do not write an overall cross-encoder superiority claim for ConvMemory.

## 6. Embedding Backbones

Backbone robustness should use retrained checkpoints, not simply plugging a new
embedding model into an MPNet-trained checkpoint.

Completed v0.48 retraining:

- BGE-large, 3 seeds
- E5-large, 3 seeds

Reading:

- gains remain about +9 to +10 Recall@10 points over raw dense retrieval;
- gains shrink as the base retriever becomes stronger, which is expected.

## 7. Order Robustness

Run:

```bash
python experiments/v041_order_robustness.py \
  --device cuda \
  --seeds 7 11 23 31 47 \
  --out results/v041/order_robustness
```

This evaluates original order, partial shuffles, full shuffle, block shuffle,
and reverse order. Treat this as synthetic perturbation evidence, not a full
production timestamp-noise study.

## 8. Error Analysis And Calibration

Run:

```bash
python experiments/v042_error_calibration.py \
  --device cuda \
  --seeds 7 11 23 31 47 \
  --out results/v042/error_calibration
```

This writes:

- ConvMemory wins over raw dense;
- ConvMemory losses against raw dense;
- per-question deltas;
- calibration bins for top-score and margin features.

Calibration is measured but not fixed. Do not present ConvMemory scores as
cross-query calibrated confidence without application-specific calibration.

## 9. Latency Fairness

Report:

- device;
- encoder batch size;
- cross-encoder batch size;
- whether tokenization is included;
- P50 / P95 / P99 when available;
- queries per second or ms/query;
- whether embeddings and memory-side indexes are cached.

Do not compare cached ConvMemory against an unoptimized production
cross-encoder serving stack without clearly labeling the limitation.

## 10. Claim Policy

Safe claim:

> ConvMemory is a lightweight temporal memory reranker that improves
> recall-oriented memory selection over raw dense retrieval on session-structured
> memory tasks, with much lower latency than full top500 modern cross-encoder
> reranking in the tested memory-family settings.

Avoid:

> ConvMemory has broad leaderboard leadership.
> ConvMemory has broad cross-encoder superiority.
> ConvMemory is proven robust across unrelated retrieval datasets.
> The retired router scalar is a core contributor.

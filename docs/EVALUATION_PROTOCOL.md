# Evaluation Protocol

This protocol turns the main review concerns into concrete experiments.

## Completion Status

This table distinguishes experiments that have been run from protocol items
that are still planned. A command appearing later in this document is not a
claim that the result has already been completed.

| Item | Status | Evidence path | Notes |
|---|---|---|---|
| MPNet + LoCoMo simple baselines / feature masks, 5 seeds | Done | `results/v040/baselines_ablation_stats_mpnet` | In-domain checkpoint evaluation. |
| MPNet + LoCoMo order robustness, 5 seeds | Done | `results/v041/order_robustness_mpnet` | Synthetic order perturbations. |
| MPNet + LoCoMo error analysis / calibration bins, 5 seeds | Done | `results/v042/error_calibration_mpnet` | Diagnostic confidence features. |
| Post-hoc confidence calibration | Done | `results/v046/confidence_calibration_mpnet` | Platt/logistic and isotonic calibration over v0.42 cases. |
| MPNet + MiniLM-L6 LoCoMo CE top500, 5 seeds | Done | `results/v040/minilm_ce_top500_5seed_full` | Includes paired bootstrap for ConvMemory vs MiniLM CE. |
| MPNet + LongMemEval-S clean fixed 500 questions | Done | `results/v044/longmemeval_clean_fixed500` | Same-family OOD; fixed dataset, so bootstrap is more meaningful than split seeds. |
| MPNet + LongMemEval-S 1000-session stress, 5 seeds | Done | `results/v044/longmemeval_stress1000_seed*` | Seed controls distractor sampling. |
| Generic synthetic agent scratchpad via v0.43 | Done | `results/v043/generic_retrieval_eval/synthetic_agent_scratchpad` | Synthetic external-format check; not a public benchmark. |
| BGE / Jina / mxbai stronger rerankers | Pending | none | Requires model artifacts; the current remote has no network access. |
| BGE-large or E5 embedding-backbone checkpoints | Pending | none | Requires retraining ConvMemory in that embedding space for a fair claim. |
| Retrained no-temporal ablation checkpoint | Pending | none | Inference-time masking is complete; retrained ablation remains a larger follow-up. |
| Additional public OOD dataset, e.g. MSC/QMSum/HotpotQA/MuSiQue | Pending | none | Requires dataset download/conversion to v0.43 JSONL. |

## 1. In-Domain And OOD Reporting

Do not present LoCoMo as an out-of-domain result when using the public LoCoMo-trained checkpoint.

Recommended table layout:

| Dataset | Split type | Training overlap | Purpose |
|---|---|---|---|
| LoCoMo | conversation-level test split | in-domain | checkpoint sanity and main memory-retrieval check |
| LongMemEval-S | same-family OOD | no LongMemEval training | long-memory transfer check |
| Converted JSONL datasets | external OOD | dataset-specific | broader robustness check |

Use `experiments/v043_generic_retrieval_eval.py` for converted datasets.

## 2. Multi-Seed Statistics

Run:

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

Headline claims should use mean/std and paired bootstrap intervals, not single numbers.

## 3. Simple Baselines

v0.40 includes:

- raw dense retrieval;
- BM25;
- recency-weighted dense retrieval;
- dense + lexical score fusion;
- dense + lexical RRF;
- dense + lexical + temporal-window RRF.

These are the first baselines to check before claiming that neural temporal reranking is necessary.

## 4. Feature Ablations

v0.40 includes inference-time feature masking:

- full ConvMemory;
- no temporal-window score;
- no lexical features;
- no router feature;
- no raw dense feature;
- temporal-window score only;
- full global windows vs candidate-local windows.

These are not a replacement for retrained ablations, but they identify which signals the current checkpoint depends on.

## 5. Stronger Cross-Encoder Baselines

The default public benchmark uses `cross-encoder/ms-marco-MiniLM-L-6-v2`. This is not enough for a strong research claim.

Run v0.40 with additional rerankers:

```bash
python experiments/v040_baselines_ablation_stats.py \
  --device cuda \
  --seeds 7 11 23 31 47 \
  --cross-encoder-models cross-encoder/ms-marco-MiniLM-L-6-v2,BAAI/bge-reranker-base,BAAI/bge-reranker-large
```

Large rerankers may require smaller `--cross-batch-size`.

## 6. Embedding Backbones

Run v0.40 separately for each embedding model:

```bash
python experiments/v040_baselines_ablation_stats.py \
  --device cuda \
  --encoder-model sentence-transformers/all-mpnet-base-v2 \
  --embedding-cache results/cache/mpnet_embeddings.sqlite

python experiments/v040_baselines_ablation_stats.py \
  --device cuda \
  --encoder-model BAAI/bge-large-en-v1.5 \
  --embedding-cache results/cache/bge_large_embeddings.sqlite

python experiments/v040_baselines_ablation_stats.py \
  --device cuda \
  --encoder-model intfloat/e5-large-v2 \
  --embedding-cache results/cache/e5_large_embeddings.sqlite
```

The MPNet checkpoint may not be optimal in another embedding space. Treat non-MPNet runs as robustness checks unless the model is retrained.

## 7. Order Robustness

Run:

```bash
python experiments/v041_order_robustness.py \
  --device cuda \
  --seeds 7 11 23 31 47 \
  --out results/v041/order_robustness
```

This evaluates original order, partial shuffles, full shuffle, block shuffle, and reverse order.

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

## 9. Latency Fairness

Use `experiments/v036_latency_benchmark.py`.

Report:

- device;
- encoder batch size;
- cross-encoder batch size;
- whether tokenization is included;
- P50 / P95 / P99;
- queries per second;
- whether embeddings and memory-side indexes are cached.

Do not compare cached ConvMemory against an unoptimized production cross-encoder serving stack without clearly labeling the limitation.

## 10. Claim Policy

Safe claim:

> ConvMemory is an early lightweight temporal memory reranker with preliminary evidence that it improves recall-oriented memory selection over raw dense retrieval and can serve as a candidate stage before a small cross-encoder pass.

Avoid until the protocol above is fully run:

> ConvMemory is state of the art.
> ConvMemory generally outperforms cross-encoders.
> ConvMemory is proven robust across memory datasets.

# Training ConvMemory

This document describes how to reproduce or retrain the public LoCoMo MPNet checkpoint.

## Public Training Entry Point

```bash
python experiments/train_locomo.py \
  --device cuda \
  --data data/locomo10.json \
  --encoder-model sentence-transformers/all-mpnet-base-v2 \
  --embedding-cache results/cache/mpnet_embeddings.sqlite \
  --embedding-cache-key sentence-transformers/all-mpnet-base-v2 \
  --teacher-cache results/cache/repro_teacher_mpnet_top500_seed23.json \
  --raw-top-n 500 \
  --candidate-top-n 500 \
  --cross-top-n 500 \
  --epochs 1 \
  --seed 23 \
  --save-pretrained checkpoints/convmemory-locomo-mpnet \
  --out results/train_locomo_seed23
```

`experiments/train_locomo.py` is a public alias over the tested implementation in `experiments/reproduce_locomo.py`. The training and reproduction paths intentionally share code so exported checkpoints and reported evaluations use the same data preparation.

## Data Split

The current LoCoMo scripts split by conversation/sample id through `choose_split(...)`, not by random QA rows. This reduces leakage where QA pairs from the same conversation appear in both dev and test.

By default, `--dev-ratio 0.5` is used by the evaluation helpers: conversation/sample ids are shuffled with the requested seed, half are assigned to the dev/training side, and the remaining ids are held out for test. The split is therefore conversation-level rather than QA-row-level. Before reporting a new checkpoint, verify that no test `question_id` sample prefix appears in the train/dev set for the same seed.

Recommended reporting:

- run at least five split seeds;
- report mean and standard deviation;
- report paired bootstrap confidence intervals for key comparisons;
- tune hyperparameters on the dev split only.

## Supervision

The training loop combines:

- gold evidence ids from LoCoMo annotations;
- cross-encoder teacher scores over dense-retrieved candidates;
- pairwise ranking loss;
- optional first-rank supervision.

The current public checkpoint uses:

- embedding model: `sentence-transformers/all-mpnet-base-v2`;
- window size: 5;
- stride: 1;
- kernel size: 3;
- candidate top-n: 500;
- raw score fusion weight: 0.025;
- lexical features: enabled;
- legacy router/DCA scalar slot: present in the historical checkpoint, but
  v0.48 retrained ablation shows no measurable contribution. Do not tune or
  report it as a model feature.

## Reproducibility Notes

For stable runs:

- cache MPNet embeddings with `--embedding-cache`;
- cache teacher cross-encoder scores with `--teacher-cache`;
- keep the data split seed in the output directory name;
- write detailed per-question CSV files for later paired tests.

## Stronger Evaluation Protocol

The following scripts harden the evaluation beyond the original checkpoint reproduction:

```bash
python experiments/v040_baselines_ablation_stats.py --device cuda --seeds 7 11 23 31 47
python experiments/v041_order_robustness.py --device cuda --seeds 7 11 23 31 47
python experiments/v042_error_calibration.py --device cuda --seeds 7 11 23 31 47
```

For stronger cross-encoder baselines, pass model ids to v0.40:

```bash
python experiments/v040_baselines_ablation_stats.py \
  --device cuda \
  --seeds 7 11 23 31 47 \
  --cross-encoder-models cross-encoder/ms-marco-MiniLM-L-6-v2,BAAI/bge-reranker-base,BAAI/bge-reranker-large
```

The v0.47/v0.48 audit extended this protocol with BGE, Jina trust-mode, and
mxbai cross-encoder baselines, retrained no-temporal/no-lexical/no-router
ablations, and BGE-large/E5-large ConvMemory retraining.

For additional embedding backbones, retrain ConvMemory in the target embedding
space. A checkpoint trained for MPNet should not be assumed optimal for other
embedding spaces.

The old remote `results/v047/V047_SUMMARY.md` is deprecated because it was a
broken `tabulate` import stub. Use the regenerated audited summary instead:
`remote_results_archive/2026-05-16_v047_v048/results/v047/V047_SUMMARY_REGENERATED.md`.

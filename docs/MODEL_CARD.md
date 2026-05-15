# ConvMemory LoCoMo MPNet Model Card

This model card covers the public `convmemory-locomo-mpnet` checkpoint.

## Model Summary

- Model type: lightweight temporal memory reranker.
- Embedding backbone: `sentence-transformers/all-mpnet-base-v2`.
- Input unit: ordered memory records, usually conversation turns or agent memory entries.
- Output: reranked memory ids and scores.
- Intended use: retrieval-stage reranking for long-term conversational or agent memory.
- Not intended for: general web search, document ranking without temporal structure, or end-to-end answer generation by itself.

## Architecture

The checkpoint combines:

- temporal Conv/Mixer window encoding over ordered memory embeddings;
- query-memory interaction features;
- raw dense retrieval features;
- candidate-local window features;
- lightweight lexical overlap features;
- a block-level router feature.

Checkpoint configuration:

| Field | Value |
|---|---:|
| Embedding dimension | 768 |
| Window size | 5 |
| Stride | 1 |
| Kernel size | 3 |
| Hidden dimension | 256 |
| Token MLP dimension | 32 |
| Channel MLP dimension | 512 |
| Extra scalar features | 5 |
| Candidate top-n | 500 |
| Raw score fusion weight | 0.025 |
| Router block size | 32 |
| Parameters | 3,648,118 |

Parameter breakdown:

| Component | Parameters |
|---|---:|
| Temporal Conv/Mixer encoder | 2,825,589 |
| CE-lite scorer | 822,529 |
| Total | 3,648,118 |

## Training Data

The public checkpoint was trained on LoCoMo-style long-term conversational memory data.

Important caveat: LoCoMo benchmark results are in-domain for this checkpoint. They should not be presented as broad out-of-domain generalization results.

## Training Objective

The training pipeline supports:

- gold evidence supervision from annotated memory ids;
- cross-encoder teacher scores over dense-retrieved candidates;
- pairwise ranking loss between stronger and weaker candidates;
- optional first-rank supervision.

The public training entrypoint is:

```bash
python experiments/train_locomo.py
```

See [TRAINING.md](TRAINING.md) for a complete command.

## Evaluation Scope

Current public results are retrieval-stage evaluations. They measure whether evidence memory ids are retrieved into the top-k list. They do not measure full answer generation.

Known evaluation gaps:

- out-of-domain coverage is still limited to same-family LongMemEval-S plus a
  synthetic agent-scratchpad sanity check;
- limited cross-encoder baselines;
- limited embedding backbone coverage;
- incomplete trained-ablation matrix;
- paired significance tests are available for the main LoCoMo and LongMemEval-S
  comparisons, but not for every exploratory experiment;
- order robustness has been tested with synthetic perturbations on LoCoMo, but
  real missing/noisy timestamp behavior still needs broader validation.

The v0.40-v0.43 scripts were added to address these gaps systematically.

## Limitations

- Memory order matters. If timestamps are missing or severely corrupted, quality may degrade.
- Scores are not calibrated by default. A post-hoc confidence calibration script
  is provided in `experiments/v046_calibrate_confidence.py`, but production
  thresholding should be validated on application data.
- The public checkpoint is optimized for the MPNet embedding space; other embedding backbones require retraining or at least careful validation.
- The model is a reranker, not a vector database and not a full QA system.
- Cascade fusion with a cross-encoder is currently research-preview code, not a stable public API.

## Responsible Use

ConvMemory retrieves memory text that may contain sensitive user information. Applications should apply their own privacy, retention, and access-control policies before storing or surfacing memories.

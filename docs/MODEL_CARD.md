# ConvMemory LoCoMo MPNet Model Card

This model card covers the public `convmemory-locomo-mpnet` checkpoint.

## Model Summary

- Model type: lightweight temporal memory reranker.
- Approximate size: 3.6M parameters.
- Embedding backbone: `sentence-transformers/all-mpnet-base-v2`.
- Input unit: ordered memory records, usually conversation turns or agent memory entries.
- Output: reranked memory ids and scores.
- Intended use: retrieval-stage reranking for long-term conversational or agent memory.
- Not intended for: general web search, document ranking without temporal structure, or end-to-end answer generation by itself.

## Architecture

The checkpoint combines:

- temporal Conv/Mixer window encoding over ordered memory embeddings;
- query-memory interaction features;
- raw dense retrieval score features;
- candidate-local window features;
- lightweight lexical overlap features;
- a CE-lite scorer over embedding-level and scalar features.

Hardened v0.48 retrained ablations show that lexical features are the largest
contributor and temporal windowing is a real but smaller contributor. A legacy
router/DCA scalar was ablated and showed no measurable benefit; it should not be
treated as an architecture selling point.

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
| Candidate top-n | 500 |
| Raw score fusion weight | 0.025 |
| Parameters | 3,648,118 |

Parameter breakdown:

| Component | Parameters |
|---|---:|
| Temporal Conv/Mixer encoder | 2,825,589 |
| CE-lite scorer | 822,529 |
| Total | 3,648,118 |

## Training Data

The public checkpoint was trained on LoCoMo-style long-term conversational
memory data.

Important caveat: LoCoMo benchmark results are in-domain for this checkpoint.
They should not be presented as broad out-of-domain generalization results.

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

Current public results are retrieval-stage evaluations. They measure whether
evidence memory ids are retrieved into the top-k list. They do not measure full
answer generation.

Canonical audited summary:

- `remote_results_archive/2026-05-16_v047_v048/results/v047/V047_SUMMARY_REGENERATED.md`
- The old remote `results/v047/V047_SUMMARY.md` is deprecated because it was a
  broken `tabulate` import stub.

Hardened result summary:

- LoCoMo strong-CE comparison:
  - ConvMemory is competitive on Recall@10.
  - It beats BGE-reranker-base/large on Recall@10.
  - It loses to `mxbai-rerank-large-v1` on Recall@10 and MRR.
- Retrained ablation:
  - lexical features are the dominant contributor;
  - temporal windowing contributes a smaller but real gain;
  - the router/DCA scalar contributes approximately zero.
- Strong-backbone retraining:
  - BGE-large and E5-large retrained checkpoints still gain about +9 to +10
    Recall@10 points over raw dense retrieval.
- External OOD:
  - QMSum is positive;
  - MSC improves over raw dense but lexical/BM25 baselines dominate the weak labels;
  - HotpotQA favors dense+lexical scoring;
  - MuSiQue is negative against raw dense.
- LongMemEval strong-CE checks:
  - ConvMemory is much lower latency than BGE-large and mxbai cross-encoders in
    the tested memory-family settings;
  - mxbai remains stronger in accuracy.

## Limitations

- ConvMemory is a memory reranker, not a vector database and not a full QA system.
- It should not be given an overall cross-encoder superiority claim.
- It is not a broad document reranker; MuSiQue shows a clear negative result.
- Memory order matters. If timestamps are missing or severely corrupted, quality may degrade.
- Scores are not calibrated by default. Production thresholding should be validated on application data.
- The public checkpoint is optimized for the MPNet embedding space; other embedding backbones require retraining or careful validation.
- Cascade fusion with a cross-encoder is currently research-preview code, not a stable public API.

## Responsible Use

ConvMemory retrieves memory text that may contain sensitive user information.
Applications should apply their own privacy, retention, and access-control
policies before storing or surfacing memories.

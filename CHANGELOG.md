# Changelog

## Unreleased

- Note: v0.40-v0.51 are internal evaluation-iteration identifiers for
  hardening experiments, not packaged PyPI releases. The installable package
  version remains 0.3.0 and the public checkpoint is unchanged.

- Re-based the documentation onto the v0.50/v0.51 negative result: the learned
  reranker remains useful, but temporal structure is not supported as the
  load-bearing mechanism.
- Added `docs/NEGATIVE_RESULTS.md` with the tuned-heuristic gate, five-seed
  retrained attribution table, and scope of the temporal-mechanism refutation.
- Hardened the public result claims against the audited v0.47/v0.48 evaluation archive.
- Retired the previous overbroad cross-encoder framing; `mxbai-rerank-large-v1`
  outperforms ConvMemory on LoCoMo Recall@10 and MRR.
- Added strong-CE LoCoMo reporting for BGE-base, Jina trust-mode, BGE-large, and mxbai.
- Added retrained ablation conclusions: lexical features dominate, the
  temporal-window effect is not temporally specific in v0.51, and the
  router/DCA scalar has approximately zero contribution.
- Added BGE-large and E5-large backbone-retraining results.
- Added external OOD reporting with mixed outcomes, including the negative
  MuSiQue result as scope-boundary evidence.
- Featured LongMemEval strong-CE cost comparisons as the main practical value
  proposition.
- Deprecated the broken remote `results/v047/V047_SUMMARY.md` in favor of the
  regenerated audited summary.
- Added a stricter evaluation protocol for ConvMemory results.
- Added multi-seed LoCoMo baseline, feature-ablation, order-robustness, and calibration scripts.
- Added full five-seed MiniLM cross-encoder comparison for LoCoMo top500 reranking.
- Added LongMemEval-S clean and 1000-session stress evaluations.
- Added post-hoc confidence calibration script.
- Added model card and public training documentation.
- Rewrote the README to separate supported claims from open research items.
- Fixed `v040_baselines_ablation_stats.py` CSV writing so mixed method schemas, including cross-encoder metadata fields, are handled correctly.

## 0.3.0

- Added compression-aware routing utilities.
- Added context expansion mode for agent memory pipelines.
- Added cascade-fusion research scripts for ConvMemory plus a small cross-encoder pass.

## 0.1.0

- Initial research-preview release.
- Added the public LoCoMo MPNet checkpoint as a GitHub release asset.

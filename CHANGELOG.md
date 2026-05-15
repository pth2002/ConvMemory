# Changelog

## Unreleased

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

# Changelog

## 0.6.2 - 2026-07-10

- Added `validity_source_map` so applications can pass one selected update
  source per target directly into the v3 context/demotion path.
- Replaced unbounded validity pair scoring with a bounded top-1 source policy;
  the built-in fallback now scores at most one source per protected target.
- Limited integrated validity processing to the requested result prefix and
  made `demote` preserve that top-k candidate set by construction.
- Applied validity after final context expansion so every selected context item
  follows the same annotation/demotion contract.
- Added later-source validation, Chinese-aware lexical fallback features, and
  package tests for source budgets, explicit evidence, and top-k preservation.

## 0.6.1 - 2026-07-03

- Added `OPCConvMemory`, an OPC-style Chinese ConvMemory student that keeps the
  ConvMemory window encoder and CE-lite reranker while using a single
  Chinese-specialized BGE encoder.
- Added `SingleSpaceTextEncoder` for BGE-backed Chinese checkpoints.
- Added package tests for loading and scoring an OPC-style checkpoint without
  requiring a live Hugging Face download.
- Published the matching checkpoint on Hugging Face Hub:
  `Purdy0228/ConvMemory-OPC-Student-BGE`.
- Added OPC-v3 validity context documentation and the matching Hugging Face
  checkpoint:
  `Purdy0228/ConvMemory-OPC-V3-Validity-Context`.

## 0.6.0 - 2026-06-09

- Added ConvMemory v3 validity context layer API support:
  `convmemory.validity` with `ValidityEvidenceModule`,
  `ValidityEvidenceConfig`, and `ValidityAnnotation`.
- Added CrossEncoder-backed validity scoring with the v506/v511
  query/source/target demotion format, including binary scorer support via
  `cross_encoder_num_labels`.
- Added batched explicit evidence scoring via
  `ValidityEvidenceModule.score_evidence_pairs(...)` for packaged v3
  query/source/target demotion scorers.
- Added `ConvMemory.attach_validity_module` and
  `ConvMemory.load_validity_module`; added `validity_mode` on retrieve/rerank
  paths with `off`, `context`, and opt-in `demote` semantics.
- Added tests for byte-identical off mode, context-mode rank preservation,
  demote candidate-set preservation, forbidden-field rejection, safe evidence
  output, invalid mode rejection, and save/load round-trip.
- Added `docs/VALIDITY_CONTEXT.md` and README entry documenting v3 as a
  validity context layer, not a default automatic graph-demotion system.
- Added `docs/V3_MODEL_CARD.md` with method-level, checkpoint-level, and
  package-level provenance plus a source-of-truth ledger.
- Added `examples/v3_validity_context_demo.py` as a no-download API shape demo.

## 0.5.0 - 2026-05-28

- Added ConvMemory v2 evidence reranker: protected top-10 token-evidence
  cross-encoder, recall-preserving over v1; opt-in API + tests + training
  recipe + load-bearing ablation backing.
- Added `convmemory.evidence_reranker` module (`EvidenceReranker`,
  `EvidenceRerankerConfig`, `FORBIDDEN_FIELDS`).
- Added `ConvMemory.attach_evidence_reranker` and
  `ConvMemory.load_evidence_reranker`; added `evidence_reranker="v2"` opt-in
  kwarg on `rerank`, `retrieve`, and `rerank_embeddings`.
- Added tests covering anti-leak field rejection, default-behavior
  byte-identity vs 0.4.0, recall preservation, and save/load round-trip.
- Added `docs/EVIDENCE_RERANKER.md` with v363 headline numbers and v364
  load-bearing ablation summary.
- Published the v0.5.0 evidence reranker checkpoint on Hugging Face Hub:
  `Purdy0228/ConvMemory-v2-Evidence-Reranker`.
- Added `examples/train_evidence_reranker.py` and
  `examples/v2_evidence_reranker_demo.py`.
- Renamed the experimental Memory-MLA expander away from the misleading `v2`
  label; module name unchanged for backward compat, documentation updated to
  call it `Memory-MLA Recall Expander`.

## 0.4.0 - 2026-05-21

- Added Hugging Face Hub checkpoint loading for `ConvMemory.from_pretrained`
  and `ConvMemory.load_ccge_editor`; Hub repo ids such as
  `Purdy0228/ConvMemory-LoCoMo-MPNet` now resolve through
  `huggingface_hub.snapshot_download` when no local path exists.
- Added the optional `convmemory.hub` resolver, `hub` extra, and tests that mock
  Hub downloads without network access.
- Published the base LoCoMo/MPNet ConvMemory checkpoint on Hugging Face Hub:
  `Purdy0228/ConvMemory-LoCoMo-MPNet`.
- Added `tests/` smoke suite and `.github/workflows/ci.yml`, so the README CI
  badge now points to a real workflow.
- Added `docs/RELEASE.md` with packaging and release instructions.
- Added `examples/ccge_la_with_checkpoint.py` as a real-checkpoint CCGE-LA
  demonstration.
- Added minimum dependency versions in `pyproject.toml` and `requirements.txt`.
- Added a warning when attaching or loading a CCGE-LA editor trained for a
  different embedding backbone.
- Changed package classifier from `Development Status :: 3 - Alpha` to
  `Development Status :: 4 - Beta`.
- Changed, breaking for the alpha API only: `ConvMemory.from_pretrained`
  `load_ccge` now defaults to `False` so CCGE-LA loading is explicit opt-in.
- Changed, breaking for the alpha API only: `editor=` now accepts only `None`,
  `"ccge_la"`, or a `CCGELowAmplitudeEditor` instance. Legacy spellings such as
  `True`, `False`, `""`, `"none"`, `"convmemory"`, `"ccge"`, and `"ccge-la"`
  now raise `ValueError`.
- Documented public `ConvMemory` methods with API-focused docstrings.
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
- Fixed `v040_baselines_ablation_stats.py` CSV writing so mixed method schemas,
  including cross-encoder metadata fields, are handled correctly.

## ccge-la-alpha-v0.1 - 2026-05-21

- Added public alpha API exports: `CCGELowAmplitudeEditor`, `CCGEConfig`,
  `CCGEFeatureBatch`, `build_ccge_features`,
  `multi_positive_retrieval_loss`, and `rank_candidates`.
- Added `ConvMemory.attach_ccge_editor` and `ConvMemory.load_ccge_editor`.
- Added `editor=` and `ccge_top_n=` parameters to `rerank`, `retrieve`,
  `expand_context`, `rerank_embeddings`, and `expand_context_embeddings`.
  They are disabled by default and preserve prior behavior unless explicitly
  enabled.
- Added `load_ccge` to `ConvMemory.from_pretrained`.
- Published the alpha LoCoMo/MPNet seed-23 editor checkpoint with SHA256
  `459ecfb2b4c35887f1d8f2cdd87dab402c37bd8dee86628655eff08f314b2e7c`.
- Marked the CCGE-LA release as alpha: the interface may receive small changes,
  and the checkpoint is a single seed-23 point.

## 0.3.0

- Added compression-aware routing utilities.
- Added context expansion mode for agent memory pipelines.
- Added cascade-fusion research scripts for ConvMemory plus a small cross-encoder pass.

## 0.1.0

- Initial research-preview release.
- Added the public LoCoMo MPNet checkpoint as a GitHub release asset.

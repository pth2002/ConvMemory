# Negative Results: Temporal Attribution

This note records the v0.50/v0.51 hardening result that changed the project
interpretation. ConvMemory remains a useful learned reranker, but the original
"CNN temporal compression is why it works" hypothesis is not supported by the
current evidence.

## Sources

Authoritative local artifacts:

- `results/v050/tuned_heuristic_fusion_full/REPORT.md`
- `results/v051/temporal_attribution_5seed/REPORT.md`
- `results/v051/temporal_attribution_5seed/paired_bootstrap.csv`
- `results/v051/temporal_attribution_5seed/summary.csv`
- `EXPERIMENT_LOG.md`

Version labels such as v0.50 and v0.51 are internal evaluation-iteration
identifiers, not packaged PyPI releases. As of package version 0.4.0, the public
base checkpoint is unchanged.

## v0.50: Tuned Heuristic Gate

The first gate asked whether a carefully tuned non-neural baseline could close
the ConvMemory gap. The tuned heuristic combined dense retrieval, BM25/lexical
signals, temporal-neighbor propagation, and time-decay. It was tuned on the dev
split and evaluated on the held-out split across five seeds: 7, 11, 23, 31, 47.

| Method | Seeds | Recall@10 | Hit@10 | MRR |
|---|---:|---:|---:|---:|
| convmemory_v040_full | 5 | 0.7798 +/- 0.0074 | 0.8350 | 0.5824 |
| tuned_dense_lexical_temporal_decay | 5 | 0.7234 +/- 0.0227 | 0.7757 | 0.4741 |
| dense_plus_bm25_lexical | 5 | 0.6473 +/- 0.0103 | 0.7034 | 0.4739 |
| raw_dense | 5 | 0.5345 +/- 0.0210 | 0.5894 | 0.3254 |

Paired bootstrap showed ConvMemory above the tuned heuristic:

| Baseline | Contender | Metric | Delta | 95% CI |
|---|---|---|---:|---:|
| tuned_dense_lexical_temporal_decay | convmemory_v040_full | Recall@10 | +0.0578 | [+0.0488, +0.0662] |
| tuned_dense_lexical_temporal_decay | convmemory_v040_full | MRR | +0.1094 | [+0.1007, +0.1179] |

This result keeps the learned reranker alive as a useful model. It does not
prove that temporal structure is the cause of the gain.

## v0.51: Five-Seed Retrained Attribution

The second gate compared the full retrained model with a no-temporal retrained
variant on the same seeds and question units. This isolates the learned
temporal-window component more directly than inference-time masking.

Five split seeds: 7, 11, 23, 31, 47.

| Method | Recall@10 |
|---|---:|
| full_control_retrained | 0.7432 +/- 0.0207 |
| no_temporal_w1_retrained | 0.7054 +/- 0.0221 |
| tuned_heuristic | 0.7234 +/- 0.0227 |
| raw_dense | 0.5345 +/- 0.0210 |

The full retrained model beats the tuned heuristic on aggregate:

| Comparison | Metric | Delta | 95% CI |
|---|---|---:|---:|
| full_control_retrained - tuned_heuristic | Recall@10 | +0.0199 | [+0.0105, +0.0283] |
| full_control_retrained - tuned_heuristic | MRR | +0.0566 | [+0.0487, +0.0645] |

## Slice Attribution

The critical test is whether the full minus no-temporal gain is larger on
temporal proxy slices than on hard non-temporal controls.

Paired bootstrap, `full_control_retrained - no_temporal_w1_retrained`,
Recall@10:

| Slice | Delta | 95% CI | Reading |
|---|---:|---:|---|
| ALL | +0.0376 | [+0.0306, +0.0451] | significant |
| T_SUP_auto | +0.0407 | [+0.0219, +0.0603] | significant; open question |
| T_REQUIRED_auto | +0.0252 | [+0.0139, +0.0363] | significant |
| T_HOP_auto | +0.0096 | [-0.0037, +0.0230] | not significant |
| OTHER | +0.0868 | [+0.0672, +0.1045] | significant |
| HARD_NON_TEMPORAL_auto | +0.0838 | [+0.0650, +0.1040] | significant |

## Interpretation

The temporal-window component contributes statistically on the aggregate test
set. It also contributes on T_SUP_auto, which remains the one open question for
future manual audit.

However, the effect is not temporally specific:

- the largest gains are on `OTHER` and `HARD_NON_TEMPORAL_auto`;
- `T_HOP_auto`, the strongest temporal/multi-hop proxy, is not statistically
  significant for Recall@10;
- `T_REQUIRED_auto` is positive but smaller than the hard non-temporal control.

This pattern mildly contradicts the founding temporal-compression thesis. The
current evidence is more consistent with generic learned fusion, neighborhood
smoothing, or capacity effects than with proven temporal-structure exploitation.

## Conclusion

ConvMemory should no longer be described as working because it exploits temporal
order or because CNN-style temporal compression is load-bearing. The honest
claim is narrower and stronger:

1. ConvMemory is a small learned reranker that remains useful as a low-latency
   memory candidate stage.
2. Its practical value is cost-effective reranking quality relative to much
   heavier cross-encoder passes in memory-family settings.
3. The v0.50/v0.51 negative result is itself a contribution: it prevents the
   project from overclaiming temporal structure and points future work toward
   better attribution, manual slice audits, or explicitly constructed memory
   state/contradiction tasks.

Future temporal-mechanism claims require a difficulty-matched, human-audited
non-temporal control slice and a temporal slice where full minus no-temporal is
larger than the matched control with paired-bootstrap confidence intervals that
do not cross zero.

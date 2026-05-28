# ConvMemory v2: Evidence Reranker

ConvMemory v2 is a protected top-10 token-evidence reranker. It is designed to
sit after the released ConvMemory v1 checkpoint:

```text
query + memories
  -> vector search / candidate generation
  -> ConvMemory v1 top-500 rerank
  -> preserve the exact v1 top-10 set
  -> v2 evidence reranker reorders only that top-10 prefix
  -> append the unchanged v1 tail
```

This means v2 is recall-preserving over v1 at top-10: it can improve ordering
inside the v1 top-10, but it cannot rescue a memory that v1 failed to place in
that protected set.

The canonical v2 evidence reranker checkpoint is published at
`Purdy0228/ConvMemory-v2-Evidence-Reranker`. Loading remains explicit.

## API

```python
from convmemory import ConvMemory

model = ConvMemory.from_pretrained("Purdy0228/ConvMemory-LoCoMo-MPNet")
model.load_evidence_reranker("Purdy0228/ConvMemory-v2-Evidence-Reranker")

ranked = model.retrieve(
    query=query,
    memories=memories,
    evidence_reranker="v2",
    top_k=10,
)
```

`retrieve(query, memories)` without `evidence_reranker="v2"` remains the pure
v1 path.

## Headline Numbers

Canonical headline = v361 5-seed run, FULL MRR 0.5824 -> 0.6560, delta +0.0734
[+0.0645, +0.0827].

v364 ablation harness baseline = 0.6677 (full_text retrained in v364 alongside
ablation arms). The +0.012 gap is ~1.3 sigma of fresh-training seed variance
(MRR std ~= 0.009); both v2 estimates significantly outperform v1.

The locked v363 headline table:

| Method | R@10 | MRR | H@1 |
| --- | ---: | ---: | ---: |
| raw_dense | 0.5345 | 0.3254 | 0.1937 |
| ConvMemory v1 | 0.7798 | 0.5824 | 0.4440 |
| ConvMemory v2 v361 | 0.7798 | 0.6560 | 0.5474 |
| mxbai CE top500 | 0.8080 | 0.6688 | 0.5646 |

## Load-Bearing Ablation

The v364 audit retrains the v2-style full-text arm in the same harness as the
ablation arms:

| Arm | FULL MRR |
| --- | ---: |
| v361 full text | 0.6677 |
| no_memory_text | 0.2966 |
| random_other_query_text | 0.2506 |
| shuffled_memory_text | 0.2731 |
| scalar_only | 0.5792 |

Paired bootstrap, full text minus ablation on FULL MRR:

| Comparison | Delta | 95% CI |
| --- | ---: | ---: |
| full - no_memory_text | +0.3712 | [+0.3599, +0.3829] |
| full - random_other_query_text | +0.4173 | [+0.4067, +0.4284] |
| full - shuffled_memory_text | +0.3948 | [+0.3834, +0.4060] |
| full - scalar_only | +0.0881 | [+0.0801, +0.0969] |

The three text perturbations all fall below raw_dense MRR 0.3254. This is the
main load-bearing result: token interaction on candidate-specific memory text is
doing the work, not scalar/rank/time shortcuts or arbitrary topic-adjacent text.

## Cost

Measured on the v362 RTX 4080 SUPER run:

| Path | ms/query |
| --- | ---: |
| ConvMemory v1 top500 | 16.8 |
| v1 + v2 evidence reranker | 28.6 |
| mxbai top500 CE | 1960.2 |

The v1+v2 path is about 1.7x v1 and about 68x cheaper than mxbai top500 CE in
that measurement.

## Relationship To The v1 Paper

The v1 technical report's core engineering claim remains about the default
ConvMemory reranker: v1 does not run a per-query/per-candidate transformer
forward over the full candidate pool, and `retrieve(query, memories)` still
uses that pure v1 path unless v2 is explicitly requested.

ConvMemory v2 deliberately changes the cost/accuracy trade-off. It reintroduces
a small token-level cross-encoder only after v1 has already narrowed the search
to a protected top-10 set. This is a bounded precision stage, not a replacement
for v1's cheap high-recall stage:

```text
v1 paper claim: avoid transformer pair scoring over the full memory pool
v2 design: use v1 to select top-10, then run token evidence scoring only there
```

So v2 does not erase the v1 result; it composes with it. The negative
attribution result in the v1 paper also remains unchanged: v2 is not presented
as evidence for the original temporal-window mechanism. Its supported mechanism
is different and is tested separately in v364: candidate-specific memory text is
load-bearing, while scalar-only and text-mismatch ablations collapse.

The honest framing is therefore:

- v1: cheap default memory reranker and high-recall candidate organizer;
- v2: opt-in top-10 evidence reranker that spends a small, bounded transformer
  budget to improve ordering inside v1's protected set;
- full cross-encoder baselines: stronger and more general, but much more
  expensive when applied over hundreds of candidates.

## Anti-Leak Guard

The public inference input is limited to:

- query text;
- candidate memory id and memory text;
- optional candidate time/position metadata;
- the protected ConvMemory v1 top-10 candidate set.

The API rejects the following candidate fields at inference:

```text
gold, gold_ids, is_current, is_latest, is_stale, stale, answer, answer_text,
ce_score, mxbai_score, teacher_score, gpt_label, entity_id, slot_id
```

Gold labels and teacher scores are allowed only as training/evaluation targets,
not as inference features.

## Limitations

- v2 depends on v1 top-10 recall. If v1 misses the gold memory at top-10, v2
  cannot recover it.
- The headline result is LoCoMo-specific fine-tuning. Cross-domain users should
  train or validate their own evidence reranker.
- Canonical checkpoint distribution:
  `Purdy0228/ConvMemory-v2-Evidence-Reranker`.
- The cross-question random-text test has been run as v364 B2
  `random_other_query_text`.

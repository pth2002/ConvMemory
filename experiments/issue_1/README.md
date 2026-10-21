# Checks run for issue #1

[Issue #1](https://github.com/pth2002/ConvMemory/issues/1) reports ConvMemory on a
real zh/en memory store with a bge-m3 dense top-60 pool. These scripts reproduce
the checks run before replying. All use the five held-out LoCoMo conversations of
the released checkpoint's split (seed 23, 937 questions, stores of 369 to 680
units) and a dense top-60 candidate pool. Counts are hit@10 out of 937. Paired
comparisons use an exact McNemar test.

## 1. How the candidate pool is handed over

`order_check.py`, English checkpoint `Purdy0228/ConvMemory-LoCoMo-MPNet`, CPU.

| same 60 candidates | hit@5 | hit@10 | ms/query |
|---|---|---|---|
| no reranking | 464 | 569 | 0 |
| whole store + `candidate_indices` | 645 | 718 | 54.1 |
| same, `window_mode="candidate_local"` | 641 | 716 | 21.4 |
| only the 60, in rank order | 603 | 679 | 15.5 |
| only the 60, re-sorted into store order | 642 | 705 | 15.2 |

Whole store versus rank order: 50 vs 11 questions (p = 4.6e-7).

## 2. Chinese OPC student and bge-m3 rerank heads

`m3_and_opc_check.py`. Chinese questions and stores come from a gpt-4o-mini
translation of LoCoMo (not redistributed). The BGE-M3 implementation is checked
against the example scores in the BAAI/bge-m3 README before use (`--selftest`).

| hit@10 of 937 | zh → zh | zh → en | en → en |
|---|---|---|---|
| bge-m3 dense top-60 | 507 | 598 | 578 |
| + bge-m3 ColBERT MaxSim | 495 | 574 | 614 |
| + bge-m3 dense/sparse/ColBERT (0.4/0.2/0.4) | 483 | 587 | 620 |
| bge-base-zh-v1.5 dense top-60 | 539 | 372 | 398 |
| + OPC student, whole store + `candidate_indices` | 553 | 380 | 413 |
| + OPC student, only the 60 in rank order | 552 | 378 | 406 |

The hand-off makes no measurable difference for the OPC student (12 vs 11,
11 vs 9, 16 vs 9). bge-m3's ColBERT and combined scores help on English pairs and
do not help on Chinese or cross-lingual pairs.

Raw outputs: `order_check_results.json`, `m3_and_opc_check_results.json`.

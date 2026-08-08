# Chinese ConvMemory

Two Chinese retrieval checkpoints and one Chinese validity module ship through
the same package interface as the English v1 reranker.

## Dual-Space GTE

The Chinese retrieval module uses a dual-space bi-encoder representation, then
applies the lightweight ConvMemory window encoder and CE-lite reranker. The
online path still encodes queries and memories separately; Jina/cross-encoder
scores were used only as offline teacher supervision.

```python
from convmemory import ChineseConvMemory

model = ChineseConvMemory.from_pretrained(
    "Purdy0228/ConvMemory-ZH-DualSpace-GTE",
    device="cuda",  # or "cpu"
)

ranked = model.rerank(
    query="用户最近喜欢什么音乐？",
    memories=[
        {"id": "m1", "text": "用户喜欢周杰伦的音乐。"},
        {"id": "m2", "text": "用户最近在学习法语。"},
    ],
    top_k=5,
)
```

The Hub checkpoint bundles the lightweight ConvMemory student (`student.pt`) and
the tuned Chinese triplet encoder (`triplet_encoder/`). Users do not need to
manually assemble v597/v601 experiment artifacts; `from_pretrained(...)` resolves
the release repository and loads the full dual-space model.

The v601 five-seed method-level result is R@10 0.7855-0.7871 and MRR
0.6145-0.6153, depending on whether the row is selected by R@10 or MRR. The
released representative checkpoint is seed 31 and is validated by a
checkpoint-load smoke test.

## OPC Student (BGE)

For Chinese OPC-style memory retrieval, ConvMemory also provides a BGE-backed
student checkpoint. It keeps the ConvMemory design pattern:

`bge-base-zh embeddings -> ConvMemory window encoder -> CE-lite reranker head`.

```python
from convmemory import OPCConvMemory

model = OPCConvMemory.from_pretrained(
    "Purdy0228/ConvMemory-OPC-Student-BGE",
    device="cuda",  # or "cpu"
)

ranked = model.rerank(
    query="现在定价方案是什么？",
    memories=[
        {"id": "m1", "text": "定价方案最终是基础版 99 元，专业版 299 元。"},
        {"id": "m2", "text": "获客渠道第一批先做小红书。"},
        {"id": "m3", "text": "旧方案曾经考虑过 49 元，后来被否掉。"},
    ],
    top_k=3,
)
```

The checkpoint was trained by offline teacher distillation from
`BAAI/bge-reranker-v2-m3`, but evaluation is against construction gold, not
teacher agreement. On the released API path, the phase-0 pass-only test split
reports R@10 0.9913, Hit@1 0.7442, and MRR 0.8484.

**Scope warning.** In warm-memory pool scans, the student is much faster than the
online teacher for large candidate pools. But the phase-0 gate did **not** show a
statistically robust improvement over `BAAI/bge-base-zh-v1.5` cosine retrieval,
so small candidate pools may be better served by the dense retriever alone. Use
this checkpoint when the pool is large enough that a cross-encoder is too
expensive and dense recall is degrading — not as a default upgrade over dense.

## OPC-v3 Validity Context Module

OPC-v3 is a validity layer for OPC-style memory updates. It does not replace the
Chinese retriever. It scores a query, a later source/update memory, and a target
memory to decide whether the target should be surfaced as possibly outdated for
that query.

```python
from convmemory import ValidityEvidenceModule

module = ValidityEvidenceModule.from_pretrained(
    "Purdy0228/ConvMemory-OPC-V3-Validity-Context",
    device="cuda",  # or "cpu"
)

scores = module.score_evidence_pairs(
    [
        {
            "query": "What is the current pricing plan?",
            "source": {
                "text": "Later update: pricing changed from 99/299 to 129/399."
            },
            "target": {"text": "Earlier note: pricing was 99/299."},
        }
    ]
)
```

The selected checkpoint is the v632 operating point. On the handwritten OPC
smoke set it reports pair accuracy 98.96%, demote recall 100.00%, protect recall
98.81%, and scenario all-correct 91.67%. See [OPC_V3_VALIDITY.md](OPC_V3_VALIDITY.md)
for the model card, boundary, and recommended use.

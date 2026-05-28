# Memory-MLA Expander

Memory-MLA is an experimental recall expander. It is an opt-in module
that runs after the base ConvMemory v1 ranking. The default API path remains v1:
`retrieve(query, memories)` does not use Memory-MLA unless `expander="memory_mla"`
is passed.

Memory-MLA is not the v0.5.0 ConvMemory v2 evidence reranker. It remains a
research-preview recall expansion module with the original `memory_mla` module
name kept for backward compatibility.

## Mechanism

Memory-MLA is a prefix-protected recall expander:

1. ConvMemory v1 produces the base ranking.
2. The top `protect_top_k` results are copied through unchanged.
3. The expander scores a small later candidate window with compressed memory
   latent codes and lightweight query-to-code interaction.
4. Only the suffix window is reordered; the protected prefix is never changed.

The packaged configuration mirrors the verified v320 setting:

- `latent_count=12`
- `code_dim=64`
- `protect_top_k=7`
- `expand_window=16`

This makes Memory-MLA a recall expansion tool, not a replacement ranker. Its
goal is to rescue additional relevant memories into the top-k list while keeping
the highest-confidence ConvMemory prefix stable.

## API

```python
from convmemory import ConvMemory

model = ConvMemory.from_pretrained("Purdy0228/ConvMemory-LoCoMo-MPNet")
model.load_expander("path-or-hub-id-for-memory-mla")

ranked = model.retrieve(
    query=query,
    memories=candidates,
    expander="memory_mla",
    protect_top_k=7,
    expand_window=16,
    top_k=20,
)
```

`expander=` accepts only `None`, `"memory_mla"`, or a `MemoryMLAExpander`
instance. Other spellings raise `ValueError`.

## Training Discipline

The expander is trained as a retrieval module. It must not receive any
gold-defining feature such as `is_answer`, `gold`, `current`, `stale`, or an
oracle conflict label. It also does not use a teacher at inference time.

The public inference signature uses only:

- base ConvMemory `RerankResult` scores and ranks;
- candidate ids, texts, and embeddings;
- the query text and query embedding;
- candidate positions in the memory store;
- deterministic lexical overlap and normalized score features;
- optional chunk embeddings from the same attached embedding encoder.

The checkpoint records `trained_embedding_model_name`. Attaching an expander
trained on a different embedding backbone emits a warning, matching the CCGE-LA
integration pattern.

## v320 Verification

These numbers come from `results/v320_capacity_m12d64_medium/summary.csv`.
They are 5-seed results over seeds `7, 11, 23, 31, 47`. The T_SUP slice is an
automatic slice, not a human-audited temporal-conflict benchmark.

The productized v2 configuration is the fixed safe arm
`v317_prefix7_expand16`, corresponding to `protect_top_k=7` and
`expand_window=16`.

| Slice | ConvMemory v1 MRR | ConvMemory v1 R@10 | Memory-MLA MRR | Memory-MLA R@10 | Delta MRR | Delta R@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FULL | 0.579707 | 0.779497 | 0.580059 | 0.785878 | +0.000352 | +0.006381 |
| T_SUP_auto | 0.559556 | 0.749984 | 0.559924 | 0.768347 | +0.000368 | +0.018363 |
| RAW_TOP1_WRONG_GOLD_IN_POOL | 0.515787 | 0.750135 | 0.516274 | 0.759247 | +0.000486 | +0.009112 |

For the hard-recall RAW_RESCUABLE_STALE_TOP1 slice, the best safe fixed arm in
the same v320 archive was `v317_prefix7_expand20`:

| Slice | ConvMemory v1 MRR | ConvMemory v1 R@10 | Best safe Memory-MLA MRR | Best safe Memory-MLA R@10 | Delta MRR | Delta R@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RAW_RESCUABLE_STALE_TOP1 | 0.507123 | 0.746044 | 0.507875 | 0.758049 | +0.000752 | +0.012005 |

Interpretation: Memory-MLA gives a consistent recall-side gain in the v320
archive, especially on T_SUP_auto and hard-recall slices. The MRR gains are
small. This is an experimental recall expander, not a SOTA claim.

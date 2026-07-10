# OPC-v3 Validity Context Module

This checkpoint is the OPC-oriented ConvMemory v3 validity module. It is not a
general Chinese retriever and it is not the OPC student reranker. It scores
query/source/target triples:

```text
USER_QUERY + SOURCE_EVIDENCE -> TARGET_MEMORY
```

The intended question is:

> Given this query and later source/update evidence, should this target memory
> be treated as outdated for the current answer?

## Hub Checkpoint

[Purdy0228/ConvMemory-OPC-V3-Validity-Context](https://huggingface.co/Purdy0228/ConvMemory-OPC-V3-Validity-Context)

The checkpoint is loadable with the public v3 API:

```python
from convmemory import ValidityEvidenceModule

module = ValidityEvidenceModule.from_pretrained(
    "Purdy0228/ConvMemory-OPC-V3-Validity-Context",
    device="cuda",  # or "cpu"
)

pairs = [
    {
        "query": "What is the current pricing plan?",
        "source": {
            "text": "Later update: pricing changed from 99/299 to 129/399."
        },
        "target": {
            "text": "Earlier note: pricing was 99/299."
        },
    }
]

scores = module.score_evidence_pairs(pairs)
```

When attached to a `ConvMemory` instance, `validity_mode="context"` preserves
the retrieved order and adds structured validity metadata. `validity_mode="demote"`
is explicit opt-in and should be used only for current-state/update workloads.
Pass the selected update for each target through `validity_source_map` when an
OPC memory index already tracks revisions. This avoids searching every memory
pair and keeps one scorer call per protected target. This integrated source-map
interface requires `convmemory>=0.6.2`.

## Selected Recipe

The selected OPC-v3 module is `v632`:

- base model: `BAAI/bge-reranker-v2-m3`
- training data: OPC synthetic base + hard + natural validity families
- positive repeat: `8`
- threshold: `0.05`
- module format: `convmemory_validity_evidence`
- source policy: top-1 source/update evidence

The `v633` change-query guard run was rejected because it over-corrected toward
protection and reduced handwritten current-query demote recall.

## Evaluation Snapshot

The selected module passed the synthetic base/hard/natural validity checks and
the handwritten OPC smoke test:

| Evaluation | Pair accuracy | Demote recall | Protect recall | Scenario all-correct |
|---|---:|---:|---:|---:|
| synthetic base/hard/natural | 100.00% | 100.00% | 100.00% | 100.00% |
| handwritten OPC smoke | 98.96% | 100.00% | 98.81% | 91.67% |

The handwritten smoke set has 12 update scenarios and 96 query/source/target
pairs. Its remaining known edge is one change-query false demotion: when the
query asks for the change history itself, the old memory should remain visible.

## Recommended Use

Use this module after ordinary retrieval/reranking:

```text
dense / ConvMemory / teacher retrieval
-> pick one later update source per candidate
-> OPC-v3 validity context via validity_source_map
-> context annotation by default
-> demote only for explicit current-state queries
```

For ordinary semantic retrieval over scattered notes, prefer a strong Chinese
embedding model or the existing OPC student reranker. This v3 checkpoint is for
validity and update handling.

## Boundary

This module does not claim:

- generic Chinese semantic retrieval;
- a mature OPC-v2 selector;
- full automatic dependency graph construction;
- universal automatic demotion for all query types.

It is a packaged validity-context component for OPC-style update and stale-memory
handling.

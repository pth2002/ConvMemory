# ConvMemory v3 Validity Context Layer

ConvMemory v3 is a validity-aware context layer for agent memory. It is not a
default replacement for v1/v2 ranking. Its stable default is to attach evidence
and a short context note so the downstream agent can reason about possibly
outdated memories without losing the v1/v2 candidate set.

## Modes

`validity_mode=None` or `validity_mode="off"` leaves ConvMemory behavior
unchanged. This is the package compatibility path.

`validity_mode="context"` attaches structured validity metadata to each returned
memory and preserves both order and candidate set. This is the stable v3 mode for
general traffic.

`validity_mode="demote"` is explicit opt-in. It may reorder candidates by a
validity penalty, but it preserves the candidate set. It is intended for dense
current-state/update workloads where automatic demotion has been validated. It is
not the default for broad LoCoMo-style retrieval.

The integrated API applies validity only to the requested result prefix. If no
`top_k` is supplied, the default protected prefix is 10 memories. Demotion can
reorder that prefix, but it cannot pull an item from outside the prefix into it.

## Output Shape

When context mode is enabled, each `RerankResult` may include a `validity`
dictionary:

```python
{
    "status": "possibly_outdated",
    "confidence": 0.82,
    "action": "context",
    "source_evidence": [
        {
            "id": "memory-42",
            "text": "Later update evidence...",
            "score": 0.82,
            "position": 42,
        }
    ],
    "context_note": "Potential update evidence found for this memory: ...",
}
```

The downstream agent can use this note when constructing prompts. Context mode
does not delete, add, or reorder memories.

## Scorer Checkpoints

`ValidityEvidenceModule` can run in two ways:

- with an injected Python scorer for custom in-process systems;
- with a saved CrossEncoder checkpoint, using the query/source/target format
  from the v506/v511 query-conditioned demotion evaluation runs.

For a binary demotion calibrator, save a config like:

```json
{
  "cross_encoder_model": "Purdy0228/ConvMemory-v3-Validity-Context",
  "cross_encoder_num_labels": 2,
  "cross_encoder_batch_size": 32,
  "max_length": 192,
  "source_policy": "top1",
  "max_sources_per_candidate": 1,
  "candidate_top_k": 10,
  "require_later_source": true,
  "mode_default": "context"
}
```

The pair format is:

```text
USER_QUERY:
...

SOURCE_EVIDENCE:
...

TASK: Decide whether the target memory should be demoted for this user query.
```

paired with:

```text
TARGET_MEMORY:
...
```

The demotion probability is interpreted as a confidence score. If the scorer
returns logits rather than probabilities, ConvMemory maps them through a sigmoid.

When source evidence has already been retrieved, use
`ValidityEvidenceModule.score_evidence_pairs(...)` to batch explicit
query/source/target pairs through the same packaged scorer. This is the
recommended path for dense current-state workloads because it avoids per-pair
CrossEncoder calls while preserving the same scoring format.

## Source Evidence

For production systems, retrieve one update source for each target and pass it
to the integrated API with `validity_source_map`:

```python
source_map = {
    "old-pricing": {
        "id": "new-pricing",
        "text": "The pricing plan changed from 99/299 to 129/399.",
        "position": 42,
    }
}

ranked = model.retrieve(
    query="What is the current pricing plan?",
    memories=candidates,
    top_k=10,
    validity_mode="context",
    validity_source_map=source_map,
)
```

Map keys are target memory ids. A map value may be a source memory dictionary
or the id of a source already present in `memories`. Explicit source evidence
skips internal source search.

When no map is provided, ConvMemory uses a bounded fallback. It considers later
memories when position metadata is available, ranks them with a cheap lexical
query/target overlap score, and sends at most one source per target to the
validity scorer. Chinese characters and character bigrams are included in this
fallback. The package does not run an unbounded all-pairs CrossEncoder search.

## Safety Contracts

The public API is designed around machine-checkable contracts:

- Off mode is byte-identical to the ordinary ConvMemory path.
- Context mode preserves ranking order and candidate set.
- Demote mode preserves the candidate set and is opt-in.
- Integrated validity scoring is bounded by the protected result prefix and the
  configured per-target source limit.
- Inference inputs reject gold, stale/current labels, teacher scores, GPT labels,
  answer text, and other evaluation-only fields.
- Context evidence output does not copy forbidden evaluation-only fields.

These are covered in `tests/test_validity_context.py`.

For checkpoint provenance, package-level benchmark numbers, latency, and the
source-of-truth ledger, see [V3 Model Card](V3_MODEL_CARD.md).

## Current Boundary

The validity context layer uses a conservative top-1 source evidence policy.
Applications with their own update index should pass `validity_source_map` or
call `score_evidence_pairs(...)` directly. Automatic strict dependency graph construction and
multi-hop learned graph propagation are advanced capabilities and are not part
of the default v3 behavior.

Recommended language:

- Stable: `context` mode, top-1 source evidence, no ranking mutation.
- Opt-in: `demote` mode for dense current-state/update workloads.
- Advanced: automatic strict dependency graph construction and broad learned
  multi-hop validity propagation.

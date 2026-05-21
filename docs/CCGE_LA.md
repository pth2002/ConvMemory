# CCGE-LA: Conflict-Aware ConvMemory Editor

CCGE-LA is the public alpha structure for stale/current memory conflicts.

The name means:

> Low-Amplitude Counterfactual Conflict Graph Editor

It is not a replacement for ConvMemory. It is a lightweight editor that runs
after ConvMemory and before agent context construction.

## Placement

```text
query
  -> vector search top-k
  -> ConvMemory rerank
  -> CCGE-LA conflict-aware score edit
  -> memory context for the downstream agent
```

ConvMemory remains the semantic memory retriever. CCGE-LA reads ConvMemory's
candidate set and applies a small residual correction when the candidate set
looks conflict-prone.

## Why This Structure Exists

Long-term memory systems often retrieve both stale and current facts. A normal
semantic retriever may rank an older but very similar memory above the current
one.

CCGE-LA targets this failure mode by reading the whole candidate set. It looks
for signs such as:

- semantically similar candidates at different time/order positions;
- text overlap between the top ConvMemory candidate and nearby alternatives;
- narrow score margins at the top of the list;
- high candidate-set entropy;
- dense clusters of memories around the same topic;
- ConvMemory top-1 failures where the gold memory is already in the pool.

The editor does not decide the final answer. It only changes the memory ranking
before the LLM sees the context.

## Score Update

The core update is residual:

```text
final_score_i = convmemory_score_i + gate(candidate_set) * residual_i
```

The gate is query/candidate-set level and intentionally low-amplitude. Internal
experiments found that making the gate more granular at the candidate level was
less stable than a small query-level editor.

## Public API

The installable implementation lives in:

```text
convmemory/ccge.py
```

It exposes:

- `build_ccge_features`
- `CCGELowAmplitudeEditor`
- `multi_positive_retrieval_loss`
- `rank_candidates`

The old `experiments/ccge_la_research_preview.py` file is now a compatibility
wrapper around the package module.

Minimal usage:

```python
from convmemory import CCGELowAmplitudeEditor, ConvMemory

model = ConvMemory.from_pretrained("Purdy0228/ConvMemory-LoCoMo-MPNet")
model.load_ccge_editor("Purdy0228/ConvMemory-CCGE-LA")

results = model.retrieve(
    query="What is the user's current hiking plan?",
    memories=memories,
    mode="rerank",
    editor="ccge_la",
    top_k=10,
)
```

For development and smoke tests, an editor can also be attached directly:

```python
model.attach_ccge_editor(CCGELowAmplitudeEditor())
```

That creates a randomly initialized editor. It verifies the API path, but it is
not a useful retrieval model until trained.

## Alpha Checkpoint

An alpha LoCoMo/MPNet checkpoint is available from Hugging Face Hub:

[Purdy0228/ConvMemory-CCGE-LA](https://huggingface.co/Purdy0228/ConvMemory-CCGE-LA)

It is intended to be attached to the matching base ConvMemory checkpoint:

[Purdy0228/ConvMemory-LoCoMo-MPNet](https://huggingface.co/Purdy0228/ConvMemory-LoCoMo-MPNet)

```python
model = ConvMemory.from_pretrained("Purdy0228/ConvMemory-LoCoMo-MPNet")
model.load_ccge_editor("Purdy0228/ConvMemory-CCGE-LA")
```

The same editor checkpoint is also available as a GitHub release asset:

[Download `convmemory-ccge-la-locomo-mpnet-seed23-alpha.zip`](https://github.com/pth2002/ConvMemory/releases/download/ccge-la-alpha-v0.1/convmemory-ccge-la-locomo-mpnet-seed23-alpha.zip)

SHA256:
`459ecfb2b4c35887f1d8f2cdd87dab402c37bd8dee86628655eff08f314b2e7c`.

It was trained on the seed-23 LoCoMo-style split over ConvMemory candidate
caches. The checkpoint is for API trials and early integration; it should not be
treated as a final benchmark release.

Seed-23 test metrics in the release manifest:

| subset | ConvMemory MRR | CCGE-LA alpha MRR | CCGE-LA R@10 |
|---|---:|---:|---:|
| FULL | 0.5501 | 0.5638 | 0.7725 |
| T_SUP_auto | 0.5424 | 0.5508 | 0.7138 |
| CONV_TOP1_WRONG_GOLD_IN_POOL | 0.2468 | 0.2994 | 0.6822 |
| RESCUABLE_STALE_TOP1 | 0.2470 | 0.3093 | 0.6877 |

## Training Signal

The intended training objective is retrieval cross-entropy over the candidate
set.

Do not train it with:

- current/stale labels;
- is-gold or is-current feature indicators;
- gold-defined "latest" inputs;
- auxiliary state supervision;
- offline distillation as the defining mechanism.

Those shortcuts would undermine the structural claim.

Minimal training skeleton:

```python
import torch

from convmemory import CCGELowAmplitudeEditor, build_ccge_features
from convmemory import multi_positive_retrieval_loss

editor = CCGELowAmplitudeEditor()
optimizer = torch.optim.AdamW(editor.parameters(), lr=2e-4)

batch = build_ccge_features(
    candidate_ids=candidate_ids,
    convmemory_scores=convmemory_scores,
    dense_scores=dense_scores,
    positions=memory_positions,
    candidate_embeddings=candidate_embeddings,
    query=query,
    candidate_texts=candidate_texts,
)

features = torch.tensor(batch.features, dtype=torch.float32).unsqueeze(0)
gold_mask = torch.tensor([[cid in gold_ids for cid in batch.candidate_ids]])

scores, gate = editor(features)
loss = multi_positive_retrieval_loss(scores, gold_mask)
loss.backward()
optimizer.step()
```

After training, save the editor and attach it to ConvMemory:

```python
editor.save_pretrained("checkpoints/my-ccge-la")
memory_reranker.load_ccge_editor("checkpoints/my-ccge-la")
```

## Internal Evidence Summary

The strongest internal real-LoCoMo line used ConvMemory as the candidate source
and trained a conflict-state residual editor over the candidate set.

Representative five-seed internal results:

| Setting | ConvMemory MRR | Best editor MRR | Delta |
|---|---:|---:|---:|
| FULL | 0.5708 | 0.5878 | +0.0170 |
| T_SUP_auto | 0.5518 | 0.5803 | +0.0285 |
| ConvMemory top-1 wrong, gold in pool | 0.2586 | 0.3285 | +0.0699 |
| Rescuable stale top-1 | 0.2636 | 0.3655 | +0.1019 |

Reading:

- The full-set improvement is modest but positive.
- The improvement is larger on supersession/stale-conflict slices.
- More complex candidate-level gates, directional graphs, and dual-head variants
  did not become the best mainline.

## Relation To Cross-Encoders

CCGE-LA is still a reranking layer. It does not remove the need for reranking.

Compared with a cross-encoder:

- cross-encoders usually have stronger top-rank precision;
- CCGE-LA is cheaper and candidate-set structured;
- CCGE-LA is designed for stale/current memory conflict repair, not generic
  document ranking.

The intended deployment pattern is:

```text
vector top-k -> ConvMemory -> CCGE-LA -> optional cross-encoder -> context
```

For cost-sensitive agents, CCGE-LA can be used as a lightweight conflict-aware
reranker. For maximum accuracy, a cross-encoder can still be used after it.

## Current Status

CCGE-LA is a public alpha API. Before it becomes a stable default feature, it
still needs:

- an end-to-end reproducible public training/evaluation command;
- same-split comparisons against raw dense retrieval, ConvMemory, and a strong
  cross-encoder.

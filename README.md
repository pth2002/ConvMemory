# ConvMemory

**A lightweight reranker for long-term agent memory.** Better memory retrieval
without running a large cross-encoder over the full candidate pool.

[![CI](https://github.com/pth2002/ConvMemory/actions/workflows/ci.yml/badge.svg)](https://github.com/pth2002/ConvMemory/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/convmemory.svg)](https://pypi.org/project/convmemory/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/pth2002/ConvMemory/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://github.com/pth2002/ConvMemory/blob/main/pyproject.toml)

![ConvMemory recovering the answer a dense search missed, on a 680-memory pool](https://raw.githubusercontent.com/pth2002/ConvMemory/main/docs/assets/demo.gif)

*Real output from [`examples/demo_locomo.py`](https://github.com/pth2002/ConvMemory/blob/main/examples/demo_locomo.py) on held-out conversations. Reproduce it in one command.*

You already have a memory store and a vector search. ConvMemory is one layer
after it:

```text
        user query
            |
   vector / hybrid search        mem0 · LangChain · LlamaIndex · your own store
            |          top-k candidates
       ConvMemory                14 MB reranker, no cross-encoder over the pool
            |          reordered memory context
          agent
```

**Keep your existing memory store. Add ConvMemory as the reranking layer.**
Drop-in adapters for [mem0, LangChain, LlamaIndex and plain vector stores](https://github.com/pth2002/ConvMemory/blob/main/examples/integrations/)
are about ten lines each.

> **On mechanism honesty:** ConvMemory was originally sold as exploiting
> *temporal structure*. A five-seed attribution study refuted that, and the
> README below says so. If that is the kind of thing you want to read:
> [I thought ConvMemory worked because of temporal reasoning. Five seeds proved
> me wrong.](https://github.com/pth2002/ConvMemory/blob/main/docs/posts/i-was-wrong-about-temporal-memory.md)

## Install

The reranker is a small torch model. The encoder stack is optional, because a
memory store that already holds embeddings does not need one:

```bash
pip install "convmemory[encode]"   # text in, text out
pip install convmemory             # reranker only: numpy + torch, no encoder stack
```

## Three lines

```python
from convmemory import ConvMemory

model = ConvMemory.from_pretrained("Purdy0228/ConvMemory-LoCoMo-MPNet")
results = model.rerank(query=query, memories=candidates, top_k=10)
```

`memories` is a list of `{"id": ..., "text": ...}` in the order your store keeps
them. Each result carries `rank`, `memory_id`, `score`, and `text`.

Already have embeddings? Then you never need the encoder — this is the path the
core install is built for, and it is roughly 50x faster because nothing is
re-encoded:

```python
results = model.rerank_embeddings(
    query_embedding=query_embedding,
    memory_embeddings=memory_embeddings,   # what your store already has
    memory_ids=memory_ids,
    memory_texts=memory_texts,
    query=query,
    top_k=10,
)
```

## Why ConvMemory?

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/pth2002/ConvMemory/main/docs/assets/pareto-dark.png">
  <img alt="Retrieval quality vs reranking latency on LoCoMo: raw dense, ConvMemory v1, ConvMemory v1+v2, and mxbai-rerank-large-v1" src="https://raw.githubusercontent.com/pth2002/ConvMemory/main/docs/assets/pareto-light.png">
</picture>

Reranking a 500-candidate pool with a large cross-encoder means 500 transformer
forward passes per query. ConvMemory scores the pool with a small learned model
over embeddings you already have, and gets most of the way there:

| Method | R@10 | MRR | ms/query |
|---|---:|---:|---:|
| raw dense (no reranker) | 0.5345 | 0.3254 | <1 |
| **ConvMemory v1** | **0.7798** | **0.5824** | **16.8** |
| **ConvMemory v1 + v2** | **0.7798** | **0.6560** | **28.6** |
| mxbai-rerank-large-v1, full pool | 0.8080 | 0.6688 | 1960.2 |

LoCoMo, 5 split seeds, top-500 candidate pool. Latency measured on an RTX 4080
SUPER with memory embeddings precomputed. `mxbai-rerank-large-v1` is still the
more accurate reranker — it costs 68x more per query to get there.

Full tables, including the ones where ConvMemory loses:
[docs/BENCHMARKS.md](https://github.com/pth2002/ConvMemory/blob/main/docs/BENCHMARKS.md).

## See it on real data

`examples/demo_locomo.py` runs dense retrieval and dense + ConvMemory side by
side on the held-out conversations of the released checkpoint's split:

```bash
python examples/demo_locomo.py --data data/locomo10.json --device cuda
```

Output on the five held-out conversations (937 questions, candidate pools of
369-680 memories):

```text
metric          dense only    dense + ConvMemory
hit@5           464 (49.5%)          682 (72.8%)
hit@10          569 (60.7%)          769 (82.1%)

per-query time (memory embeddings precomputed, lexical signatures prewarmed):
  query encoding (shared by both paths): 8.05 ms
  dense scoring:                         0.26 ms
  ConvMemory reranking:                 18.53 ms
```

A single question from that run, pool of 680 memories:

> **Q: When did John take a trip to the Rocky Mountains?**

```text
dense only
  1.    John: Wow, that sounds awesome! How challenging was the trek through the Himalayas?
  2.    John: We went camping in the mountains and it was stunning! The air was so refreshing.
  3.    Tim: The book mentioned that the trek was tough but worth it, with challenging terrain...
  4.    John: I stumbled across this spot while hiking. The sound of that river was so soothing...
  5.    John: Wow, great view! Have you visited any other places?

dense + ConvMemory
  1. OK John: ... This was my Rocky Mountains trip last year and it was stunning. Seeing those
        mountains, fresh air - it makes you realize how incredible the world is.
  2.    John: We went camping in the mountains and it was stunning! The air was so refreshing.
  3.    Tim: I snapped that pic on my trip to the Smoky Mountains last year...
```

Dense retrieval fills the top of the list with memories that are *about*
mountains. The evidence turn itself sits below the cut.

LoCoMo is in-domain for this checkpoint — it was trained on the other half of
the same dataset. This demo shows the retrieval-stage before/after on held-out
conversations, not out-of-domain generalization. Get `locomo10.json` from
[snap-research/locomo](https://github.com/snap-research/locomo).

## When it will not help you

Worth reading before you install anything.

- **Small candidate pools.** On a ~100-memory pool where the answer is lexically
  obvious, plain dense search already puts the right memory in the top 5 and
  there is nothing to recover. `examples/demo_side_by_side.py` is that case, kept
  in the repo on purpose. ConvMemory earns its place at hundreds to thousands of
  candidates, where raw dense recall drops (0.90 to 0.55 on the LongMemEval
  stress setting).
- **General document retrieval.** ConvMemory regresses below raw dense on
  MuSiQue, and a plain dense+lexical baseline beats it on HotpotQA. It is a
  memory reranker, not a multi-hop document reranker.
- **Maximum top-rank precision, cost no object.** If you can afford 2 seconds
  per query, `mxbai-rerank-large-v1` is more accurate.
- **Calibrated confidence.** Scores are comparable within a query, not across
  queries.

## How it works, and one thing I got wrong

ConvMemory scores each candidate from features that are cheap once your
embeddings exist: dense similarity, lexical interaction between query and
memory, and a small learned window over neighboring memories. There is no
per-candidate transformer forward pass in the v1 path.

The original claim was that the **temporal window** was the reason it worked.
A 5-seed retrained paired-bootstrap attribution study does not support that:

| Slice | Delta R@10 from the temporal window | 95% CI |
|---|---:|---:|
| all questions | +0.0376 | [+0.0306, +0.0451] |
| most-temporal multi-hop proxy | +0.0096 | [-0.0037, +0.0230] |
| hard **non**-temporal control | +0.0838 | [+0.0650, +0.1040] |

The window helps, but it helps *most* on the non-temporal control and not
significantly on the most temporal slice. That is the signature of generic
neighborhood smoothing, not temporal-structure exploitation. The retrained
ablation points the same way: removing lexical interaction costs -0.089 R@10,
removing the temporal window costs -0.035, and removing the router costs
nothing at all.

So the supported description is: **a lightweight learned reranker over fused
dense + lexical features, with a small neighborhood window** — not a temporal
reasoning mechanism. The engineering result survived the negative attribution
result; the explanation did not.

The long version, including how the experiment was designed to be able to fail:
[I thought ConvMemory worked because of temporal reasoning. Five seeds proved me
wrong.](https://github.com/pth2002/ConvMemory/blob/main/docs/posts/i-was-wrong-about-temporal-memory.md)
Numbers and protocol: [docs/NEGATIVE_RESULTS.md](https://github.com/pth2002/ConvMemory/blob/main/docs/NEGATIVE_RESULTS.md).

## Integrations

Keep your store, add the layer. Examples in [`examples/integrations/`](https://github.com/pth2002/ConvMemory/blob/main/examples/integrations/):

| Example | What it shows |
|---|---|
| [`basic_vector_store.py`](https://github.com/pth2002/ConvMemory/blob/main/examples/integrations/basic_vector_store.py) | numpy vector store, no framework |
| [`langchain_retriever.py`](https://github.com/pth2002/ConvMemory/blob/main/examples/integrations/langchain_retriever.py) | a LangChain-compatible retriever wrapper |
| [`llamaindex_postprocessor.py`](https://github.com/pth2002/ConvMemory/blob/main/examples/integrations/llamaindex_postprocessor.py) | a LlamaIndex node postprocessor |
| [`mem0_rerank.py`](https://github.com/pth2002/ConvMemory/blob/main/examples/integrations/mem0_rerank.py) | reranking mem0 search results |
| [`custom_agent.py`](https://github.com/pth2002/ConvMemory/blob/main/examples/integrations/custom_agent.py) | a plain agent memory loop |

## Benchmark it on your own data

```bash
convmemory benchmark --queries queries.jsonl --memories memories.jsonl
```

Reports Recall@k, MRR and latency before and after ConvMemory on your memories,
and writes a `--json` summary that is easy to paste into an issue.
See [docs/BENCHMARK_CLI.md](https://github.com/pth2002/ConvMemory/blob/main/docs/BENCHMARK_CLI.md) for the file format.

**Read the "before" row carefully.** It is cosine similarity in *this
checkpoint's* MPNet space, not your production retriever. If you retrieve with a
stronger embedding model, your real baseline is higher than that row and the
delta the tool prints is larger than what you would actually gain. The question
it answers is "how much does the reranker add on top of the space it scores in",
not "should I replace my retriever". It also needs labelled `gold_ids` per
query, which many memory systems do not have lying around.

If it does not help on your data, that is a useful result — please open an issue
with what you saw.

## More layers

The v1 reranker above is the core. These are opt-in and off by default:

| Layer | What it adds | Docs |
|---|---|---|
| **v2 evidence reranker** | token-level rescoring of the protected v1 top-10; +0.073 MRR, recall-preserving | [EVIDENCE_RERANKER.md](https://github.com/pth2002/ConvMemory/blob/main/docs/EVIDENCE_RERANKER.md) |
| **v3 validity context** | marks a returned memory as possibly outdated, with the update evidence | [VALIDITY_CONTEXT.md](https://github.com/pth2002/ConvMemory/blob/main/docs/VALIDITY_CONTEXT.md) |
| **Chinese models** | dual-space GTE retriever, BGE student, OPC-v3 validity | [CHINESE.md](https://github.com/pth2002/ConvMemory/blob/main/docs/CHINESE.md) |
| **CCGE-LA (alpha)** | conflict-aware editing of stale/current memory pairs | [CCGE_LA.md](https://github.com/pth2002/ConvMemory/blob/main/docs/CCGE_LA.md) |
| **Memory-MLA (experimental)** | prefix-protected recall expander | [MEMORY_MLA.md](https://github.com/pth2002/ConvMemory/blob/main/docs/MEMORY_MLA.md) |

Note on the stale/current problem: v1 does **not** resolve it. Asked "where does
Alice work now", the v1 reranker will happily rank an outdated memory first.
That is what the v3 validity layer is for, and it attaches evidence rather than
silently reordering.

## Checkpoints

| Checkpoint | Hub |
|---|---|
| ConvMemory v1 (LoCoMo/MPNet) | [Purdy0228/ConvMemory-LoCoMo-MPNet](https://huggingface.co/Purdy0228/ConvMemory-LoCoMo-MPNet) |
| v2 evidence reranker | [Purdy0228/ConvMemory-v2-Evidence-Reranker](https://huggingface.co/Purdy0228/ConvMemory-v2-Evidence-Reranker) |
| v3 validity context | [Purdy0228/ConvMemory-v3-Validity-Context](https://huggingface.co/Purdy0228/ConvMemory-v3-Validity-Context) |
| CCGE-LA alpha editor | [Purdy0228/ConvMemory-CCGE-LA](https://huggingface.co/Purdy0228/ConvMemory-CCGE-LA) |
| Chinese dual-space GTE | [Purdy0228/ConvMemory-ZH-DualSpace-GTE](https://huggingface.co/Purdy0228/ConvMemory-ZH-DualSpace-GTE) |
| Chinese OPC student (BGE) | [Purdy0228/ConvMemory-OPC-Student-BGE](https://huggingface.co/Purdy0228/ConvMemory-OPC-Student-BGE) |

The v1 checkpoint is also a [GitHub release asset](https://github.com/pth2002/ConvMemory/releases/download/v0.1.0/convmemory-locomo-mpnet.zip);
offline setup instructions are in [docs/MODEL_CARD.md](https://github.com/pth2002/ConvMemory/blob/main/docs/MODEL_CARD.md).
Checkpoints and embeddings must share an embedding model family and dimension.

## API surface

Stable:
`ConvMemory.from_pretrained`, `.rerank`, `.retrieve`, `.expand_context`,
`.rerank_embeddings`, `.encode`, `.prewarm_lexical`.

Public alpha: `.load_ccge_editor`, `.attach_ccge_editor`,
`CCGELowAmplitudeEditor`, `build_ccge_features`.

Research preview: context-expansion policies, cascade fusion with cross-encoder
scoring, JSONL adapters for external datasets. See [docs/MODULES.md](https://github.com/pth2002/ConvMemory/blob/main/docs/MODULES.md).

## Technical report

[ConvMemory: A Lightweight Learned Memory Reranker, a Negative Attribution
Result, and a Research-Preview Conflict Editor](https://github.com/pth2002/ConvMemory/blob/main/paper/convmemory_report.pdf)
(May 2026)

Reproducibility: [EVALUATION_PROTOCOL.md](https://github.com/pth2002/ConvMemory/blob/main/docs/EVALUATION_PROTOCOL.md),
[MODEL_CARD.md](https://github.com/pth2002/ConvMemory/blob/main/docs/MODEL_CARD.md), [TRAINING.md](https://github.com/pth2002/ConvMemory/blob/main/docs/TRAINING.md).
Raw datasets, checkpoints, embedding caches, and per-question CSVs are
intentionally kept out of Git history.

## License

MIT

# ConvMemory

**A lightweight reranker for long-term agent memory.** Better memory retrieval,
at a fraction of the cost of running a large cross-encoder over the pool.

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

> Every mechanism claim here is backed by a five-seed retrained attribution
> study with paired-bootstrap intervals. The write-up of how that study was
> designed, and what it changed:
> [I thought ConvMemory worked because of temporal reasoning. Five seeds proved
> me wrong.](https://github.com/pth2002/ConvMemory/blob/main/docs/posts/i-was-wrong-about-temporal-memory.md)

## Install

The reranker is a small torch model, and the encoder stack is optional — a
store that already holds embeddings runs on the core install alone:

```bash
pip install "convmemory[encode]"   # text in, text out
pip install convmemory             # reranker only: numpy + torch
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
over embeddings you already have, and lands on the cost/quality frontier:

| Method | R@10 | MRR | ms/query |
|---|---:|---:|---:|
| raw dense (no reranker) | 0.5345 | 0.3254 | <1 |
| BGE-reranker-base, full pool | 0.6967 | 0.5469 | — |
| BGE-reranker-large, full pool | 0.7621 | 0.6120 | — |
| **ConvMemory v1** | **0.7798** | 0.5824 | **16.8** |
| **ConvMemory v1 + v2** | **0.7798** | **0.6560** | **28.6** |
| mxbai-rerank-large-v1, full pool | 0.8080 | 0.6688 | 1960.2 |

**ConvMemory v1 + v2 beats both BGE rerankers on Recall@10 and MRR, at 28.6 ms —
against models that run a full cross-encoder pass over all 500 candidates.** It
reaches 98% of the mxbai MRR for 1/68th of the latency.

On a CPU box, or anywhere a query budget is measured in milliseconds, this is
the reranking layer you can deploy today.

<sub>LoCoMo, 5 split seeds, top-500 pool, RTX 4080 SUPER, memory embeddings precomputed. BGE latency comes from this repo's LongMemEval harness (BGE-large, 556 ms/query) rather than the timed LoCoMo run. Complete results across six datasets: [docs/BENCHMARKS.md](https://github.com/pth2002/ConvMemory/blob/main/docs/BENCHMARKS.md).</sub>

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

ConvMemory promotes the evidence turn to rank 1. Dense similarity alone fills
the top of the list with memories that are *about* mountains.

<sub>Held-out conversations of the released checkpoint's split; LoCoMo is in-domain for this checkpoint. Get `locomo10.json` from [snap-research/locomo](https://github.com/snap-research/locomo).</sub>

## Where it fits

ConvMemory is built for candidate pools of hundreds to thousands — the regime
where dense recall slides from 0.90 to 0.55 on the LongMemEval stress setting,
and where a cross-encoder over the whole pool prices itself out.

It is tuned for conversational and agent memory: conversations, user histories,
agent traces, task logs, session notes. Results across six datasets, with the
scope of each, are in [docs/BENCHMARKS.md](https://github.com/pth2002/ConvMemory/blob/main/docs/BENCHMARKS.md).

## How it works

A lightweight learned reranker over fused dense + lexical features, with a small
neighborhood window. Every feature is cheap once your embeddings exist, and
there is no per-candidate transformer forward pass in the v1 path — that is
where the 68x comes from.

That description is the one the evidence supports, and earning it took a
five-seed retrained attribution study with paired-bootstrap intervals across
question slices. The design of that study is the most useful thing this project
can hand another ML engineer:

**→ [I thought ConvMemory worked because of temporal reasoning. Five seeds proved
me wrong.](https://github.com/pth2002/ConvMemory/blob/main/docs/posts/i-was-wrong-about-temporal-memory.md)**

Slice tables, intervals, and protocol: [docs/NEGATIVE_RESULTS.md](https://github.com/pth2002/ConvMemory/blob/main/docs/NEGATIVE_RESULTS.md).

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

Bring queries with labelled `gold_ids`. The "before" row is dense retrieval in
the checkpoint's own embedding space — [how to read the delta](https://github.com/pth2002/ConvMemory/blob/main/docs/BENCHMARK_CLI.md#reading-the-result).

Whatever you measure, an issue with your numbers is welcome — reports from real
memory stores are what shape the next checkpoint.

## More layers

The v1 reranker above is the core. These are opt-in and off by default:

| Layer | What it adds | Docs |
|---|---|---|
| **v2 evidence reranker** | token-level rescoring of the protected v1 top-10; +0.073 MRR, recall-preserving | [EVIDENCE_RERANKER.md](https://github.com/pth2002/ConvMemory/blob/main/docs/EVIDENCE_RERANKER.md) |
| **v3 validity context** | surfaces the later update evidence alongside a memory | [VALIDITY_CONTEXT.md](https://github.com/pth2002/ConvMemory/blob/main/docs/VALIDITY_CONTEXT.md) |
| **Chinese models** | dual-space GTE retriever, BGE student, OPC-v3 validity | [CHINESE.md](https://github.com/pth2002/ConvMemory/blob/main/docs/CHINESE.md) |
| **CCGE-LA (alpha)** | conflict-aware editing of stale/current memory pairs | [CCGE_LA.md](https://github.com/pth2002/ConvMemory/blob/main/docs/CCGE_LA.md) |
| **Memory-MLA (experimental)** | prefix-protected recall expander | [MEMORY_MLA.md](https://github.com/pth2002/ConvMemory/blob/main/docs/MEMORY_MLA.md) |

Ask "where does Alice work *now*" and relevance alone will rank the superseded
memory alongside the current one, because both are on topic. The v3 validity
layer handles that case: it attaches the update evidence to a memory so the
agent can reason about it.

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
Raw datasets, checkpoints, embedding caches, and per-question CSVs live outside
Git history to keep the repository light.

## License

MIT

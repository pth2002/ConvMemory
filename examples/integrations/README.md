# Integrations

Keep your existing memory store. Add ConvMemory as the reranking layer.

If your store already holds embeddings, the core install is all you need:
`pip install convmemory` gives you numpy + torch and the `rerank_embeddings`
path. Only the text-in/text-out helpers need `pip install "convmemory[encode]"`.

Every example here is the same three steps:

1. ask your store for a **wider** candidate list than you actually want (100-500);
2. hand those candidates to ConvMemory;
3. give the reranked short list to the agent.

| File | Stack | Runs here? |
|---|---|---|
| [`basic_vector_store.py`](basic_vector_store.py) | numpy, no framework | yes |
| [`custom_agent.py`](custom_agent.py) | plain agent memory loop | yes |
| [`mem0_rerank.py`](mem0_rerank.py) | mem0 | yes, with a stub client |
| [`langchain_retriever.py`](langchain_retriever.py) | LangChain `BaseDocumentCompressor` | needs `langchain-core` |
| [`llamaindex_postprocessor.py`](llamaindex_postprocessor.py) | LlamaIndex `BaseNodePostprocessor` | needs `llama-index-core` |

The LangChain and LlamaIndex adapters are written against those projects'
reranker interfaces but are **not exercised by this repo's CI**, since neither
framework is a ConvMemory dependency. If one breaks against your version, open
an issue with the traceback — that is a bug worth fixing.

## Two things that matter for quality

**Pass memories in a stable order.** ConvMemory reads a small learned window
over neighboring memories. If your retriever shuffles memories out of the order
your store keeps them in, that signal is noise. Chronological insertion order is
the natural choice for agent memory.

**Reuse embeddings you already have.** `rerank(query, memories)` re-encodes the
memory texts on every call, which dominates latency. If your store already keeps
vectors, call `rerank_embeddings(...)` instead. Measured on LoCoMo pools of
369-680 memories, that is the difference between roughly 930 ms and roughly
19 ms per query.

```python
ranked = model.rerank_embeddings(
    query_embedding=query_vector,
    memory_embeddings=store_vectors,     # what your store already has
    memory_ids=store_ids,
    memory_texts=store_texts,
    query=query,
    candidate_indices=dense_top_200,     # optional: restrict to a candidate pool
    top_k=10,
)
```

The embeddings must come from the same embedding model family and dimension as
the checkpoint. The released v1 checkpoint uses
`sentence-transformers/all-mpnet-base-v2` (768-d). For a different retriever,
retrain rather than assume transfer — see [TRAINING.md](../../docs/TRAINING.md).

## Does it actually help on your data?

```bash
convmemory benchmark --queries queries.jsonl --memories memories.jsonl
```

See [BENCHMARK_CLI.md](../../docs/BENCHMARK_CLI.md).

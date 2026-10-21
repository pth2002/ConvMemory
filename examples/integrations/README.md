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
reranker interfaces. CI covers the framework-free examples, since neither
framework is a ConvMemory dependency; if your version of either needs a tweak,
open an issue with the traceback and it gets fixed.

## Two things that matter for quality

**Pass the store, or at least store order.** ConvMemory's position feature and
its neighbor windows are defined over the order your store keeps memories in.
Best: pass the whole store plus the candidate positions, as
`basic_vector_store.py` and `custom_agent.py` do. When a framework hands you only
the retrieved candidates, sort them back into store order before reranking. The
mem0, LangChain and LlamaIndex adapters do this with a `created_at` field by
default (`order_key=` to change it) and warn when the field is missing. On
held-out LoCoMo with a 60-candidate pool, hit@10 is 718 of 937 with the whole
store, 705 with the candidates in store order, and 679 with the candidates in
similarity order ([issue #1](https://github.com/pth2002/ConvMemory/issues/1)).

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
    candidate_indices=dense_top_200,     # store positions of your candidate pool
    window_mode="candidate_local",       # windows only around the candidates
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

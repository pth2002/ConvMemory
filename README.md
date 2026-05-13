# ConvMemory

Lightweight temporal reranking for long-term conversational and agent memory.

ConvMemory sits between a fast vector retriever and an expensive cross-encoder. It is designed for memory systems where records are ordered over time, such as multi-session conversations, user profiles, agent scratchpads, and event histories.

It does not replace your vector database. It reranks the candidate memories your retriever already found.

```text
User query -> vector search top-k -> ConvMemory rerank -> memory context for your agent
```

## Why ConvMemory?

Long-term memory is not just a bag of independent chunks. In conversations and agent workflows, useful evidence often appears as short event chains:

```text
turn t-2 -> turn t-1 -> turn t -> turn t+1
```

ConvMemory uses lightweight temporal window scoring over precomputed embeddings, then fuses that signal with dense and lexical relevance features. The goal is practical: recover much of the reranking gain at a lower runtime cost than scoring every candidate with a token-level cross-encoder.

## When To Use It

Use ConvMemory when:

- You already retrieve candidate memories with embeddings.
- Your memories have a meaningful chronological order.
- You want better memory selection before giving context to an agent or LLM.
- Cross-encoder reranking over hundreds of candidates is too slow or too costly.

ConvMemory is not meant to be a general web/document reranker. It is most useful for long-term memory traces where neighboring memories can provide signal.

## Results

Fresh module-based evaluation on LoCoMo, full test split, MPNet embeddings, top500 candidate pool.

| Method | Test questions | Recall@10 | Hit@10 | MRR |
|---|---:|---:|---:|---:|
| Raw MPNet retrieval | 937 | 0.553 | 0.607 | 0.334 |
| Cross-encoder top500 | 937 | 0.749 | 0.790 | 0.614 |
| ConvMemory top500 | 937 | 0.768 | 0.821 | 0.554 |

Interpretation:

- ConvMemory substantially improves over raw dense retrieval.
- ConvMemory slightly outperforms the tested cross-encoder on Recall@10 and Hit@10.
- The cross-encoder still has stronger MRR, meaning it is better at placing the best evidence at the very top.

The practical sweet spot is to use ConvMemory as a cheap memory reranker, or as a first reranking stage before a smaller cross-encoder pass.

Additional zero-shot check: the LoCoMo-trained checkpoint was evaluated on LongMemEval-S, 500 questions, with no LongMemEval training. This is a retrieval-stage evaluation, not the full answer-generation leaderboard.

| Method | Questions | Recall@5 | Hit@5 | Recall@10 | Hit@10 | MRR | ms/query |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw MPNet retrieval | 500 | 0.823 | 0.902 | 0.905 | 0.946 | 0.783 | 0.0 |
| Cross-encoder top500 | 500 | 0.882 | 0.934 | 0.933 | 0.956 | 0.890 | 115.6 |
| ConvMemory zero-shot | 500 | 0.920 | 0.968 | 0.959 | 0.988 | 0.897 | 48.7 |

On this run, ConvMemory is the strongest overall retrieval-stage reranker while being faster than the tested cross-encoder on cached embeddings.

LongMemEval-S distractor stress test: each question's memory pool is expanded with sessions from other questions. ConvMemory and the cross-encoder rerank the raw top500 candidates.

| Memory pool | Raw R@10 / MRR | Cross-encoder R@10 / MRR / ms | ConvMemory R@10 / MRR / ms |
|---|---:|---:|---:|
| Original, avg 48 sessions | 0.905 / 0.783 | 0.933 / 0.890 / 115.6 | 0.959 / 0.897 / 48.7 |
| 500 sessions | 0.641 / 0.524 | 0.791 / 0.730 / 528.3 | 0.809 / 0.711 / 357.8 |
| 1000 sessions, top500 rerank | 0.548 / 0.442 | 0.730 / 0.671 / 519.9 | 0.739 / 0.623 / 862.9 |
| 1000 sessions, candidate-local + cached lexical | 0.548 / 0.442 | 0.730 / 0.671 / 519.9 | 0.729 / 0.618 / 89.1 |

The optimized 1000-session row uses candidate-local temporal windows and cached memory lexical signatures. It keeps Recall@10 roughly tied with the tested cross-encoder while being about 5.8x faster in online reranking latency. MRR is still lower, so the cross-encoder remains better at placing the best evidence at rank 1.

The lexical cache is comparable to cached memory embeddings: memory-side signatures can be built when memories are indexed. In the 1000-session run above, preprocessing 18,464 unique memory texts took 10.9s and is not counted in online reranking latency.

## Cost Benchmark

Latency benchmark on the first 100 LoCoMo test questions, CPU, cached MPNet embeddings, average 419 candidates per query. The timing measures reranking after embeddings are available.

| Method | Recall@10 | Hit@10 | MRR | ms/query | Speedup vs CE top500 |
|---|---:|---:|---:|---:|---:|
| Raw vector retrieval | 0.466 | 0.570 | 0.295 | 0.6 | retrieval only |
| ConvMemory top500 | 0.706 | 0.810 | 0.491 | 302.9 | 13.8x |
| ConvMemory top500 + cross-encoder top50 | 0.696 | 0.780 | 0.613 | 989.7 | 4.2x |
| Cross-encoder top500 | 0.739 | 0.800 | 0.623 | 4171.7 | 1.0x |

This benchmark intentionally separates reranking cost from embedding cost. In a production memory system, memory embeddings are usually precomputed and query embeddings are shared with the vector search step.

## Install

From the repository root:

```bash
pip install -e .
```

For a minimal local setup:

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from convmemory import ConvMemory

model = ConvMemory.from_pretrained(
    "checkpoints/convmemory-locomo-mpnet",
    device="cuda",
)

memories = [
    {"id": "m1", "text": "The user said their hiking trip moved to Sunday."},
    {"id": "m2", "text": "The assistant recommended bringing extra water."},
    {"id": "m3", "text": "The user has an exam next Friday."},
]

results = model.rerank(
    "When is the hiking trip?",
    memories,
    top_k=2,
)

for item in results:
    print(item.rank, item.memory_id, item.score, item.text)
```

Memory order matters. Pass memories in chronological order whenever possible.

## In An Agent Memory Pipeline

ConvMemory is usually called after vector search and before prompt construction.

```python
from convmemory import ConvMemory

memory_reranker = ConvMemory.from_pretrained(
    "checkpoints/convmemory-locomo-mpnet",
    device="cuda",
)

def retrieve_agent_memory(query, memory_store, top_k=20):
    candidates = memory_store.vector_search(query, top_k=500)

    ranked = memory_reranker.rerank(
        query=query,
        memories=candidates,
        top_k=top_k,
    )

    return [
        {"id": item.memory_id, "text": item.text, "score": item.score}
        for item in ranked
    ]
```

If your vector database returns candidate ids but the full memory list lives elsewhere, pass the candidate ids:

```python
memory_reranker.prewarm_lexical(all_user_memories)
candidate_ids = [item["id"] for item in memory_store.vector_search(query, top_k=500)]

ranked = memory_reranker.rerank(
    query=query,
    memories=all_user_memories,
    candidate_ids=candidate_ids,
    top_k=20,
    window_mode="candidate_local",
)
```

## Use Precomputed Embeddings

For production systems, you may already have normalized embeddings. Use `rerank_embeddings` to avoid re-encoding memory text.

```python
ranked = memory_reranker.rerank_embeddings(
    query_embedding=query_embedding,
    memory_embeddings=memory_embeddings,
    memory_ids=memory_ids,
    memory_texts=memory_texts,
    candidate_indices=candidate_indices,
    query=query,
    top_k=20,
)
```

The checkpoint and embeddings must use the same embedding dimension and embedding model family.

## Checkpoint

The recommended checkpoint layout is:

```text
checkpoints/convmemory-locomo-mpnet/
  config.json
  model.pt
```

`checkpoints/` is intentionally ignored by Git. Publish trained weights through a release artifact or model hosting service instead of committing them to the repository.

## Reproduce The Main Evaluation

```bash
python experiments/reproduce_locomo.py \
  --device cpu \
  --encoder-model sentence-transformers/all-mpnet-base-v2 \
  --embedding-cache results/cache/mpnet_embeddings.sqlite \
  --embedding-cache-key sentence-transformers/all-mpnet-base-v2 \
  --teacher-cache results/cache/repro_teacher_mpnet_top500_seed23.json \
  --raw-top-n 500 \
  --candidate-top-n 500 \
  --cross-top-n 500 \
  --eval-cross-encoder \
  --out results/reproduce_mpnet_full_top500_seed23
```

Set `--device cuda` for faster local runs when a GPU is available.

Run the cost benchmark:

```bash
python experiments/benchmark_cost.py \
  --device cpu \
  --checkpoint checkpoints/convmemory-locomo-mpnet \
  --encoder-model sentence-transformers/all-mpnet-base-v2 \
  --embedding-cache results/cache/mpnet_embeddings.sqlite \
  --embedding-cache-key sentence-transformers/all-mpnet-base-v2 \
  --candidate-top-n 500 \
  --cross-top-n 500 \
  --cascade-ce-top-n 50 \
  --max-test 100 \
  --out results/benchmark_cost_mpnet_100_top500
```

Run the LongMemEval zero-shot retrieval check:

```bash
python experiments/evaluate_longmemeval_zero_shot.py \
  --device cuda \
  --checkpoint checkpoints/convmemory-locomo-mpnet \
  --encoder-model sentence-transformers/all-mpnet-base-v2 \
  --embedding-cache results/cache/longmemeval_mpnet_embeddings.sqlite \
  --embedding-cache-key sentence-transformers/all-mpnet-base-v2 \
  --candidate-top-n 500 \
  --eval-cross-encoder \
  --out results/longmemeval_zero_shot_full_gpu
```

For the large-pool optimized path, add:

```bash
  --distractor-sessions 1000 \
  --window-mode candidate_local \
  --precache-lexical
```

## Project Status

ConvMemory is an early open-source research library. The current release focuses on:

- a simple public API;
- reusable pretrained checkpoint loading;
- memory reranking over text or precomputed embeddings;
- reproducible LoCoMo evaluation scripts;
- honest quality and latency reporting.

The model is useful today as a lightweight memory reranker, but it is still evolving. The next priorities are broader agent-memory benchmarks, cleaner checkpoint distribution, and further optimization of very large memory pools.

For release steps, see [docs/PUBLISHING.md](docs/PUBLISHING.md).

## License

MIT

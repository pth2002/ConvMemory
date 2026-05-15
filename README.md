# ConvMemory

Lightweight temporal reranking for long-term conversational and agent memory.

Current package version: `v0.2.0`.

ConvMemory sits between a fast vector retriever and an expensive cross-encoder. It is designed for memory systems where records are ordered over time, such as multi-session conversations, user profiles, agent scratchpads, and event histories.

It does not replace your vector database. It reranks the candidate memories your retriever already found, and can optionally expand the final memory context when your agent has room for a few additional evidence candidates.

```text
User query -> vector search top-k -> ConvMemory rerank/expand -> memory context for your agent
```

Research preview: ConvMemory can also act as a high-coverage candidate stage before a small cross-encoder pass. The current public package keeps this cascade path in `experiments/` while the stable API remains focused on `rerank` and `expand`.

## What's New In v0.2

ConvMemory v0.2 adds a public context-expansion API for agent memory systems:

- `retrieve(..., mode="rerank")`: the standard ConvMemory reranker.
- `retrieve(..., mode="expand")`: protect the strongest reranked memories, then fill a larger context budget with complementary candidates.
- `expand_context(...)` and `expand_context_embeddings(...)`: explicit APIs for systems that separate retrieval, memory storage, and prompt construction.

The v0.2 API is compatible with the existing LoCoMo MPNet checkpoint. No new checkpoint is required to use context expansion.

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

## Retrieval Modes

ConvMemory exposes two public modes:

| Mode | Use when | Output |
|---|---|---|
| `rerank` | You need the strongest ordered top-k memories. | A ConvMemory reranked list. |
| `expand` | Your agent can read a slightly wider memory context and missing evidence is costly. | The protected ConvMemory top-k plus complementary candidates. |

The expansion mode is intentionally conservative: it keeps the strongest reranked memories at the front, then fills the remaining context budget from complementary rankings such as raw dense retrieval, candidate-local temporal windows, and optional extra ConvMemory checkpoints.

## Results

Evaluation scope:

- These are retrieval-stage evaluations, not end-to-end answer generation benchmarks.
- Latency numbers measure online reranking after memory embeddings and memory-side indexes are available.
- Cross-encoder comparisons use `cross-encoder/ms-marco-MiniLM-L-6-v2` unless otherwise noted.
- Tables with different candidate pools, devices, or split averaging are reported separately.

LoCoMo retrieval-stage evaluation, full test split, MPNet embeddings, top500 candidate pool.

| Method | Test questions | Recall@10 | Hit@10 | MRR |
|---|---:|---:|---:|---:|
| Raw MPNet retrieval | 937 | 0.553 | 0.607 | 0.334 |
| Cross-encoder top500 | 937 | 0.749 | 0.790 | 0.614 |
| ConvMemory top500 | 937 | 0.768 | 0.821 | 0.554 |

Interpretation:

- ConvMemory substantially improves over raw dense retrieval.
- ConvMemory has higher Recall@10 and Hit@10 than the tested cross-encoder in this setup.
- The cross-encoder has stronger MRR, meaning it is better at placing the first relevant evidence near the top.

The practical sweet spot is to use ConvMemory as a low-cost memory reranker, or as a first reranking stage before a smaller cross-encoder pass.

## v0.2 Context Expansion

The v0.2 expansion API is not a new headline model score. It is a practical context-construction mode for agent memory systems:

```text
Keep the strongest ConvMemory memories -> add a few complementary candidates when the prompt budget allows it
```

In LoCoMo-style long-term memory retrieval, the average gains from fixed expansion policies are modest, so they should not be read as a major benchmark jump. The useful finding is mechanistic: expansion mainly helps when the base reranker misses the evidence entirely.

Mechanism check on LoCoMo seed 23 with `protected_k=10`, `context_budget=15`:

| Query bucket | Questions | Delta Recall | Delta Hit | Reading |
|---|---:|---:|---:|---|
| v0.1 already found all gold memories | 714 | -0.0025 | -0.0014 | mostly unchanged |
| v0.1 missed the evidence | 132 | +0.0587 | +0.0682 | main benefit |
| v0.1 found partial evidence | 91 | +0.0018 | -0.0110 | mixed |

This means v0.2 is best understood as a recall-oriented memory context expander. It is useful when an agent can read a slightly larger memory context and missing a key memory is more costly than including a few extra candidates. It is not meant to replace the main reranker or improve top-1 precision on every query.

The recommended default is simple: use `mode="rerank"` for strict top-k retrieval, and use `mode="expand"` only when your downstream agent can consume a wider memory context. On LongMemEval-S, where the v0.1 checkpoint is already close to saturated at larger k, expansion gives little or inconsistent benefit.

Additional zero-shot check: the LoCoMo-trained checkpoint was evaluated on LongMemEval-S, 500 questions, with no LongMemEval training. This is a retrieval-stage evaluation, not the full answer-generation leaderboard.

| Method | Questions | Recall@5 | Hit@5 | Recall@10 | Hit@10 | MRR | ms/query |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw MPNet retrieval | 500 | 0.823 | 0.902 | 0.905 | 0.946 | 0.783 | 0.0 |
| Cross-encoder top500 | 500 | 0.882 | 0.934 | 0.933 | 0.956 | 0.890 | 115.6 |
| ConvMemory zero-shot | 500 | 0.920 | 0.968 | 0.959 | 0.988 | 0.897 | 48.7 |

In this retrieval-stage setup, ConvMemory improves over raw MPNet and the tested cross-encoder on aggregate retrieval metrics while remaining faster than the tested cross-encoder on cached embeddings.

LongMemEval-S distractor stress test: each question's memory pool is expanded with sessions from other questions. ConvMemory and the cross-encoder rerank the raw top500 candidates.

| Memory pool | Raw R@10 / MRR | Cross-encoder R@10 / MRR / ms | ConvMemory R@10 / MRR / ms |
|---|---:|---:|---:|
| Original, avg 48 sessions | 0.905 / 0.783 | 0.933 / 0.890 / 115.6 | 0.959 / 0.897 / 48.7 |
| 500 sessions | 0.641 / 0.524 | 0.791 / 0.730 / 528.3 | 0.809 / 0.711 / 357.8 |
| 1000 sessions, top500 rerank | 0.548 / 0.442 | 0.730 / 0.671 / 519.9 | 0.739 / 0.623 / 862.9 |
| 1000 sessions, candidate-local + cached lexical | 0.548 / 0.442 | 0.730 / 0.671 / 519.9 | 0.729 / 0.618 / 89.1 |

The optimized 1000-session row uses candidate-local temporal windows and cached memory lexical signatures. It keeps Recall@10 roughly tied with the tested cross-encoder while being about 5.8x faster in online reranking latency. MRR is still lower, so the cross-encoder remains better at top-rank precision.

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

## Cascade Fusion Preview

The strongest current research-preview path is a two-stage cascade:

```text
vector top500 -> ConvMemory top100 -> small cross-encoder pass -> score fusion
```

This keeps ConvMemory as the high-recall temporal memory reranker and uses the cross-encoder only as a precision booster over a much smaller candidate pool.

LoCoMo test splits, seeds 7/11/23, MPNet embeddings, `ms-marco-MiniLM-L-6-v2` cross-encoder:

| Method | Recall@10 | Hit@10 | MRR@10 | CE pairs/query |
|---|---:|---:|---:|---:|
| Cross-encoder raw top500 | 0.7371 | 0.7842 | 0.5978 | 483.8 |
| ConvMemory balanced | 0.7798 | 0.8361 | 0.5698 | 0.0 |
| ConvMemory + CE fusion top100 | 0.7908 | 0.8425 | 0.6124 | 100.0 |

This result should not be read as "ConvMemory replaces cross-encoders." It shows a more practical pattern: ConvMemory can build a smaller, higher-coverage candidate pool, and a small cross-encoder pass can then focus on precision.

Latency benchmark on an RTX 4090, 600 LoCoMo test queries, 10 warmup queries per seed. Timing measures online reranking after embeddings and memory-side indexes are available.

| Method | Recall@10 | Hit@10 | MRR@10 | Mean ms/query | P95 ms/query | Speedup vs CE top500 |
|---|---:|---:|---:|---:|---:|---:|
| ConvMemory balanced | 0.7408 | 0.8083 | 0.5214 | 10.2 | 13.3 | 10.26x |
| Cross-encoder raw top500 | 0.7290 | 0.7750 | 0.5813 | 104.5 | 125.4 | 1.00x |
| ConvMemory + CE fusion top100 | 0.7487 | 0.8033 | 0.5887 | 38.2 | 43.9 | 2.74x |

The cascade numbers should be read as a research-preview result, not a new default API mode. They show that ConvMemory can reduce the number of cross-encoder pairs while preserving or improving retrieval quality in this retrieval-stage setting.

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

To build a wider memory context for an agent, use `mode="expand"`:

```python
context = model.retrieve(
    "When is the hiking trip?",
    memories,
    mode="expand",
    protected_k=10,
    top_k=15,
)
```

This protects the first 10 ConvMemory-ranked memories and fills the remaining 5 slots with complementary candidates. For explicit code, the same behavior is available as `expand_context(...)`.

## In An Agent Memory Pipeline

ConvMemory is usually called after vector search and before prompt construction.

```python
from convmemory import ConvMemory

memory_reranker = ConvMemory.from_pretrained(
    "checkpoints/convmemory-locomo-mpnet",
    device="cuda",
)

def retrieve_agent_memory(query, memory_store, top_k=15):
    candidates = memory_store.vector_search(query, top_k=500)

    ranked = memory_reranker.retrieve(
        query=query,
        memories=candidates,
        mode="expand",
        protected_k=10,
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

For context expansion over precomputed embeddings:

```python
context = memory_reranker.expand_context_embeddings(
    query_embedding=query_embedding,
    memory_embeddings=memory_embeddings,
    memory_ids=memory_ids,
    memory_texts=memory_texts,
    candidate_indices=candidate_indices,
    query=query,
    protected_k=10,
    context_budget=15,
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

The pretrained LoCoMo MPNet checkpoint is available as a release asset:

- [convmemory-locomo-mpnet.zip](https://github.com/pth2002/ConvMemory/releases/download/v0.1.0/convmemory-locomo-mpnet.zip)

This checkpoint was released with `v0.1.0` and remains the recommended checkpoint for `v0.2.0`. The v0.2 context-expansion API reuses the same reranker weights and does not require a separate model file.

Download and extract the archive from the repository root:

```bash
mkdir -p checkpoints
unzip convmemory-locomo-mpnet.zip -d checkpoints
```

On Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force -Path checkpoints
Expand-Archive .\convmemory-locomo-mpnet.zip -DestinationPath .\checkpoints -Force
```

The resulting layout should be:

```text
checkpoints/convmemory-locomo-mpnet/
  config.json
  model.pt
```

After extracting the checkpoint, verify loading with:

```bash
python examples/load_pretrained.py
```

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
  --device cuda \
  --checkpoint checkpoints/convmemory-locomo-mpnet \
  --encoder-model sentence-transformers/all-mpnet-base-v2 \
  --embedding-cache results/cache/mpnet_embeddings.sqlite \
  --embedding-cache-key sentence-transformers/all-mpnet-base-v2 \
  --candidate-top-n 500 \
  --cascade-top-n 50 \
  --limit 100 \
  --out results/benchmark_cost_mpnet_100_top500
```

Run the cascade-fusion research preview:

```bash
python experiments/v035_ce_fusion.py \
  --device cuda \
  --checkpoint checkpoints/convmemory-locomo-mpnet \
  --encoder-model sentence-transformers/all-mpnet-base-v2 \
  --embedding-cache results/cache/mpnet_embeddings.sqlite \
  --embedding-cache-key sentence-transformers/all-mpnet-base-v2 \
  --cross-encoder-model cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --seeds 7 11 23 \
  --out results/v035/ce_fusion
```

Run the latency benchmark:

```bash
python experiments/v036_latency_benchmark.py \
  --device cuda \
  --checkpoint checkpoints/convmemory-locomo-mpnet \
  --encoder-model sentence-transformers/all-mpnet-base-v2 \
  --embedding-cache results/cache/mpnet_embeddings.sqlite \
  --embedding-cache-key sentence-transformers/all-mpnet-base-v2 \
  --cross-encoder-model cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --seeds 7 11 23 \
  --limit 200 \
  --warmup 10 \
  --out results/v036/latency_benchmark
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

ConvMemory is an early open-source research library. The current package release is `v0.2.0` and focuses on:

- a simple public API;
- reusable pretrained checkpoint loading;
- memory reranking over text or precomputed embeddings;
- conservative memory context expansion for agent prompt construction;
- reproducible LoCoMo evaluation scripts;
- honest quality and latency reporting.

The model is useful today as a lightweight memory reranker, but it is still evolving. The next priorities are broader agent-memory benchmarks, cleaner checkpoint distribution, further optimization of very large memory pools, and turning the cascade-fusion research path into a clean optional API.

## License

MIT

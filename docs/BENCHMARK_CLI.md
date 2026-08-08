# `convmemory benchmark`

Measure what ConvMemory does to *your* retrieval, on *your* memories, before
deciding whether to keep it.

```bash
pip install "convmemory[encode]"
convmemory benchmark --queries queries.jsonl --memories memories.jsonl
```

The CLI encodes your memory texts, so it needs the `[encode]` extra.

From a source checkout without reinstalling, use the module form:

```bash
python -m convmemory.cli benchmark --queries queries.jsonl --memories memories.jsonl
```

## Input format

**`memories.jsonl`** — one memory per line, in the order your store keeps them.
Order matters: ConvMemory's window features read neighboring memories.

```json
{"id": "m1", "text": "User: we switched the analytics store to ClickHouse", "group": "user-42"}
{"id": "m2", "text": "Assistant: noted, ClickHouse for analytics", "group": "user-42"}
```

| Field | Required | Meaning |
|---|---|---|
| `text` | yes | the memory content |
| `id` | no | defaults to the line index |
| `group` | no | candidate-pool scope, e.g. a user or session id |

**`queries.jsonl`** — one evaluation query per line.

```json
{"query": "what database do we use for analytics?", "gold_ids": ["m1"], "group": "user-42"}
```

| Field | Required | Meaning |
|---|---|---|
| `query` | yes | the query text |
| `gold_ids` | yes | the memory ids that should be retrieved (`gold_id` also accepted) |
| `group` | no | must match a group in the memory file |

`group` is what keeps the benchmark honest for multi-tenant stores: each query
is scored against its own group's memories, not against everyone's.

## Output

```text
memories: 419 in 1 group(s), pool size 419-419
queries:  60
indexed 419 memories in 0.8s (one-off, not counted in per-query latency)

query encoding (shared by both paths): 7.97 ms/query

Before ConvMemory (dense only)
  Recall@5   0.3069     Hit@5   0.4167
  Recall@10  0.4431     Hit@10  0.5833
  MRR       0.3011
  Latency   0.26 ms/query

After ConvMemory
  Recall@5   0.5569     Hit@5   0.6667
  Recall@10  0.6764     Hit@10  0.8000
  MRR       0.4929
  Latency   15.09 ms/query

Delta
  Recall@5     0.3069  ->  0.5569   (+0.2500)  better
  Recall@10    0.4431  ->  0.6764   (+0.2333)  better
  MRR          0.3011  ->  0.4929   (+0.1918)  better
  Latency      0.26 ms  ->  15.09 ms
```

That run is a single held-out LoCoMo conversation on an RTX 4060 Laptop, shown
so you can compare shapes, not as a benchmark claim.

The "before" path is cosine similarity in the checkpoint's own embedding space,
so both rows use the same encoder and the only difference is the reranker. If
your production retriever is a different (usually stronger) embedding model,
the "before" row here is not your production baseline — retrain or compare
against your own retriever before drawing conclusions.

## Options

| Flag | Default | Meaning |
|---|---|---|
| `--checkpoint` | `Purdy0228/ConvMemory-LoCoMo-MPNet` | Hub id or local path |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--k` | `5 10` | cutoffs for Recall@k and Hit@k |
| `--top-k` | `10` | results kept from the reranker |
| `--candidate-pool` | `500` | how many dense candidates get reranked |
| `--limit` | all | run only the first N queries |
| `--progress` | off | print progress every N queries |
| `--json` | none | write the summary to a JSON file |

## Interpreting the result

- **Recall@k up, latency up ~15 ms/query**: the expected outcome on pools of
  hundreds of memories.
- **No change**: usually a small pool. If dense already puts the gold memory in
  the top 5, there is nothing for a reranker to recover. The CLI prints a note
  when your pools are under 100 memories.
- **Worse**: real and worth reporting. The most common causes are a candidate
  pool that is already precision-ranked by a stronger retriever, or a domain far
  from conversational/agent memory (see the MuSiQue row in
  [BENCHMARKS.md](BENCHMARKS.md)). Please open an issue with the shape of your
  data and what you measured.

The `--json` output is designed to be pasted into an issue.

# `convmemory benchmark`

Measure what ConvMemory does to *your* memories, without writing an eval harness.

```bash
pip install "convmemory[encode]"
convmemory benchmark --queries queries.jsonl --memories memories.jsonl
```

The CLI encodes your memory texts, so it needs the `[encode]` extra.

## Reading the result

The delta measures what the reranker adds on top of dense retrieval **in the
checkpoint's own embedding space**, on your memories, at a measured latency cost.
The "before" row is cosine similarity in that space (`all-mpnet-base-v2` for the
released v1 checkpoint), so both rows share an encoder and the reranker is the
only variable.

Running ConvMemory in a different embedding space means training a checkpoint
there, and that gain travels well: retrained on BGE-large and E5-large,
ConvMemory held **+0.105** and **+0.089** Recall@10 over raw dense *in those
spaces* — more than its MPNet gain. See "Strong-backbone retraining" in
[BENCHMARKS.md](BENCHMARKS.md), and [TRAINING.md](TRAINING.md) for the recipe.

**Bring labelled `gold_ids`** for each query — the ids a good retrieval should
surface. That annotated set is what turns this into a measurement.

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

Before ConvMemory (dense only, sentence-transformers/all-mpnet-base-v2)
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

How to read this: the 'before' row is cosine similarity in
sentence-transformers/all-mpnet-base-v2, not your production retriever. ...
```

That run is a single held-out LoCoMo conversation on an RTX 4060 Laptop, shown
so you can compare shapes, not as a benchmark claim.

Both rows use the same encoder, so the reranker is the only variable between
them.

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
- **Flat or negative**: worth an issue. The usual causes are a candidate pool
  already precision-ranked by a strong retriever, or a domain further from
  conversational and agent memory (see the dataset table in
  [BENCHMARKS.md](BENCHMARKS.md)). Send the shape of your data and what you
  measured — that is the input that improves the next checkpoint.

The `--json` output is designed to be pasted into an issue.

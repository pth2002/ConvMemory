# Architecture Notes

This note summarizes the main components behind ConvMemory. Most users only need
the public `ConvMemory` API.

Runtime code lives in the installable `convmemory/` package:

- `convmemory/api.py`: user-facing `ConvMemory` wrapper.
- `convmemory/reranker.py`: embedding-level reranking and candidate-local windowing.
- `convmemory/encoder.py`: temporal Conv/Mixer window encoder.
- `convmemory/scoring.py`: CE-lite scorer, lexical cache, and score fusion helpers.
- `convmemory/metrics.py`: small retrieval metrics used by examples and experiments.

The root-level experiment helpers are used by the reproduction scripts. They are
kept outside the installable wheel so the library API stays small.

## 1. Temporal Conv/Mixer Encoder

The core memory encoder reads a short temporal window of memory embeddings:

```text
[memory_t-2, memory_t-1, memory_t, memory_t+1, memory_t+2]
```

It uses lightweight temporal convolution plus Mixer-style token/channel mixing to model local event-chain structure.

## 2. DCA Router Signal

The current DCA component is a lightweight block router signal.

It scores coarse memory blocks and adds a block-level temporal/context signal to the CE-lite scorer. In the current mainline, DCA is useful as an auxiliary feature, not as the whole retrieval mechanism.

## 3. Lexical Features

ConvMemory includes small lexical overlap features:

- token overlap
- token recall against the query
- bigram overlap
- bigram recall against the query

These features help recover lexical anchors that pure embedding similarity can miss.

## 4. CE-lite Scorer

The CE-lite scorer fuses:

- query embedding
- candidate memory embedding
- query-memory interaction features
- raw dense score
- ConvMemory window score
- rank / temporal position
- DCA router score
- lexical features

It is not a token-level cross-encoder. It is much cheaper and operates over precomputed embeddings plus lightweight side features.

## 5. Raw Dense Score Fusion

The final ranking can retain a small amount of raw retriever score:

```text
final_score = raw_weight * raw_score + (1 - raw_weight) * convmemory_score
```

The current best setting usually keeps `raw_weight` near `0` to `0.025`.

The current module description is:

```text
Temporal Conv/Mixer + DCA router signal + lexical CE-lite reranking
```

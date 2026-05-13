# Publishing ConvMemory

This is the recommended first public release flow.

## 1. Create A Fresh Git Repository

From the project root:

```bash
git init
git add convmemory examples experiments docs .github
git add convmem_ce_lite.py convmem_chain_benchmark.py convmem_locomo_benchmark.py convmem_longmemeval.py locomo_crossencoder_baseline.py
git add data/README.md checkpoints/README.md README.md LICENSE pyproject.toml requirements.txt .gitignore
git status
```

Check `git status` carefully. Do not commit:

- `data/*.json`
- `results/`
- `checkpoints/*/model.pt`
- embedding caches such as `*.sqlite`
- local run logs or exploratory scripts

## 2. Verify The Package

```bash
pip install -e . --no-deps
python -m compileall -q convmemory
python examples/basic_api.py
```

If dependencies are not already installed, use:

```bash
pip install -e .
```

## 3. Publish The Checkpoint Separately

Do not commit model weights to Git. Publish this folder as a release asset or on
a model hub:

```text
checkpoints/convmemory-locomo-mpnet/
  config.json
  model.pt
```

After uploading it, replace the placeholder links in `pyproject.toml` and add the
checkpoint download URL to `README.md`.

## 4. Suggested GitHub Release Text

```text
ConvMemory v0.1.0 is a research-preview memory reranker for long-term
conversational and agent memory.

Highlights:
- Temporal Conv/Mixer memory-window scoring.
- CE-lite scoring over precomputed embeddings.
- Candidate-local reranking path for large memory pools.
- LoCoMo and LongMemEval-S retrieval-stage evaluation scripts.

Known limitation:
- ConvMemory approaches cross-encoder Recall@10 in the reported large-pool
  stress test, but cross-encoders still have stronger MRR/top-1 precision.
```

## 5. Repository Description

Use something like:

```text
Lightweight temporal reranking for long-term conversational and agent memory.
```

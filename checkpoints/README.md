# Checkpoints

Place ConvMemory checkpoints here when running local examples or evaluations.

Expected layout:

```text
checkpoints/convmemory-locomo-mpnet/
  config.json
  model.pt
```

Model weights are intentionally not committed to Git. For a public release, attach
the checkpoint as a GitHub Release asset or host it on a model hub, then update the
README with the download link.

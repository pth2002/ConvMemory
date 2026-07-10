# ConvMemory v3 Release Sanity

Verified on 2026-07-10 for package `convmemory==0.6.2`.

## Package

| Check | Result |
|---|---|
| Wheel built | `dist/0.6.2/convmemory-0.6.2-py3-none-any.whl` |
| Source distribution built | `dist/0.6.2/convmemory-0.6.2.tar.gz` |
| Wheel size | `55,596` bytes |
| Source distribution size | `68,651` bytes |
| No checkpoint/results/experiment artifacts in wheel | pass |
| No checkpoint/results/experiment artifacts in source distribution | pass |
| Clean install imports v3 validity exports | pass |

## Tests

| Check | Result |
|---|---|
| Full package test suite | `58 passed` |
| v3 validity context tests | `20 passed` |
| Hub loading and validity-focused tests | `22 passed` |

## Source Budget

The integrated v3 path now protects a bounded result prefix and selects at most
one update source per target by default. A 500-memory smoke with a protected
prefix of 10 produced exactly 10 validity scorer calls and preserved the full
context order. Applications with an update index can bypass internal selection
with `validity_source_map`.

The `0.6.2` wheel was installed in a clean virtual environment outside the
repository source path. The installed metadata reported `0.6.2`, the public
`ConvMemory.retrieve` signature exposed `validity_source_map`, and an explicit
query/source/target context-scoring smoke passed.

## Checkpoint

| Check | Result |
|---|---|
| Hub repository | `Purdy0228/ConvMemory-v3-Validity-Context` |
| Current Hub repository commit after model-card update | `3f1487d6d21287dd6314d9bf534bbc62871f325d` |
| Checkpoint upload commit | `0883a43fe6df608030ebe9ec29286280e83c857c` |
| `cross_encoder/model.safetensors` SHA256 | `446ee0cf6df4a8967e1a78c46d2ff3a2d777de65efbf475d2278d99468faa8d9` |
| `validity_config.json` SHA256 | `81eddb5f2ff4545dcf4b7655fedd1f7cf846248ad8962394195e6960a2e07849` |
| HF model card metadata | pass |

## Public Load Path

Cold-start user path was verified with:

- a temporary virtual environment;
- a temporary `HF_HOME`;
- a temporary `CONVMEMORY_CACHE`;
- package installation from the built wheel;
- `ValidityEvidenceModule.from_pretrained("Purdy0228/ConvMemory-v3-Validity-Context", device="cpu")`;
- one public Hub download and scoring call.

Smoke pair score:

```text
0.966189
```

## Mode Contracts

| Mode | Release contract |
|---|---|
| `off` / `None` | no v3 participation |
| `context` | attaches validity metadata without changing rank order or candidate set |
| `demote` | opt-in; may reorder while preserving the candidate set |

The public default for v3 usage is `context`.

# ConvMemory v3 Release Sanity

Verified on 2026-06-10 for package `convmemory==0.6.0`.

## Package

| Check | Result |
|---|---|
| Wheel built | `dist/convmemory-0.6.0-py3-none-any.whl` |
| Source distribution built | `dist/convmemory-0.6.0.tar.gz` |
| Wheel size | `47,504` bytes |
| Source distribution size | `56,928` bytes |
| No checkpoint/results/experiment artifacts in wheel | pass |
| No checkpoint/results/experiment artifacts in source distribution | pass |
| Clean install imports v3 validity exports | pass |

## Tests

| Check | Result |
|---|---|
| Full package test suite | `41 passed` |
| v3 validity context tests | `12 passed` |
| Hub loading and validity-focused tests | `14 passed` |

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

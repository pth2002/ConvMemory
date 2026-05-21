"""Optional Hugging Face Hub path resolution helpers."""

from __future__ import annotations

from pathlib import Path

try:
    from huggingface_hub import snapshot_download as _hf_snapshot_download
except Exception:  # pragma: no cover - exercised when optional dep is absent
    _hf_snapshot_download = None


def looks_like_hub_id(path: str | Path) -> bool:
    """Return whether a missing path looks like a `namespace/repo` Hub id."""

    text = str(path).replace("\\", "/").strip()
    if not text or "://" in text or ":" in text:
        return False
    if text.startswith(("/", "./", "../", "~")):
        return False
    parts = text.split("/")
    return len(parts) == 2 and all(parts)


def resolve_checkpoint_path(path: str | Path, *, repo_type: str = "model") -> Path:
    """Resolve a local checkpoint path or download a Hugging Face Hub repo id."""

    candidate = Path(path)
    if candidate.exists():
        return candidate
    if not looks_like_hub_id(path):
        return candidate
    if _hf_snapshot_download is None:
        raise ValueError(
            "Checkpoint path does not exist and looks like a Hugging Face Hub "
            "repo id, but `huggingface_hub` is not installed. Install it with "
            "`pip install huggingface_hub` or pass a local checkpoint path."
        )
    try:
        return Path(_hf_snapshot_download(repo_id=str(path), repo_type=repo_type))
    except Exception as exc:
        raise ValueError(
            f"Could not download Hugging Face Hub checkpoint repo '{path}'. "
            "Pass a local checkpoint path or verify repo access."
        ) from exc

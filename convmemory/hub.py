"""Optional Hugging Face Hub path resolution helpers."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

try:
    from huggingface_hub import HfApi as _HfApi
    from huggingface_hub import hf_hub_download as _hf_hub_download
    from huggingface_hub import snapshot_download as _hf_snapshot_download
except Exception:  # pragma: no cover - exercised when optional dep is absent
    _HfApi = None
    _hf_hub_download = None
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
        try:
            return _download_hub_repo_without_snapshot_symlinks(str(path), repo_type=repo_type)
        except Exception as fallback_exc:
            raise ValueError(
                f"Could not download Hugging Face Hub checkpoint repo '{path}'. "
                "Pass a local checkpoint path or verify repo access."
            ) from fallback_exc


def _download_hub_repo_without_snapshot_symlinks(repo_id: str, *, repo_type: str = "model") -> Path:
    """Download a small checkpoint repo without relying on snapshot symlinks.

    Some Windows environments do not allow symlink creation. Hugging Face Hub's
    snapshot cache can fail there before it has a chance to fall back cleanly, so
    ConvMemory downloads each file into a plain local directory as a compatibility
    path.
    """

    if _HfApi is None or _hf_hub_download is None:
        raise RuntimeError("huggingface_hub is not installed")
    cache_root = os.environ.get("CONVMEMORY_CACHE")
    if cache_root:
        base = Path(cache_root)
    else:
        base = Path.home() / ".cache" / "convmemory"
    digest = hashlib.sha1(repo_id.encode("utf-8")).hexdigest()[:12]
    target = base / "hub" / f"{repo_id.replace('/', '--')}-{digest}"
    target.mkdir(parents=True, exist_ok=True)

    files = _HfApi().list_repo_files(repo_id=repo_id, repo_type=repo_type)
    for filename in files:
        if filename.endswith("/"):
            continue
        _hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            local_dir=target,
        )
    return target

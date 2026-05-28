"""Protected top-k token-evidence reranker for ConvMemory v2.

The reranker is intentionally opt-in. It scores only a protected prefix from an
existing ConvMemory ranking and never accepts gold-defining inference fields.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np
from sentence_transformers import CrossEncoder

from .hub import resolve_checkpoint_path


FORBIDDEN_FIELDS = frozenset(
    {
        "gold",
        "gold_ids",
        "is_current",
        "is_latest",
        "is_stale",
        "stale",
        "answer",
        "answer_text",
        "ce_score",
        "mxbai_score",
        "teacher_score",
        "gpt_label",
        "entity_id",
        "slot_id",
    }
)

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass(frozen=True)
class EvidenceRerankerConfig:
    """Configuration for the protected top-k evidence reranker."""

    top_k: int = 10
    max_length: int = 256
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class EvidenceReranker:
    """Token-evidence scorer for reordering a protected ConvMemory prefix.

    Inputs are limited to query text, candidate memory text, candidate id, and
    optional time/position metadata. Gold ids, answer text, CE/teacher scores,
    and current/stale labels are rejected at inference time.
    """

    def __init__(
        self,
        config: EvidenceRerankerConfig | None = None,
        *,
        cross_encoder=None,
        device: str | None = None,
    ) -> None:
        self.config = config or EvidenceRerankerConfig()
        self.device = device
        self.cross_encoder = cross_encoder
        if self.cross_encoder is None:
            self.cross_encoder = CrossEncoder(
                self.config.cross_encoder_model,
                num_labels=1,
                max_length=int(self.config.max_length),
                device=device,
            )

    @classmethod
    def from_pretrained(cls, path_or_hub_id, device: str | None = None):
        """Load an evidence reranker from a local path or Hugging Face Hub id."""

        path = resolve_checkpoint_path(path_or_hub_id)
        config_path = path / "evidence_reranker_config.json"
        if config_path.exists():
            config = EvidenceRerankerConfig(
                **json.loads(config_path.read_text(encoding="utf-8"))
            )
        else:
            config = EvidenceRerankerConfig()
        model_path = path / "cross_encoder"
        if not model_path.exists():
            model_path = path
        cross_encoder = CrossEncoder(
            str(model_path),
            num_labels=1,
            max_length=int(config.max_length),
            device=device,
        )
        return cls(config=config, cross_encoder=cross_encoder, device=device)

    def save_pretrained(self, path) -> None:
        """Save the reranker config and CrossEncoder checkpoint."""

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "evidence_reranker_config.json").write_text(
            json.dumps(asdict(self.config), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        model_path = path / "cross_encoder"
        model_path.mkdir(parents=True, exist_ok=True)
        if not hasattr(self.cross_encoder, "save"):
            raise ValueError("cross_encoder must provide save(path) to be serialized")
        self.cross_encoder.save(str(model_path))

    def score(
        self,
        query: str,
        candidates: Sequence[Mapping[str, object]],
        device: str | None = None,
    ) -> list[float]:
        """Score candidates with the v361 query/memory text format.

        Each candidate must contain `id` and `text`. Optional `position` or
        `time` is used for the `MEMORY_TIME` prefix. Any forbidden inference
        field raises `ValueError`.
        """

        checked = [self._validate_candidate(candidate) for candidate in candidates]
        if not checked:
            return []
        positions = [self._candidate_position(candidate, idx) for idx, candidate in enumerate(checked)]
        query_time = max(positions) if positions else 0.0
        query_text = f"QUERY_TIME: {query_time:.0f}. {query}"
        pairs = [
            (
                query_text,
                f"MEMORY_TIME: {position:.0f}. {candidate['text']}",
            )
            for candidate, position in zip(checked, positions)
        ]
        if device is not None and hasattr(getattr(self.cross_encoder, "model", None), "to"):
            self.cross_encoder.model.to(device)
        scores = self.cross_encoder.predict(
            pairs,
            batch_size=min(32, max(1, len(pairs))),
            show_progress_bar=False,
        )
        return [float(x) for x in np.asarray(scores, dtype=np.float32).reshape(-1)]

    @staticmethod
    def _validate_candidate(candidate: Mapping[str, object]) -> dict[str, object]:
        keys = {str(key) for key in candidate.keys()}
        forbidden = sorted(keys & FORBIDDEN_FIELDS)
        if forbidden:
            raise ValueError(f"field '{forbidden[0]}' is not allowed at inference")
        if "id" not in candidate:
            raise ValueError("candidate must contain an 'id' field")
        if "text" not in candidate:
            raise ValueError("candidate must contain a 'text' field")
        return dict(candidate)

    @staticmethod
    def _candidate_position(candidate: Mapping[str, object], fallback: int) -> float:
        for key in ("position", "time"):
            if key in candidate:
                try:
                    return float(candidate[key])
                except (TypeError, ValueError):
                    break
        matches = _NUMBER_RE.findall(str(candidate.get("id", "")))
        if matches:
            return float(matches[-1])
        return float(fallback)

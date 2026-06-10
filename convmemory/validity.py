import json
from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Callable, Optional, Sequence

import numpy as np
from sentence_transformers import CrossEncoder

from .hub import resolve_checkpoint_path
from .reranker import RerankResult


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


@dataclass
class ValidityEvidenceConfig:
    """Configuration for the v3 validity context layer.

    The stable default is context annotation: it adds structured validity
    evidence for downstream agents without mutating the ranking.
    """

    mode_default: str = "context"
    source_policy: str = "top1"
    max_sources_per_candidate: int = 1
    context_threshold: float = 0.05
    demote_threshold: float = 0.35
    demote_score_scale: float = 1.0
    max_length: int = 192
    cross_encoder_batch_size: int = 32
    cross_encoder_model: Optional[str] = None
    cross_encoder_num_labels: int = 1


@dataclass
class ValidityAnnotation:
    status: str
    confidence: float
    action: str
    source_evidence: list
    context_note: str

    def to_dict(self):
        return {
            "status": self.status,
            "confidence": float(self.confidence),
            "action": self.action,
            "source_evidence": list(self.source_evidence),
            "context_note": self.context_note,
        }


class ValidityEvidenceModule:
    """Validity context layer for ConvMemory v3.

    This module is intentionally conservative. In `context` mode it annotates
    results with update evidence and never changes their order. In `demote`
    mode it may reorder by applying a score penalty, but it never removes or
    adds candidates.
    """

    def __init__(
        self,
        config: Optional[ValidityEvidenceConfig] = None,
        scorer: Optional[Callable[[str, dict, dict], float]] = None,
        cross_encoder=None,
        device: Optional[str] = None,
    ):
        self.config = config or ValidityEvidenceConfig()
        self.scorer = scorer
        self.device = device
        self.cross_encoder = cross_encoder
        if self.cross_encoder is None and self.config.cross_encoder_model:
            self.cross_encoder = CrossEncoder(
                self.config.cross_encoder_model,
                num_labels=int(self.config.cross_encoder_num_labels),
                max_length=int(self.config.max_length),
                device=device,
            )

    def save_pretrained(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "convmemory_validity_evidence",
            "version": 1,
            "config": asdict(self.config),
        }
        (path / "validity_config.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if self.cross_encoder is not None:
            model_path = path / "cross_encoder"
            model_path.mkdir(parents=True, exist_ok=True)
            if not hasattr(self.cross_encoder, "save"):
                raise ValueError("cross_encoder must provide save(path) to be serialized")
            self.cross_encoder.save(str(model_path))

    @classmethod
    def from_pretrained(cls, path_or_hub_id, device=None):
        path = resolve_checkpoint_path(path_or_hub_id)
        payload = json.loads((path / "validity_config.json").read_text(encoding="utf-8"))
        config = ValidityEvidenceConfig(**payload["config"])
        model_path = path / "cross_encoder"
        cross_encoder = None
        if model_path.exists():
            cross_encoder = CrossEncoder(
                str(model_path),
                num_labels=int(config.cross_encoder_num_labels),
                max_length=int(config.max_length),
                device=device,
            )
        elif config.cross_encoder_model:
            cross_encoder = CrossEncoder(
                config.cross_encoder_model,
                num_labels=int(config.cross_encoder_num_labels),
                max_length=int(config.max_length),
                device=device,
            )
        return cls(config, cross_encoder=cross_encoder, device=device)

    def annotate(
        self,
        *,
        query: str,
        results: Sequence[RerankResult],
        memories: Sequence[dict],
        mode: str = "context",
    ):
        mode = self._normalize_mode(mode)
        self._validate_memories(memories)
        memory_by_id = {str(memory.get("id", idx)): memory for idx, memory in enumerate(memories)}
        top_sources = self._top_sources(query=query, results=results, memories=memories, memory_by_id=memory_by_id)
        annotations = []
        for result in results:
            target = memory_by_id.get(result.memory_id)
            if target is None:
                target = {"id": result.memory_id, "text": result.text or ""}
            source, score = top_sources.get(result.memory_id, (None, 0.0))
            annotations.append(self._build_annotation(target, source, score, mode))
        return annotations

    def score_evidence_pairs(self, pairs: Sequence[dict]) -> list[float]:
        """Score explicit query/source/target evidence pairs in batches.

        Each item must contain `query`, `target`, and `source`. `target` and
        `source` may be dictionaries with a `text` field or plain strings.
        This method is useful when evidence retrieval has already selected a
        source per target and callers want packaged v3 scoring without the
        source-selection step.
        """
        normalized = []
        for pair in pairs:
            target = pair.get("target", {})
            source = pair.get("source", {})
            if not isinstance(target, dict):
                target = {"text": str(target)}
            if not isinstance(source, dict):
                source = {"text": str(source)}
            normalized.append(
                {
                    "query": str(pair.get("query", "")),
                    "target": target,
                    "source": source,
                }
            )
        return self._score_pair_batch(normalized)

    def apply(
        self,
        *,
        query: str,
        results: Sequence[RerankResult],
        memories: Sequence[dict],
        mode: str = "context",
    ):
        mode = self._normalize_mode(mode)
        annotations = self.annotate(query=query, results=results, memories=memories, mode=mode)
        copied = [
            RerankResult(
                memory_id=result.memory_id,
                score=float(result.score),
                raw_score=float(result.raw_score),
                rank=int(result.rank),
                text=result.text,
                validity=annotation.to_dict(),
            )
            for result, annotation in zip(results, annotations)
        ]
        if mode == "context":
            return copied

        adjusted = []
        for result in copied:
            validity = result.validity or {}
            penalty = 0.0
            if validity.get("action") == "demote":
                penalty = float(validity.get("confidence", 0.0)) * float(
                    self.config.demote_score_scale
                )
            adjusted.append(
                RerankResult(
                    memory_id=result.memory_id,
                    score=float(result.score) - penalty,
                    raw_score=float(result.raw_score),
                    rank=int(result.rank),
                    text=result.text,
                    validity=validity,
                )
            )
        return [
            RerankResult(
                memory_id=result.memory_id,
                score=result.score,
                raw_score=result.raw_score,
                rank=rank,
                text=result.text,
                validity=result.validity,
            )
            for rank, result in enumerate(
                sorted(adjusted, key=lambda item: item.score, reverse=True),
                start=1,
            )
        ]

    def _top_sources(
        self,
        *,
        query: str,
        results: Sequence[RerankResult],
        memories: Sequence[dict],
        memory_by_id: dict,
    ):
        pair_specs = []
        fallback = {}
        for result in results:
            target = memory_by_id.get(result.memory_id)
            if target is None:
                target = {"id": result.memory_id, "text": result.text or ""}
            target_id = str(target.get("id", ""))
            candidates = []
            for memory in memories:
                if str(memory.get("id", "")) == target_id:
                    continue
                candidates.append(memory)
                pair_specs.append(
                    {
                        "result_id": result.memory_id,
                        "query": query,
                        "target": target,
                        "source": memory,
                    }
                )
            if not candidates:
                fallback[result.memory_id] = (None, 0.0)

        if not pair_specs:
            return fallback

        scores = self._score_pair_batch(pair_specs)
        best = dict(fallback)
        for spec, score in zip(pair_specs, scores):
            result_id = spec["result_id"]
            current = best.get(result_id)
            if current is None or float(score) > float(current[1]):
                best[result_id] = (spec["source"], float(score))
        return best

    def _score_source(self, *, query: str, target: dict, source: dict) -> float:
        return self._score_pair_batch([{"query": query, "target": target, "source": source}])[0]

    def _score_pair_batch(self, pairs: Sequence[dict]) -> list[float]:
        if not pairs:
            return []
        if self.scorer is not None:
            return [
                self._to_probability(
                    float(self.scorer(str(pair["query"]), pair["target"], pair["source"]))
                )
                for pair in pairs
            ]
        if self.cross_encoder is not None:
            ce_pairs = [
                self._ce_pair(
                    query=str(pair["query"]),
                    target=pair["target"],
                    source=pair["source"],
                )
                for pair in pairs
            ]
            scores = self.cross_encoder.predict(
                ce_pairs,
                batch_size=int(self.config.cross_encoder_batch_size),
                show_progress_bar=False,
            )
            score_array = np.asarray(scores)
            if score_array.ndim >= 2:
                score_array = score_array[..., -1]
            return [
                self._to_probability(float(score))
                for score in score_array.reshape(-1).tolist()
            ]
        out = []
        for pair in pairs:
            query = str(pair["query"])
            target = pair["target"]
            source = pair["source"]
            target_text = str(target.get("text", ""))
            source_text = str(source.get("text", ""))
            query_tokens = self._tokens(query)
            target_tokens = self._tokens(target_text)
            source_tokens = self._tokens(source_text)
            source_target = self._jaccard(source_tokens, target_tokens)
            source_query = self._jaccard(source_tokens, query_tokens)
            recency_bonus = 0.0
            target_pos = self._position(target)
            source_pos = self._position(source)
            if target_pos is not None and source_pos is not None and source_pos > target_pos:
                recency_bonus = 0.1
            out.append(min(1.0, 0.75 * source_target + 0.25 * source_query + recency_bonus))
        return out

    @staticmethod
    def _ce_pair(*, query: str, target: dict, source: dict) -> tuple[str, str]:
        return (
            "USER_QUERY:\n"
            f"{query}\n\n"
            "SOURCE_EVIDENCE:\n"
            f"{source.get('text', '')}\n\n"
            "TASK: Decide whether the target memory should be demoted for this user query.",
            f"TARGET_MEMORY:\n{target.get('text', '')}",
        )

    @staticmethod
    def _to_probability(value: float) -> float:
        if 0.0 <= value <= 1.0:
            return float(value)
        return float(1.0 / (1.0 + np.exp(-value)))

    def _build_annotation(self, target, source, score: float, mode: str):
        confidence = float(max(0.0, min(1.0, score)))
        has_context = source is not None and confidence >= float(self.config.context_threshold)
        if not has_context:
            return ValidityAnnotation(
                status="unknown",
                confidence=confidence,
                action="keep",
                source_evidence=[],
                context_note="No strong update evidence found.",
            )

        source_text = str(source.get("text", ""))
        source_id = str(source.get("id", ""))
        evidence = {
            "id": source_id,
            "text": source_text,
            "score": confidence,
        }
        if "position" in source:
            evidence["position"] = source.get("position")
        if "time" in source:
            evidence["time"] = source.get("time")

        if mode == "demote" and confidence >= float(self.config.demote_threshold):
            action = "demote"
            status = "possibly_outdated"
        else:
            action = "context"
            status = "possibly_outdated"
        target_text = str(target.get("text", ""))
        note = (
            "Potential update evidence found for this memory: "
            f"{source_text[:160]}"
        )
        if not target_text:
            note = f"Potential update evidence found: {source_text[:160]}"
        return ValidityAnnotation(
            status=status,
            confidence=confidence,
            action=action,
            source_evidence=[evidence],
            context_note=note,
        )

    @staticmethod
    def _normalize_mode(mode):
        mode = str(mode).lower().strip()
        if mode not in {"context", "demote"}:
            raise ValueError("validity mode must be 'context' or 'demote'")
        return mode

    @classmethod
    def _validate_memories(cls, memories):
        for memory in memories:
            blocked = FORBIDDEN_FIELDS.intersection(memory.keys())
            if blocked:
                field = sorted(blocked)[0]
                raise ValueError(f"field '{field}' is not allowed at inference")

    @staticmethod
    def _tokens(text):
        return set(re.findall(r"[a-zA-Z0-9_]+", str(text).lower()))

    @staticmethod
    def _jaccard(left, right):
        if not left or not right:
            return 0.0
        return len(left.intersection(right)) / max(1, len(left.union(right)))

    @staticmethod
    def _position(memory):
        for key in ("position", "time"):
            if key in memory:
                try:
                    return float(memory[key])
                except (TypeError, ValueError):
                    return None
        return None

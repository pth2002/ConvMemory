from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class CompressionRouteConfig:
    """Configuration for note-to-memory candidate routing.

    The defaults are intentionally conservative and reflect the stable
    LoCoMo v0.31 setting: use compressed notes to select a smaller raw-memory
    pool, then let ConvMemory rerank that pool.
    """

    note_depth: int = 240
    max_sources_per_note: int = 5
    max_candidates: int = 450
    raw_anchor: int = 80


@dataclass(frozen=True)
class CompressionRouteResult:
    candidate_indices: List[int]
    candidate_ids: List[str]
    note_indices: List[int]
    raw_anchor_count: int


@dataclass(frozen=True)
class CompressedNoteConfig:
    """Configuration for lightweight raw-memory note construction."""

    mode: str = "session"
    block_size: int = 32
    representatives: int = 3
    strategy: str = "central"
    session_key: str = "session_id"


class CompressionRouter:
    """Route raw memories through compressed note blocks.

    Compressed memories are dictionaries with at least:
    - `text`: note text used for embedding
    - `source_ids`: raw memory ids covered by this note

    The router does not call ConvMemory directly. It only returns candidate ids
    so it can be plugged into any retrieval or agent-memory pipeline.
    """

    def __init__(self, config: Optional[CompressionRouteConfig] = None):
        self.config = config or CompressionRouteConfig()

    def route(
        self,
        query_embedding,
        memory_embeddings,
        memory_ids: Sequence[str],
        compressed_embeddings,
        compressed_memories: Iterable[Mapping],
    ) -> CompressionRouteResult:
        memory_ids = [str(memory_id) for memory_id in memory_ids]
        compressed_memories = list(compressed_memories)
        if len(compressed_memories) != len(compressed_embeddings):
            raise ValueError("compressed_memories and compressed_embeddings must have the same length")

        query = _normalize_vector(query_embedding)
        memories = _normalize_matrix(memory_embeddings)
        notes = _normalize_matrix(compressed_embeddings)

        raw_scores = memories @ query
        raw_order = np.argsort(-raw_scores)
        note_scores = notes @ query if len(notes) else np.asarray([], dtype=np.float32)
        note_order = np.argsort(-note_scores)[: max(0, int(self.config.note_depth))]

        id_to_index = {memory_id: idx for idx, memory_id in enumerate(memory_ids)}
        selected: List[int] = []
        seen = set()

        for idx in raw_order[: max(0, int(self.config.raw_anchor))]:
            self._add_candidate(int(idx), selected, seen)
            if len(selected) >= self.config.max_candidates:
                return self._result(selected, memory_ids, note_order)

        for note_idx in note_order:
            source_indices = []
            for source_id in compressed_memories[int(note_idx)].get("source_ids", []):
                source_key = str(source_id)
                if source_key in id_to_index:
                    source_indices.append(id_to_index[source_key])
            source_indices.sort(key=lambda idx: -float(raw_scores[idx]))
            limit = int(self.config.max_sources_per_note)
            if limit > 0:
                source_indices = source_indices[:limit]

            for idx in source_indices:
                self._add_candidate(int(idx), selected, seen)
                if len(selected) >= self.config.max_candidates:
                    return self._result(selected, memory_ids, note_order)

        return self._result(selected, memory_ids, note_order)

    @staticmethod
    def _add_candidate(idx: int, selected: List[int], seen) -> None:
        if idx in seen:
            return
        selected.append(idx)
        seen.add(idx)

    def _result(self, selected: List[int], memory_ids: Sequence[str], note_order) -> CompressionRouteResult:
        return CompressionRouteResult(
            candidate_indices=list(selected),
            candidate_ids=[memory_ids[idx] for idx in selected],
            note_indices=[int(idx) for idx in note_order],
            raw_anchor_count=min(len(selected), max(0, int(self.config.raw_anchor))),
        )


def _normalize_vector(x):
    arr = np.asarray(x, dtype=np.float32)
    return arr / (np.linalg.norm(arr) + 1e-8)


def _normalize_matrix(x):
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8)


def build_compressed_notes(
    memories: Iterable[Mapping],
    memory_embeddings,
    config: Optional[CompressedNoteConfig] = None,
):
    """Build compressed notes from raw memories.

    This helper is intentionally simple: it groups an ordered memory stream by
    session or fixed-size blocks, chooses representative turns, and keeps
    `source_ids` so the note can be expanded back to raw memories.
    """

    cfg = config or CompressedNoteConfig()
    memories = list(memories)
    embeddings = _normalize_matrix(memory_embeddings)
    if len(memories) != len(embeddings):
        raise ValueError("memories and memory_embeddings must have the same length")
    if cfg.mode not in {"session", "block"}:
        raise ValueError("CompressedNoteConfig.mode must be 'session' or 'block'")
    if cfg.strategy not in {"central", "first"}:
        raise ValueError("CompressedNoteConfig.strategy must be 'central' or 'first'")

    groups = _session_groups(memories, cfg.session_key)
    if cfg.mode == "block":
        groups = [
            group[start : start + cfg.block_size]
            for group in groups
            for start in range(0, len(group), cfg.block_size)
        ]

    notes = []
    for note_idx, group in enumerate(groups):
        if not group:
            continue
        reps = _representative_indices(group, embeddings, cfg.strategy, cfg.representatives)
        text = " ".join(str(memories[idx].get("text", "")) for idx in reps)
        source_ids = [str(memories[idx].get("id", idx)) for idx in group]
        session_id = str(memories[group[0]].get(cfg.session_key, ""))
        notes.append(
            {
                "id": f"{cfg.mode}:{note_idx}",
                "text": text,
                "source_ids": source_ids,
                cfg.session_key: session_id,
                "granularity": cfg.mode,
            }
        )
    return notes


def _session_groups(memories, session_key):
    groups = []
    current_value = None
    current = []
    for idx, item in enumerate(memories):
        value = item.get(session_key, "")
        if current and value != current_value:
            groups.append(current)
            current = []
        current_value = value
        current.append(idx)
    if current:
        groups.append(current)
    return groups


def _representative_indices(group, embeddings, strategy, representatives):
    count = max(1, min(int(representatives), len(group)))
    if strategy == "first":
        return group[:count]

    local_embeddings = embeddings[group]
    centroid = local_embeddings.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
    scores = local_embeddings @ centroid
    picked = np.argsort(-scores)[:count]
    return [group[int(i)] for i in sorted(picked)]

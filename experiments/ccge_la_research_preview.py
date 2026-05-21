"""Research-preview CCGE-LA conflict editor.

CCGE-LA stands for Low-Amplitude Counterfactual Conflict Graph Editor.

This file provides a clean, public prototype of the conflict-aware editor that
was studied internally after the core ConvMemory evaluation work. It is not a
packaged stable API and it does not ship a pretrained checkpoint.

Intended use:

    vector search -> ConvMemory rerank -> CCGE-LA candidate-set edit -> context

The editor reads the full ConvMemory candidate set, builds conflict-state
features, and applies a small gated residual correction to ConvMemory scores.

Training signal should be retrieval cross-entropy only. Do not add gold-defined
features such as "is current", "is stale", "is latest", or "is gold".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn


FEATURE_NAMES = [
    "base_score_z",
    "dense_score_z",
    "position_z",
    "query_overlap_z",
    "base_rank_norm",
    "dense_rank_norm",
    "sim_to_base_top",
    "sim_to_dense_top",
    "semantic_density_top16",
    "overlap_to_top",
    "newer_than_base_top",
    "older_than_base_top",
    "abs_pos_gap_top_z",
    "base_margin_1_2",
    "base_entropy_top16",
    "conflict_density_top16",
    "time_span_top16",
    "top_overlap",
]


@dataclass(frozen=True)
class ConflictFeatureBatch:
    """Candidate-set features consumed by CCGE-LA.

    The score features are normalized within each query's candidate set. The
    final editor score is therefore in the same normalized score space.
    """

    candidate_ids: list[str]
    features: np.ndarray


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    std = float(values.std())
    if std < 1.0e-6:
        return values - float(values.mean())
    return (values - float(values.mean())) / std


def rank_norm(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, kind="mergesort")
    ranks = np.zeros(len(scores), dtype=np.float32)
    for rank, idx in enumerate(order):
        ranks[int(idx)] = rank / max(1, len(scores) - 1)
    return ranks


def softmax_entropy(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    x = np.asarray(values, dtype=np.float32)
    x = x - float(x.max())
    p = np.exp(x)
    p = p / max(float(p.sum()), 1.0e-8)
    return float(-(p * np.log(p + 1.0e-8)).sum() / np.log(len(p)))


def normalized_embeddings(embeddings: np.ndarray | None, n: int) -> np.ndarray:
    if embeddings is None:
        return np.eye(n, dtype=np.float32)
    x = np.asarray(embeddings, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] != n:
        raise ValueError("candidate_embeddings must have shape [num_candidates, dim]")
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1.0e-8)


def build_conflict_features(
    *,
    candidate_ids: Sequence[str],
    convmemory_scores: Sequence[float],
    dense_scores: Sequence[float] | None = None,
    positions: Sequence[float] | None = None,
    candidate_embeddings: np.ndarray | None = None,
    query_overlaps: Sequence[float] | None = None,
    top_k_density: int = 16,
) -> ConflictFeatureBatch:
    """Build CCGE-LA candidate-set features.

    Parameters
    ----------
    candidate_ids:
        Stable ids for the candidates in ConvMemory candidate order.
    convmemory_scores:
        Base ConvMemory scores for the same candidates.
    dense_scores:
        Optional dense/vector-search scores. If omitted, ConvMemory scores are
        reused.
    positions:
        Optional memory order or timestamp values. If omitted, candidate order is
        used.
    candidate_embeddings:
        Optional candidate embeddings for semantic density features.
    query_overlaps:
        Optional lexical/query-overlap feature per candidate.

    Returns
    -------
    ConflictFeatureBatch
        Feature matrix with columns listed in ``FEATURE_NAMES``.
    """

    ids = [str(x) for x in candidate_ids]
    n = len(ids)
    if n == 0:
        raise ValueError("candidate_ids must not be empty")

    base = np.asarray(convmemory_scores, dtype=np.float32)
    if base.shape[0] != n:
        raise ValueError("convmemory_scores must match candidate_ids")

    dense = np.asarray(dense_scores if dense_scores is not None else base, dtype=np.float32)
    pos = np.asarray(positions if positions is not None else np.arange(n), dtype=np.float32)
    overlap = np.asarray(query_overlaps if query_overlaps is not None else np.zeros(n), dtype=np.float32)
    if dense.shape[0] != n or pos.shape[0] != n or overlap.shape[0] != n:
        raise ValueError("dense_scores, positions, and query_overlaps must match candidate_ids")

    emb = normalized_embeddings(candidate_embeddings, n)
    base_order = np.argsort(-base, kind="mergesort")
    dense_order = np.argsort(-dense, kind="mergesort")
    top_base = int(base_order[0])
    top_dense = int(dense_order[0])
    topk = base_order[: min(top_k_density, n)]

    sim_to_base_top = emb @ emb[top_base]
    sim_to_dense_top = emb @ emb[top_dense]
    density = (emb @ emb[topk].T).mean(axis=1) if len(topk) else np.zeros(n, dtype=np.float32)
    pos_gap = np.abs(pos - pos[top_base])

    sorted_base_z = np.sort(zscore(base))[::-1]
    margin = float(sorted_base_z[0] - sorted_base_z[1]) if len(sorted_base_z) > 1 else 0.0
    entropy = softmax_entropy(zscore(base)[topk])
    conflict_density = float(np.mean((sim_to_base_top[topk] > 0.45) & (np.abs(pos[topk] - pos[top_base]) > 0))) if len(topk) else 0.0
    span = float(pos[topk].max() - pos[topk].min()) if len(topk) else 0.0
    full_span = max(1.0, float(pos.max() - pos.min()))
    top_overlap = float(overlap[top_base])

    features = np.stack(
        [
            zscore(base),
            zscore(dense),
            zscore(pos),
            zscore(overlap),
            rank_norm(base),
            rank_norm(dense),
            sim_to_base_top.astype(np.float32),
            sim_to_dense_top.astype(np.float32),
            density.astype(np.float32),
            np.full(n, top_overlap, dtype=np.float32),
            (pos > pos[top_base]).astype(np.float32),
            (pos < pos[top_base]).astype(np.float32),
            zscore(pos_gap),
            np.full(n, margin, dtype=np.float32),
            np.full(n, entropy, dtype=np.float32),
            np.full(n, conflict_density, dtype=np.float32),
            np.full(n, span / full_span, dtype=np.float32),
            np.full(n, top_overlap, dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    return ConflictFeatureBatch(candidate_ids=ids, features=features)


class CCGELowAmplitudeEditor(nn.Module):
    """Low-amplitude candidate-set editor over ConvMemory scores.

    The model consumes one query's full candidate feature matrix with shape
    ``[batch, candidates, feature_dim]`` and returns edited scores plus a
    query-level gate. It is intentionally small and residual: ConvMemory remains
    the base scorer.
    """

    def __init__(
        self,
        feature_dim: int = len(FEATURE_NAMES),
        *,
        model_dim: int = 96,
        layers: int = 2,
        gate_bias: float = -2.0,
        residual_init: float = 0.35,
    ):
        super().__init__()
        self.in_proj = nn.Sequential(nn.Linear(feature_dim, model_dim), nn.GELU(), nn.LayerNorm(model_dim))
        enc = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=4,
            dim_feedforward=model_dim * 3,
            dropout=0.08,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.residual = nn.Sequential(nn.Linear(model_dim, model_dim), nn.GELU(), nn.Dropout(0.05), nn.Linear(model_dim, 1))
        self.gate = nn.Sequential(nn.Linear(model_dim + 7, 64), nn.GELU(), nn.Linear(64, 1))
        self.residual_scale = nn.Parameter(torch.tensor(residual_init))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, gate_bias)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        base = features[..., 0]
        h = self.encoder(self.in_proj(features))
        residual = self.residual(h).squeeze(-1)
        pooled = h.mean(dim=1)
        state = torch.stack(
            [
                features[..., 13].mean(dim=1),
                features[..., 14].mean(dim=1),
                features[..., 15].mean(dim=1),
                features[..., 16].mean(dim=1),
                features[..., 17].mean(dim=1),
                (features[..., 4] < 0.05).float().mean(dim=1),
                features[..., 8].max(dim=1).values,
            ],
            dim=-1,
        )
        gate = torch.sigmoid(self.gate(torch.cat([pooled, state], dim=-1))).squeeze(-1)
        scores = base + gate.unsqueeze(-1) * torch.clamp(self.residual_scale, 0.05, 2.0) * residual
        return scores, gate


def multi_positive_retrieval_loss(scores: torch.Tensor, gold_mask: torch.Tensor) -> torch.Tensor:
    """Retrieval cross-entropy for one or more positive candidates."""

    all_lse = torch.logsumexp(scores, dim=-1)
    masked = scores.masked_fill(~gold_mask, -1.0e9)
    gold_lse = torch.logsumexp(masked, dim=-1)
    return -(gold_lse - all_lse).mean()


@torch.no_grad()
def rank_candidates(
    editor: CCGELowAmplitudeEditor,
    batch: ConflictFeatureBatch,
    *,
    device: str | torch.device = "cpu",
) -> list[tuple[str, float]]:
    """Return candidate ids sorted by edited CCGE-LA score."""

    editor.eval()
    x = torch.tensor(batch.features, dtype=torch.float32, device=device).unsqueeze(0)
    scores, _ = editor.to(device)(x)
    values = scores.detach().cpu().numpy()[0]
    order = np.argsort(-values, kind="mergesort")
    return [(batch.candidate_ids[int(i)], float(values[int(i)])) for i in order]


__all__ = [
    "FEATURE_NAMES",
    "ConflictFeatureBatch",
    "CCGELowAmplitudeEditor",
    "build_conflict_features",
    "multi_positive_retrieval_loss",
    "rank_candidates",
]

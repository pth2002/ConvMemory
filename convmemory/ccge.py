"""CCGE-LA conflict-aware candidate-set editor.

CCGE-LA stands for Low-Amplitude Counterfactual Conflict Graph Editor. It is a
lightweight editor that runs after ConvMemory and applies a small residual score
correction when the retrieved candidate set looks conflict-prone.

The module is intentionally checkpoint-agnostic. Applications can attach a
trained editor with ``ConvMemory.attach_ccge_editor`` or load one from disk with
``ConvMemory.load_ccge_editor``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

from .scoring import lexical_signature


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
    "token_overlap_to_top",
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
class CCGEConfig:
    """Configuration for the public CCGE-LA editor."""

    feature_dim: int = len(FEATURE_NAMES)
    model_dim: int = 96
    layers: int = 2
    num_heads: int = 4
    dropout: float = 0.08
    gate_bias: float = -2.0
    residual_init: float = 0.35


@dataclass(frozen=True)
class CCGEFeatureBatch:
    """Feature matrix for one query's candidate set."""

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


def query_overlap_scores(query: str, candidate_texts: Sequence[str]) -> np.ndarray:
    """Lexical overlap scores for query and candidate memories."""

    query_set, _ = lexical_signature(query)
    values = []
    for text in candidate_texts:
        memory_set, _ = lexical_signature(str(text))
        values.append(len(query_set & memory_set) / max(1, len(query_set)))
    return np.asarray(values, dtype=np.float32)


def token_overlap_to_text(candidate_texts: Sequence[str], top_index: int) -> np.ndarray:
    """Token overlap between each candidate and the selected top candidate."""

    top_set, _ = lexical_signature(str(candidate_texts[int(top_index)]))
    values = []
    for text in candidate_texts:
        memory_set, _ = lexical_signature(str(text))
        union = top_set | memory_set
        values.append(len(top_set & memory_set) / max(1, len(union)))
    return np.asarray(values, dtype=np.float32)


def build_ccge_features(
    *,
    candidate_ids: Sequence[str],
    convmemory_scores: Sequence[float],
    dense_scores: Sequence[float] | None = None,
    positions: Sequence[float] | None = None,
    candidate_embeddings: np.ndarray | None = None,
    query_overlaps: Sequence[float] | None = None,
    query: str | None = None,
    candidate_texts: Sequence[str] | None = None,
    top_k_density: int = 16,
) -> CCGEFeatureBatch:
    """Build CCGE-LA candidate-set features.

    The features describe the retrieved candidate set. They do not encode
    gold/current/stale labels and are safe to compute at inference time.
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
    if query_overlaps is not None:
        overlap = np.asarray(query_overlaps, dtype=np.float32)
    elif query is not None and candidate_texts is not None:
        overlap = query_overlap_scores(query, candidate_texts)
    else:
        overlap = np.zeros(n, dtype=np.float32)
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
    if candidate_texts is not None:
        overlap_to_top = token_overlap_to_text(candidate_texts, top_base)
    else:
        overlap_to_top = np.full(n, float(overlap[top_base]), dtype=np.float32)
    pos_gap = np.abs(pos - pos[top_base])

    sorted_base_z = np.sort(zscore(base))[::-1]
    margin = float(sorted_base_z[0] - sorted_base_z[1]) if len(sorted_base_z) > 1 else 0.0
    entropy = softmax_entropy(zscore(base)[topk])
    conflict_density = (
        float(np.mean((sim_to_base_top[topk] > 0.45) & (np.abs(pos[topk] - pos[top_base]) > 0)))
        if len(topk)
        else 0.0
    )
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
            overlap_to_top.astype(np.float32),
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
    return CCGEFeatureBatch(candidate_ids=ids, features=features)


class CCGELowAmplitudeEditor(nn.Module):
    """Low-amplitude residual editor over ConvMemory candidate scores."""

    def __init__(
        self,
        feature_dim: int = len(FEATURE_NAMES),
        *,
        model_dim: int = 96,
        layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.08,
        gate_bias: float = -2.0,
        residual_init: float = 0.35,
    ):
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        self.config = CCGEConfig(
            feature_dim=int(feature_dim),
            model_dim=int(model_dim),
            layers=int(layers),
            num_heads=int(num_heads),
            dropout=float(dropout),
            gate_bias=float(gate_bias),
            residual_init=float(residual_init),
        )
        self.in_proj = nn.Sequential(
            nn.Linear(feature_dim, model_dim),
            nn.GELU(),
            nn.LayerNorm(model_dim),
        )
        enc = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.residual = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(model_dim, 1),
        )
        self.gate = nn.Sequential(nn.Linear(model_dim + 7, 64), nn.GELU(), nn.Linear(64, 1))
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_init)))
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
        scale = torch.clamp(self.residual_scale, 0.05, 2.0)
        scores = base + gate.unsqueeze(-1) * scale * residual
        return scores, gate

    @torch.no_grad()
    def edit_batch(
        self,
        batch: CCGEFeatureBatch,
        *,
        device: str | torch.device | None = None,
    ) -> tuple[np.ndarray, float]:
        """Return edited scores and the query-level gate for one feature batch."""

        if device is None:
            device = next(self.parameters()).device
        self.eval()
        x = torch.tensor(batch.features, dtype=torch.float32, device=device).unsqueeze(0)
        scores, gate = self.to(device)(x)
        return scores.detach().cpu().numpy()[0], float(gate.detach().cpu().numpy()[0])

    def save_pretrained(self, path: str | Path) -> None:
        path = Path(path)
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
            target = path
        else:
            path.mkdir(parents=True, exist_ok=True)
            target = path / "ccge_la.pt"
        torch.save(
            {
                "format": "convmemory-ccge-la",
                "version": 1,
                "config": asdict(self.config),
                "state_dict": self.state_dict(),
            },
            target,
        )

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
        strict: bool = True,
    ) -> "CCGELowAmplitudeEditor":
        path = Path(path)
        source = path / "ccge_la.pt" if path.is_dir() else path
        payload = torch.load(source, map_location="cpu")
        config = payload.get("config", {})
        model = cls(**config)
        state_dict = payload.get("state_dict", payload)
        model.load_state_dict(state_dict, strict=strict)
        return model.to(device).eval()


def multi_positive_retrieval_loss(scores: torch.Tensor, gold_mask: torch.Tensor) -> torch.Tensor:
    """Retrieval cross-entropy for one or more positive candidates."""

    all_lse = torch.logsumexp(scores, dim=-1)
    masked = scores.masked_fill(~gold_mask, -1.0e9)
    gold_lse = torch.logsumexp(masked, dim=-1)
    return -(gold_lse - all_lse).mean()


@torch.no_grad()
def rank_candidates(
    editor: CCGELowAmplitudeEditor,
    batch: CCGEFeatureBatch,
    *,
    device: str | torch.device = "cpu",
) -> list[tuple[str, float]]:
    """Return candidate ids sorted by edited CCGE-LA score."""

    values, _ = editor.edit_batch(batch, device=device)
    order = np.argsort(-values, kind="mergesort")
    return [(batch.candidate_ids[int(i)], float(values[int(i)])) for i in order]


__all__ = [
    "FEATURE_NAMES",
    "CCGEConfig",
    "CCGEFeatureBatch",
    "CCGELowAmplitudeEditor",
    "build_ccge_features",
    "multi_positive_retrieval_loss",
    "query_overlap_scores",
    "rank_candidates",
    "token_overlap_to_text",
]

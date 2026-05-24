"""Experimental ConvMemory v2 Memory-MLA prefix-protected expander.

Memory-MLA runs after the base ConvMemory ranking. It keeps the strongest
prefix unchanged and only reorders a later candidate window with lightweight
latent-slot interaction scores. The module is opt-in and does not use gold,
current/stale labels, or teacher scores at inference time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn

from .hub import resolve_checkpoint_path
from .reranker import RerankResult
from .scoring import lexical_signature


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _zscore(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr
    std = float(arr.std())
    if std < 1.0e-8:
        return arr * 0.0
    return (arr - float(arr.mean())) / std


def _stable_projection(input_dim: int, output_dim: int, seed: int) -> np.ndarray | None:
    if input_dim == output_dim:
        return None
    rng = np.random.default_rng(int(seed))
    return rng.normal(
        0.0,
        1.0 / math.sqrt(max(1, output_dim)),
        size=(int(input_dim), int(output_dim)),
    ).astype(np.float32)


def _project(values: np.ndarray, projection: np.ndarray | None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if projection is None:
        return values.astype(np.float32)
    return (values @ projection).astype(np.float32)


def _chunk_text(text: str, latent_count: int, max_tokens_per_chunk: int) -> list[str]:
    tokens = TOKEN_RE.findall(str(text))
    if not tokens:
        return [str(text)[:256] or "[empty]"] * int(latent_count)
    target = max(1, min(int(max_tokens_per_chunk), math.ceil(len(tokens) / max(1, int(latent_count)))))
    chunks: list[str] = []
    for start in range(0, len(tokens), target):
        chunks.append(" ".join(tokens[start : start + target]))
        if len(chunks) >= int(latent_count):
            break
    while len(chunks) < int(latent_count):
        chunks.append(chunks[-1])
    return chunks[: int(latent_count)]


def _encoder_encode(encoder, texts: Sequence[str]) -> np.ndarray:
    if hasattr(encoder, "encode"):
        return np.asarray(
            encoder.encode(
                list(texts),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )
    if hasattr(encoder, "transform"):
        return np.asarray(encoder.transform(list(texts)), dtype=np.float32)
    raise ValueError("encoder must provide an encode(...) or transform(...) method")


def _query_overlap(query: str, text: str) -> float:
    query_set, _ = lexical_signature(query or "")
    if not query_set:
        return 0.0
    memory_set, _ = lexical_signature(str(text))
    return len(query_set & memory_set) / max(1, len(query_set))


@dataclass(frozen=True)
class MemoryMLAConfig:
    """Configuration for the experimental Memory-MLA expander."""

    embedding_dim: int = 64
    scalar_dim: int = 6
    latent_count: int = 12
    code_dim: int = 64
    model_dim: int = 224
    hidden_dim: int = 384
    setwise_layers: int = 0
    heads: int = 4
    base_feature_idx: int = 5
    residual_max: float = 0.26
    gate_bias_init: float = -1.1
    query_slots: int = 8
    pair_top_k: int = 64
    cluster_top_k: int = 12
    projection_seed: int = 152
    max_tokens_per_chunk: int = 16
    candidate_top_n: int = 384
    protect_top_k: int = 7
    expand_window: int = 16


class MemoryMLAExpander(nn.Module):
    """Prefix-protected latent-slot expander for ConvMemory rankings."""

    def __init__(
        self,
        embedding_dim: int = 64,
        *,
        scalar_dim: int = 6,
        latent_count: int = 12,
        code_dim: int = 64,
        model_dim: int = 224,
        hidden_dim: int = 384,
        setwise_layers: int = 0,
        heads: int = 4,
        base_feature_idx: int = 5,
        residual_max: float = 0.26,
        gate_bias_init: float = -1.1,
        query_slots: int = 8,
        pair_top_k: int = 64,
        cluster_top_k: int = 12,
        projection_seed: int = 152,
        max_tokens_per_chunk: int = 16,
        candidate_top_n: int = 384,
        protect_top_k: int = 7,
        expand_window: int = 16,
    ) -> None:
        super().__init__()
        if isinstance(embedding_dim, MemoryMLAConfig):
            cfg = embedding_dim
            embedding_dim = cfg.embedding_dim
            scalar_dim = cfg.scalar_dim
            latent_count = cfg.latent_count
            code_dim = cfg.code_dim
            model_dim = cfg.model_dim
            hidden_dim = cfg.hidden_dim
            setwise_layers = cfg.setwise_layers
            heads = cfg.heads
            base_feature_idx = cfg.base_feature_idx
            residual_max = cfg.residual_max
            gate_bias_init = cfg.gate_bias_init
            query_slots = cfg.query_slots
            pair_top_k = cfg.pair_top_k
            cluster_top_k = cfg.cluster_top_k
            projection_seed = cfg.projection_seed
            max_tokens_per_chunk = cfg.max_tokens_per_chunk
            candidate_top_n = cfg.candidate_top_n
            protect_top_k = cfg.protect_top_k
            expand_window = cfg.expand_window
        if model_dim % heads != 0:
            raise ValueError("model_dim must be divisible by heads")
        self.config = MemoryMLAConfig(
            embedding_dim=int(embedding_dim),
            scalar_dim=int(scalar_dim),
            latent_count=int(latent_count),
            code_dim=int(code_dim),
            model_dim=int(model_dim),
            hidden_dim=int(hidden_dim),
            setwise_layers=int(setwise_layers),
            heads=int(heads),
            base_feature_idx=int(base_feature_idx),
            residual_max=float(residual_max),
            gate_bias_init=float(gate_bias_init),
            query_slots=int(query_slots),
            pair_top_k=int(pair_top_k),
            cluster_top_k=int(cluster_top_k),
            projection_seed=int(projection_seed),
            max_tokens_per_chunk=int(max_tokens_per_chunk),
            candidate_top_n=int(candidate_top_n),
            protect_top_k=int(protect_top_k),
            expand_window=int(expand_window),
        )
        self.trained_embedding_model_name = None
        self.base_feature_idx = self.config.base_feature_idx
        self.residual_max = self.config.residual_max
        self.query_slots = self.config.query_slots
        self.pair_top_k = self.config.pair_top_k
        self.cluster_top_k = self.config.cluster_top_k

        self.query_proj = nn.Sequential(nn.Linear(code_dim, model_dim), nn.GELU(), nn.LayerNorm(model_dim))
        self.query_slot_proj = nn.Sequential(nn.Linear(code_dim, model_dim * self.query_slots), nn.GELU())
        self.query_slot_norm = nn.LayerNorm(model_dim)
        self.query_slot_bias = nn.Parameter(torch.zeros(self.query_slots, model_dim))
        self.code_proj = nn.Sequential(nn.Linear(code_dim, model_dim), nn.GELU(), nn.LayerNorm(model_dim))
        self.scalar_proj = nn.Sequential(nn.Linear(scalar_dim, model_dim), nn.GELU(), nn.LayerNorm(model_dim))

        self.score_in = nn.Sequential(
            nn.Linear(model_dim * 9 + scalar_dim + self.query_slots * 3 + 6, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        if setwise_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=heads,
                dim_feedforward=hidden_dim * 3,
                dropout=0.08,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.setwise = nn.TransformerEncoder(layer, num_layers=setwise_layers)
        else:
            self.setwise = None

        self.unary = nn.Sequential(nn.Linear(hidden_dim + 4, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
        self.pair = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 10, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.residual = nn.Sequential(
            nn.Linear(hidden_dim + 7, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim + 7, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.08))
        nn.init.constant_(self.gate[-1].bias, float(gate_bias_init))

    def _slot_hidden(self, query: torch.Tensor, codes: torch.Tensor, scalars: torch.Tensor, *, use_setwise: bool) -> torch.Tensor:
        q = self.query_proj(query)
        q_slots = self.query_slot_proj(query).view(query.shape[0], self.query_slots, -1)
        q_slots = self.query_slot_norm(q_slots + self.query_slot_bias[None, :, :])
        c = self.code_proj(codes)

        attn = torch.einsum("bqf,bnmf->bnqm", q_slots, c) / math.sqrt(c.shape[-1])
        weights = torch.softmax(attn, dim=-1)
        soft_ctx = torch.einsum("bnqm,bnmf->bnqf", weights, c)
        sharp_weights = torch.softmax(attn * 6.0, dim=-1)
        sharp_ctx = torch.einsum("bnqm,bnmf->bnqf", sharp_weights, c)

        ctx = soft_ctx.mean(dim=2)
        sharp = sharp_ctx.mean(dim=2)
        pooled = c.mean(dim=2)
        top_vals = torch.topk(attn, k=min(2, attn.shape[-1]), dim=-1).values
        maxsim = top_vals[..., 0]
        max_margin = top_vals[..., 0] - top_vals[..., 1] if top_vals.shape[-1] > 1 else torch.zeros_like(maxsim)
        attn_std = attn.std(dim=-1, unbiased=False)
        entropy = -(weights * torch.log(weights.clamp_min(1.0e-8))).sum(dim=-1) / math.log(max(2, attn.shape[-1]))
        slot_stats = torch.cat(
            [
                maxsim,
                max_margin,
                attn_std,
                maxsim.max(dim=-1, keepdim=True).values,
                maxsim.mean(dim=-1, keepdim=True),
                max_margin.max(dim=-1, keepdim=True).values,
                max_margin.mean(dim=-1, keepdim=True),
                attn_std.mean(dim=-1, keepdim=True),
                entropy.mean(dim=-1, keepdim=True),
            ],
            dim=-1,
        )
        feature_ctx = self.scalar_proj(scalars)
        q_expanded = q[:, None, :].expand(-1, c.shape[1], -1)
        h = self.score_in(
            torch.cat(
                [
                    q_expanded,
                    ctx,
                    sharp,
                    pooled,
                    q_expanded * ctx,
                    torch.abs(q_expanded - ctx),
                    q_expanded * sharp,
                    torch.abs(q_expanded - sharp),
                    feature_ctx,
                    slot_stats,
                    scalars,
                ],
                dim=-1,
            )
        )
        if self.setwise is not None and use_setwise:
            h = self.setwise(h)
        return h

    def _cluster_features(self, h: torch.Tensor, base_score: torch.Tensor) -> torch.Tensor:
        n = h.shape[1]
        k = min(self.cluster_top_k, max(1, n - 1))
        norm = torch.nn.functional.normalize(h, dim=-1)
        sim = torch.matmul(norm, norm.transpose(1, 2))
        eye = torch.eye(n, dtype=torch.bool, device=h.device)[None, :, :]
        sim = sim.masked_fill(eye, -1.0e4)
        top_vals, top_idx = torch.topk(sim, k=k, dim=-1)
        support_mean = top_vals.mean(dim=-1)
        support_max = top_vals.max(dim=-1).values
        isolation = 1.0 - support_max
        gathered_scores = torch.gather(base_score[:, None, :].expand(-1, n, -1), dim=-1, index=top_idx)
        score_support = (top_vals.clamp_min(0.0) * gathered_scores).mean(dim=-1)
        return torch.stack([support_mean, support_max, isolation, score_support], dim=-1)

    def forward(
        self,
        query: torch.Tensor,
        codes: torch.Tensor,
        scalars: torch.Tensor,
        *,
        return_parts: bool = False,
        use_pair: bool = True,
        use_cluster: bool = True,
        use_setwise: bool = True,
    ):
        h = self._slot_hidden(query, codes, scalars, use_setwise=use_setwise)
        base_score = scalars[..., self.base_feature_idx]
        rank_feature = scalars[..., self.base_feature_idx + 1] if self.base_feature_idx + 1 < scalars.shape[-1] else torch.zeros_like(base_score)

        cluster = self._cluster_features(h, base_score) if use_cluster else torch.zeros(*h.shape[:2], 4, device=h.device, dtype=h.dtype)
        unary = self.unary(torch.cat([h, cluster], dim=-1)).squeeze(-1)

        n = h.shape[1]
        k = min(self.pair_top_k, n)
        top_idx = torch.topk(base_score, k=k, dim=-1).indices
        refs = torch.gather(h, dim=1, index=top_idx[:, :, None].expand(-1, -1, h.shape[-1]))
        ref_base = torch.gather(base_score, dim=1, index=top_idx)
        ref_rank = torch.gather(rank_feature, dim=1, index=top_idx)
        ref_cluster = torch.gather(cluster, dim=1, index=top_idx[:, :, None].expand(-1, -1, 4))

        h_i = h[:, :, None, :].expand(-1, n, k, -1)
        h_j = refs[:, None, :, :].expand(-1, n, -1, -1)
        base_i = base_score[:, :, None].expand(-1, n, k)
        base_j = ref_base[:, None, :].expand(-1, n, -1)
        rank_i = rank_feature[:, :, None].expand(-1, n, k)
        rank_j = ref_rank[:, None, :].expand(-1, n, -1)
        cluster_i = cluster[:, :, None, :].expand(-1, n, k, -1)
        cluster_j = ref_cluster[:, None, :, :].expand(-1, n, -1, -1)

        pair_scalar = torch.cat(
            [
                (base_i - base_j)[..., None],
                torch.abs(base_i - base_j)[..., None],
                (rank_i - rank_j)[..., None],
                torch.abs(rank_i - rank_j)[..., None],
                cluster_i - cluster_j,
                torch.abs(cluster_i[..., :2] - cluster_j[..., :2]),
            ],
            dim=-1,
        )
        pair_feat = torch.cat([h_i, h_j, h_i - h_j, h_i * h_j, pair_scalar], dim=-1)
        pair_logits = self.pair(pair_feat).squeeze(-1)
        self_mask = torch.arange(n, device=h.device)[None, :, None] == top_idx[:, None, :]
        pair_logits = pair_logits.masked_fill(self_mask, -1.0e4)
        if not use_pair:
            pair_gain = torch.zeros_like(base_score)
        else:
            pair_max = pair_logits.max(dim=-1).values
            valid_pair = pair_logits.masked_fill(pair_logits < -1.0e3, 0.0)
            pair_mean = valid_pair.mean(dim=-1)
            pair_gain = pair_max + 0.25 * pair_mean

        controller = torch.cat([h, cluster, unary[:, :, None], pair_gain[:, :, None], base_score[:, :, None]], dim=-1)
        residual = self.residual(controller).squeeze(-1) + 0.50 * pair_gain + 0.25 * unary
        gate = torch.sigmoid(self.gate(controller).squeeze(-1))
        scale = torch.clamp(self.residual_scale, 0.0, self.residual_max)
        delta = gate * scale * residual
        scores = base_score + delta
        if return_parts:
            return scores, base_score, gate, delta, pair_gain, cluster
        return scores

    def _project_query(self, query_embedding: np.ndarray) -> np.ndarray:
        query = np.asarray(query_embedding, dtype=np.float32)
        projection = _stable_projection(query.shape[-1], self.config.code_dim, self.config.projection_seed)
        return _project(query, projection)

    def build_codes_from_embeddings(self, candidate_embeddings: np.ndarray) -> np.ndarray:
        """Build deterministic fallback latent codes from candidate embeddings."""

        values = np.asarray(candidate_embeddings, dtype=np.float32)
        if values.ndim == 3:
            if values.shape[1] != self.config.latent_count or values.shape[2] != self.config.code_dim:
                raise ValueError("candidate_codes must have shape [n, latent_count, code_dim]")
            return values.astype(np.float32)
        if values.ndim != 2:
            raise ValueError("candidate_embeddings must have shape [n, dim]")
        projection = _stable_projection(
            values.shape[-1],
            self.config.latent_count * self.config.code_dim,
            self.config.projection_seed + 17,
        )
        if projection is None:
            projected = np.repeat(values[:, None, :], self.config.latent_count, axis=1)
        else:
            projected = _project(values, projection).reshape(len(values), self.config.latent_count, self.config.code_dim)
        return projected.astype(np.float32)

    def build_codes_from_texts(self, encoder, candidate_texts: Sequence[str]) -> np.ndarray:
        """Encode memory text chunks into Memory-MLA candidate-side codes."""

        chunks: list[str] = []
        for text in candidate_texts:
            chunks.extend(_chunk_text(text, self.config.latent_count, self.config.max_tokens_per_chunk))
        encoded = _encoder_encode(encoder, chunks)
        projection = _stable_projection(encoded.shape[-1], self.config.code_dim, self.config.projection_seed)
        codes = _project(encoded, projection)
        return codes.reshape(len(candidate_texts), self.config.latent_count, self.config.code_dim).astype(np.float32)

    def build_features(
        self,
        *,
        query: str,
        candidate_results: Sequence[RerankResult],
        candidate_indices: Sequence[int],
        candidate_texts: Sequence[str],
    ) -> np.ndarray:
        """Build deployable v320 scalar features; no labels or teacher scores."""

        n = len(candidate_results)
        raw = np.asarray([r.raw_score for r in candidate_results], dtype=np.float32)
        conv = np.asarray([r.score for r in candidate_results], dtype=np.float32)
        overlap = np.asarray([_query_overlap(query, text) for text in candidate_texts], dtype=np.float32)
        positions = np.asarray(candidate_indices, dtype=np.float32)
        raw_order = np.argsort(-raw, kind="mergesort")
        raw_rank = np.zeros(n, dtype=np.float32)
        for rank, idx in enumerate(raw_order):
            raw_rank[int(idx)] = rank / max(1, n - 1)
        return np.stack(
            [
                _zscore(raw),
                _zscore(overlap),
                _zscore(positions),
                raw_rank,
                overlap,
                _zscore(conv),
            ],
            axis=1,
        ).astype(np.float32)

    @torch.no_grad()
    def score_batch(
        self,
        *,
        query_embedding: np.ndarray,
        candidate_codes: np.ndarray,
        features: np.ndarray,
        device: str | torch.device | None = None,
    ) -> np.ndarray:
        """Return Memory-MLA scores aligned with a candidate batch."""

        if device is None:
            device = next(self.parameters()).device
        self.eval()
        q = torch.tensor(self._project_query(query_embedding), dtype=torch.float32, device=device).unsqueeze(0)
        c = torch.tensor(np.asarray(candidate_codes, dtype=np.float32), dtype=torch.float32, device=device).unsqueeze(0)
        s = torch.tensor(np.asarray(features, dtype=np.float32), dtype=torch.float32, device=device).unsqueeze(0)
        scores = self.to(device)(q, c, s, use_setwise=False)
        return scores.squeeze(0).detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def expand_results(
        self,
        *,
        results: Sequence[RerankResult],
        query_embedding: np.ndarray,
        memory_embeddings: np.ndarray,
        memory_ids: Sequence[str],
        memory_texts: Sequence[str] | None = None,
        query: str = "",
        candidate_indices=None,
        protect_top_k: int | None = None,
        expand_window: int | None = None,
        encoder=None,
        device: str | torch.device | None = None,
    ) -> list[RerankResult]:
        """Apply prefix-protected expansion to a base ConvMemory ranking."""

        if not results:
            return []
        protect = self.config.protect_top_k if protect_top_k is None else int(protect_top_k)
        window = self.config.expand_window if expand_window is None else int(expand_window)
        protect = max(0, protect)
        window = max(protect, window)
        if protect >= len(results) or window <= protect:
            return self._with_ranks(list(results))

        memory_ids = [str(x) for x in memory_ids]
        id_to_idx = {memory_id: idx for idx, memory_id in enumerate(memory_ids)}
        matrix = np.asarray(memory_embeddings, dtype=np.float32)
        if matrix.shape[0] != len(memory_ids):
            raise ValueError("memory_embeddings must match memory_ids")

        if candidate_indices is None:
            allowed = None
        else:
            allowed = {
                memory_ids[int(idx)]
                for idx in np.asarray(candidate_indices, dtype=np.int64)
                if 0 <= int(idx) < len(memory_ids)
            }
        pool_limit = max(window, self.config.candidate_top_n, self.config.pair_top_k, self.config.cluster_top_k)
        pool = [r for r in results if allowed is None or r.memory_id in allowed][: min(pool_limit, len(results))]
        needed_ids = {r.memory_id for r in results[:window]}
        present_ids = {r.memory_id for r in pool}
        for result in results:
            if result.memory_id in needed_ids and result.memory_id not in present_ids:
                pool.append(result)
                present_ids.add(result.memory_id)
        if not pool:
            return self._with_ranks(list(results))

        pool_ids = [r.memory_id for r in pool]
        pool_indices = [id_to_idx[memory_id] for memory_id in pool_ids]
        if memory_texts is None:
            text_by_id = {r.memory_id: r.text or "" for r in results}
        else:
            text_by_id = {memory_id: str(text) for memory_id, text in zip(memory_ids, memory_texts)}
            for r in results:
                text_by_id.setdefault(r.memory_id, r.text or "")
        pool_texts = [text_by_id.get(memory_id, "") for memory_id in pool_ids]
        if encoder is not None and any(pool_texts):
            codes = self.build_codes_from_texts(encoder, pool_texts)
        else:
            codes = self.build_codes_from_embeddings(matrix[pool_indices])
        features = self.build_features(
            query=query,
            candidate_results=pool,
            candidate_indices=pool_indices,
            candidate_texts=pool_texts,
        )
        scores = self.score_batch(
            query_embedding=query_embedding,
            candidate_codes=codes,
            features=features,
            device=device,
        )
        score_by_id = {memory_id: float(score) for memory_id, score in zip(pool_ids, scores)}
        original_by_id = {r.memory_id: r for r in results}
        base_ids = [r.memory_id for r in results]
        prefix = base_ids[:protect]
        window_ids = base_ids[protect:window]
        refined = sorted(
            window_ids,
            key=lambda memory_id: (-score_by_id.get(memory_id, float("-inf")), window_ids.index(memory_id)),
        )
        used = set(prefix + refined)
        tail = [memory_id for memory_id in base_ids if memory_id not in used]
        ordered = [*prefix, *refined, *tail]
        out: list[RerankResult] = []
        refined_set = set(refined)
        for rank, memory_id in enumerate(ordered, start=1):
            original = original_by_id[memory_id]
            out.append(
                RerankResult(
                    memory_id=memory_id,
                    score=score_by_id[memory_id] if memory_id in refined_set and memory_id in score_by_id else original.score,
                    raw_score=original.raw_score,
                    rank=rank,
                    text=original.text,
                )
            )
        return out

    @staticmethod
    def _with_ranks(results: Sequence[RerankResult]) -> list[RerankResult]:
        return [
            RerankResult(
                memory_id=r.memory_id,
                score=r.score,
                raw_score=r.raw_score,
                rank=rank,
                text=r.text,
            )
            for rank, r in enumerate(results, start=1)
        ]

    def save_pretrained(self, path: str | Path) -> None:
        """Save a Memory-MLA expander checkpoint."""

        path = Path(path)
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
            target = path
        else:
            path.mkdir(parents=True, exist_ok=True)
            target = path / "memory_mla.pt"
        torch.save(
            {
                "format": "convmemory-memory-mla",
                "version": 1,
                "config": asdict(self.config),
                "state_dict": self.state_dict(),
                "trained_embedding_model_name": getattr(self, "trained_embedding_model_name", None),
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
    ) -> "MemoryMLAExpander":
        """Load a Memory-MLA expander checkpoint from disk or Hugging Face Hub."""

        path = resolve_checkpoint_path(path)
        source = path / "memory_mla.pt" if path.is_dir() else path
        payload = torch.load(source, map_location="cpu")
        config = payload.get("config", {})
        model = cls(**config)
        state_dict = payload.get("state_dict", payload)
        model.load_state_dict(state_dict, strict=strict)
        model.trained_embedding_model_name = payload.get("trained_embedding_model_name")
        return model.to(device).eval()


__all__ = ["MemoryMLAConfig", "MemoryMLAExpander"]

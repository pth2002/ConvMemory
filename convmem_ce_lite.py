import argparse
from functools import lru_cache
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import CrossEncoder

from convmem_chain_benchmark import (
    load_chain_examples,
    prepare_encoded_example,
    recall_at_k,
    hit_at_k,
    mrr,
    summarize,
    write_csv,
)
from convmem_locomo_benchmark import load_locomo_examples
from convmem_longmemeval import (
    DCAConvMemoryEncoder,
    MixerConvMemoryEncoder,
    SentenceTransformerTextEncoder,
    convmem_rerank,
    resolve_local_model_path,
    train_conv,
)
from locomo_crossencoder_baseline import choose_split


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "him",
    "his",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "she",
    "that",
    "the",
    "their",
    "they",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def cosine_scores(query, matrix):
    return matrix @ query


def normalize_scores(scores):
    scores = np.asarray(scores, dtype=np.float32)
    std = float(scores.std())
    if std < 1e-8:
        return scores - float(scores.mean())
    return (scores - float(scores.mean())) / std


class CELiteScorer(torch.nn.Module):
    def __init__(self, dim, hidden_dim=256, extra_scalar_features=0, extra_dense_features=0):
        super().__init__()
        self.input_dim = dim * 4 + 4 + extra_scalar_features + extra_dense_features
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features):
        return self.net(features).squeeze(-1)


class PairwiseCELiteScorer(torch.nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim=256,
        extra_scalar_features=0,
        extra_dense_features=0,
        interaction_dim=96,
    ):
        super().__init__()
        self.dim = dim
        self.input_dim = dim * 4 + 4 + extra_scalar_features + extra_dense_features
        self.q_proj = torch.nn.Sequential(
            torch.nn.Linear(dim, interaction_dim),
            torch.nn.GELU(),
            torch.nn.LayerNorm(interaction_dim),
        )
        self.m_proj = torch.nn.Sequential(
            torch.nn.Linear(dim, interaction_dim),
            torch.nn.GELU(),
            torch.nn.LayerNorm(interaction_dim),
        )
        self.base = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden_dim),
        )
        self.interaction = torch.nn.Sequential(
            torch.nn.Linear(interaction_dim * 4 + 4, hidden_dim // 2),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden_dim // 2),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim // 2),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features):
        q = features[:, : self.dim]
        m = features[:, self.dim : self.dim * 2]
        scalars = features[:, self.dim * 4 : self.dim * 4 + 4]
        qz = self.q_proj(q)
        mz = self.m_proj(m)
        interaction_features = torch.cat(
            [
                qz,
                mz,
                qz * mz,
                torch.abs(qz - mz),
                scalars,
            ],
            dim=-1,
        )
        h_base = self.base(features)
        h_interaction = self.interaction(interaction_features)
        return self.out(torch.cat([h_base, h_interaction], dim=-1)).squeeze(-1)


class GatedCELiteScorer(torch.nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim=256,
        extra_scalar_features=0,
        extra_dense_features=0,
        router_signal_count=0,
    ):
        super().__init__()
        self.scalar_start = dim * 4
        self.router_start = dim * 4 + 4
        self.router_signal_count = router_signal_count
        input_dim = dim * 4 + 4 + extra_scalar_features + extra_dense_features
        self.residual = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim // 2, 1),
        )
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim // 2),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden_dim // 2),
            torch.nn.Linear(hidden_dim // 2, 4),
        )

    def forward(self, features):
        raw_score = features[:, self.scalar_start]
        conv_score = features[:, self.scalar_start + 1]
        if self.router_signal_count > 0:
            router_score = features[
                :,
                self.router_start : self.router_start + self.router_signal_count,
            ].mean(dim=1)
        else:
            router_score = torch.zeros_like(raw_score)
        residual_score = self.residual(features).squeeze(-1)
        gates = torch.softmax(self.gate(features), dim=-1)
        signals = torch.stack([raw_score, conv_score, router_score, residual_score], dim=-1)
        return (gates * signals).sum(dim=-1)


class FiLMCELiteScorer(torch.nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim=256,
        extra_scalar_features=0,
        extra_dense_features=0,
        compressed_summary_dim=64,
        film_scale=0.2,
    ):
        super().__init__()
        self.dim = dim
        self.base_dim = dim * 4 + 4 + extra_scalar_features
        self.compressed_dense_dim = extra_dense_features
        self.film_scale = film_scale
        self.input = torch.nn.Linear(self.base_dim, hidden_dim)
        self.input_norm = torch.nn.LayerNorm(hidden_dim)
        if extra_dense_features > 0:
            self.block_compress = torch.nn.Linear(dim * 3, compressed_summary_dim)
            self.history_compress = torch.nn.Linear(dim, compressed_summary_dim)
            self.query_compress = torch.nn.Linear(dim, compressed_summary_dim)
            self.film = torch.nn.Sequential(
                torch.nn.Linear(compressed_summary_dim * 3, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, hidden_dim * 2),
            )
        else:
            self.block_compress = None
            self.history_compress = None
            self.query_compress = None
            self.film = None
        self.output = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features):
        base = features[:, : self.base_dim]
        h = torch.nn.functional.gelu(self.input(base))
        if self.film is not None and features.shape[1] >= self.base_dim + self.dim * 4:
            compressed = features[:, self.base_dim : self.base_dim + self.dim * 4]
            block_stats = compressed[:, : self.dim * 3]
            history = compressed[:, self.dim * 3 : self.dim * 4]
            query = base[:, : self.dim]
            block_z = torch.nn.functional.gelu(self.block_compress(block_stats))
            history_z = torch.nn.functional.gelu(self.history_compress(history))
            query_z = torch.nn.functional.gelu(self.query_compress(query))
            gamma, beta = self.film(torch.cat([block_z, history_z, query_z], dim=-1)).chunk(2, dim=-1)
            h = h * (1.0 + self.film_scale * torch.tanh(gamma)) + self.film_scale * beta
        h = self.input_norm(h)
        return self.output(h).squeeze(-1)


class CompressedRouterCELiteScorer(torch.nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim=256,
        extra_scalar_features=0,
        extra_dense_features=0,
        compressed_summary_dim=64,
    ):
        super().__init__()
        self.dim = dim
        self.scalar_start = dim * 4
        self.base_dim = dim * 4 + 4 + extra_scalar_features
        self.has_compressed = extra_dense_features >= dim * 4
        self.residual = torch.nn.Sequential(
            torch.nn.Linear(self.base_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim // 2, 1),
        )
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(self.base_dim, hidden_dim // 2),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden_dim // 2),
            torch.nn.Linear(hidden_dim // 2, 4),
        )
        if self.has_compressed:
            self.block_compress = torch.nn.Linear(dim * 3, compressed_summary_dim)
            self.history_compress = torch.nn.Linear(dim, compressed_summary_dim)
            self.query_compress = torch.nn.Linear(dim, compressed_summary_dim)
            self.memory_compress = torch.nn.Linear(dim, compressed_summary_dim)
            self.compressed_router = torch.nn.Sequential(
                torch.nn.Linear(compressed_summary_dim * 4, hidden_dim // 2),
                torch.nn.GELU(),
                torch.nn.LayerNorm(hidden_dim // 2),
                torch.nn.Linear(hidden_dim // 2, 1),
            )
        else:
            self.block_compress = None
            self.history_compress = None
            self.query_compress = None
            self.memory_compress = None
            self.compressed_router = None

    def forward(self, features):
        base = features[:, : self.base_dim]
        raw_score = features[:, self.scalar_start]
        conv_score = features[:, self.scalar_start + 1]
        residual_score = self.residual(base).squeeze(-1)
        if self.compressed_router is not None and features.shape[1] >= self.base_dim + self.dim * 4:
            compressed = features[:, self.base_dim : self.base_dim + self.dim * 4]
            block_stats = compressed[:, : self.dim * 3]
            history = compressed[:, self.dim * 3 : self.dim * 4]
            query = base[:, : self.dim]
            memory = base[:, self.dim : self.dim * 2]
            z = torch.cat(
                [
                    torch.nn.functional.gelu(self.block_compress(block_stats)),
                    torch.nn.functional.gelu(self.history_compress(history)),
                    torch.nn.functional.gelu(self.query_compress(query)),
                    torch.nn.functional.gelu(self.memory_compress(memory)),
                ],
                dim=-1,
            )
            compressed_score = self.compressed_router(z).squeeze(-1)
        else:
            compressed_score = torch.zeros_like(raw_score)
        gates = torch.softmax(self.gate(base), dim=-1)
        signals = torch.stack([raw_score, conv_score, compressed_score, residual_score], dim=-1)
        return (gates * signals).sum(dim=-1)


class CompressedResidualCELiteScorer(torch.nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim=256,
        extra_scalar_features=0,
        extra_dense_features=0,
        compressed_summary_dim=64,
    ):
        super().__init__()
        self.dim = dim
        self.base_dim = dim * 4 + 4 + extra_scalar_features
        self.has_compressed = extra_dense_features >= dim * 4
        self.base_scorer = torch.nn.Sequential(
            torch.nn.Linear(self.base_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim // 2, 1),
        )
        if self.has_compressed:
            self.block_compress = torch.nn.Linear(dim * 3, compressed_summary_dim)
            self.history_compress = torch.nn.Linear(dim, compressed_summary_dim)
            self.query_compress = torch.nn.Linear(dim, compressed_summary_dim)
            self.memory_compress = torch.nn.Linear(dim, compressed_summary_dim)
            self.compressed_residual = torch.nn.Sequential(
                torch.nn.Linear(compressed_summary_dim * 4, hidden_dim // 2),
                torch.nn.GELU(),
                torch.nn.LayerNorm(hidden_dim // 2),
                torch.nn.Linear(hidden_dim // 2, 1),
            )
            self.residual_scale = torch.nn.Parameter(torch.tensor(0.1))
        else:
            self.block_compress = None
            self.history_compress = None
            self.query_compress = None
            self.memory_compress = None
            self.compressed_residual = None
            self.residual_scale = None

    def forward(self, features):
        base = features[:, : self.base_dim]
        score = self.base_scorer(base).squeeze(-1)
        if self.compressed_residual is None or features.shape[1] < self.base_dim + self.dim * 4:
            return score
        compressed = features[:, self.base_dim : self.base_dim + self.dim * 4]
        block_stats = compressed[:, : self.dim * 3]
        history = compressed[:, self.dim * 3 : self.dim * 4]
        query = base[:, : self.dim]
        memory = base[:, self.dim : self.dim * 2]
        z = torch.cat(
            [
                torch.nn.functional.gelu(self.block_compress(block_stats)),
                torch.nn.functional.gelu(self.history_compress(history)),
                torch.nn.functional.gelu(self.query_compress(query)),
                torch.nn.functional.gelu(self.memory_compress(memory)),
            ],
            dim=-1,
        )
        residual = self.compressed_residual(z).squeeze(-1)
        return score + torch.tanh(self.residual_scale) * residual


def teacher_turn_scores(item, cross_encoder, raw_top_n, batch_size, candidate_indices=None):
    memory_texts = [m["text"] for m in item["memories"]]
    if candidate_indices is None:
        q = item["query_embedding"]
        memory_embeddings = item["memory_embeddings"]
        raw_scores = cosine_scores(q, memory_embeddings)
        raw_order = list(np.argsort(-raw_scores))
        candidate_indices = raw_order[: min(raw_top_n, len(raw_order))]
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    pairs = [(item["query"], memory_texts[i]) for i in candidate_indices]
    ce_scores = np.asarray(
        cross_encoder.predict(pairs, batch_size=batch_size, show_progress_bar=False),
        dtype=np.float32,
    )
    return np.asarray(candidate_indices, dtype=np.int64), ce_scores


def load_teacher_cache(path):
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_teacher_cache(path, cache):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp_path.write_text(json.dumps(cache), encoding="utf-8")
    tmp_path.replace(path)


def cached_teacher_turn_scores(
    item,
    cross_encoder,
    raw_top_n,
    batch_size,
    cache,
    train_candidate_source="raw",
    train_candidate_top_n=None,
    dca_union_raw_frac=0.5,
    dca_union_block_size=16,
    dca_union_top_blocks=2,
    dca_union_neighbor_blocks=0,
    dca_union_block_weight=0.0,
):
    if train_candidate_source == "raw":
        key = f"{item['question_id']}|top{raw_top_n}"
        candidate_indices = None
    else:
        candidate_top_n = train_candidate_top_n or raw_top_n
        raw_scores = cosine_scores(item["query_embedding"], item["memory_embeddings"])
        if train_candidate_source == "mixed_union":
            raw_order = list(np.argsort(-raw_scores))
            union_indices = build_eval_candidate_indices(
                item,
                raw_scores,
                candidate_top_n,
                candidate_source="dca_union",
                dca_union_raw_frac=dca_union_raw_frac,
                dca_union_block_size=dca_union_block_size,
                dca_union_top_blocks=dca_union_top_blocks,
                dca_union_neighbor_blocks=dca_union_neighbor_blocks,
                dca_union_block_weight=dca_union_block_weight,
            )
            out = []
            seen = set()
            for idx in raw_order[: min(raw_top_n, len(raw_order))]:
                idx = int(idx)
                out.append(idx)
                seen.add(idx)
            for idx in union_indices:
                idx = int(idx)
                if idx not in seen:
                    out.append(idx)
                    seen.add(idx)
            candidate_indices = np.asarray(out, dtype=np.int64)
        else:
            candidate_indices = build_eval_candidate_indices(
                item,
                raw_scores,
                candidate_top_n,
                candidate_source=train_candidate_source,
                dca_union_raw_frac=dca_union_raw_frac,
                dca_union_block_size=dca_union_block_size,
                dca_union_top_blocks=dca_union_top_blocks,
                dca_union_neighbor_blocks=dca_union_neighbor_blocks,
                dca_union_block_weight=dca_union_block_weight,
            )
        key = (
            f"{item['question_id']}|train_{train_candidate_source}"
            f"|top{candidate_top_n}"
            f"|raw{raw_top_n}"
            f"|rf{dca_union_raw_frac:g}"
            f"|b{dca_union_block_size}"
            f"|tb{dca_union_top_blocks}"
            f"|nb{dca_union_neighbor_blocks}"
            f"|bw{dca_union_block_weight:g}"
        )
    if key in cache:
        cached = cache[key]
        return (
            np.asarray(cached["indices"], dtype=np.int64),
            np.asarray(cached["scores"], dtype=np.float32),
            True,
        )

    indices, scores = teacher_turn_scores(
        item,
        cross_encoder,
        raw_top_n,
        batch_size,
        candidate_indices=candidate_indices,
    )
    cache[key] = {
        "indices": indices.astype(int).tolist(),
        "scores": scores.astype(float).tolist(),
    }
    return indices, scores, False


def turn_scores_to_window_scores(item, candidate_indices, candidate_scores):
    turn_teacher = {
        int(idx): float(score)
        for idx, score in zip(candidate_indices, candidate_scores)
    }
    floor = float(np.min(candidate_scores) - 5.0) if len(candidate_scores) else -5.0
    scores = []
    for window in item["windows"]:
        values = [turn_teacher[i] for i in window if i in turn_teacher]
        scores.append(max(values) if values else floor)
    return np.asarray(scores, dtype=np.float32)


def window_scores(model, item, device):
    q = item["query_embedding"]
    query = torch.tensor(q[None, :], dtype=torch.float32, device=device)
    window_batch = item["window_tensor"].to(device)
    query_batch = query.expand(window_batch.shape[0], -1)
    kwargs = {}
    if "window_type_tensor" in item:
        kwargs["type_ids"] = item["window_type_tensor"].to(device)

    if hasattr(model, "score_windows") and getattr(model, "score_mode", "cosine") != "cosine":
        return model.score_windows(window_batch, query=query_batch, **kwargs)

    blocks = model(window_batch, query=query_batch, **kwargs)
    return (query @ blocks.T).squeeze(0)


def local_window_scores(model, item, device):
    q = item["query_embedding"]
    query = torch.tensor(q[None, :], dtype=torch.float32, device=device)
    window_batch = item["window_tensor"].to(device)
    query_batch = query.expand(window_batch.shape[0], -1)
    kwargs = {}
    if "window_type_tensor" in item:
        kwargs["type_ids"] = item["window_type_tensor"].to(device)
    blocks = model(window_batch, query=query_batch, **kwargs)
    return (query @ blocks.T).squeeze(0)


def best_window_scores_for_candidates(window_logits, windows, candidate_indices, device):
    out = []
    for memory_idx in candidate_indices:
        touching = [w_idx for w_idx, window in enumerate(windows) if int(memory_idx) in window]
        if touching:
            idx = torch.tensor(touching, dtype=torch.long, device=device)
            out.append(window_logits.index_select(0, idx).max())
        else:
            out.append(window_logits.min())
    return torch.stack(out)


def build_memory_to_windows(windows):
    memory_to_windows = {}
    for window_idx, window in enumerate(windows):
        for memory_idx in window:
            memory_to_windows.setdefault(int(memory_idx), []).append(window_idx)
    return memory_to_windows


def best_window_scores_for_candidates_fast(
    window_logits,
    memory_to_windows,
    candidate_indices,
    device,
):
    if not torch.is_grad_enabled():
        fallback_value = float(window_logits.min().detach().cpu())
        window_values = window_logits.detach().cpu().numpy()
        values = []
        for memory_idx in candidate_indices:
            touching = memory_to_windows.get(int(memory_idx))
            if not touching:
                values.append(fallback_value)
            else:
                values.append(float(window_values[touching].max()))
        return torch.tensor(values, dtype=torch.float32, device=device)

    fallback = window_logits.min()
    values = []
    for memory_idx in candidate_indices:
        touching = memory_to_windows.get(int(memory_idx))
        if not touching:
            values.append(fallback)
            continue
        idx = torch.tensor(touching, dtype=torch.long, device=device)
        values.append(window_logits.index_select(0, idx).max())
    return torch.stack(values)


def dca_router_outputs(
    item,
    candidate_indices,
    block_size,
    device,
    return_context=False,
    return_global_features=False,
    temperature=0.05,
):
    memory_embeddings = item["memory_embeddings"].astype(np.float32)
    q_np = item["query_embedding"].astype(np.float32)
    num_memories = memory_embeddings.shape[0]
    if num_memories == 0:
        scores = torch.zeros(len(candidate_indices), dtype=torch.float32, device=device)
        context = None
        global_features = None
        if return_context:
            context = torch.zeros((len(candidate_indices), memory_embeddings.shape[1]), dtype=torch.float32, device=device)
        if return_global_features:
            global_features = torch.zeros((len(candidate_indices), 3), dtype=torch.float32, device=device)
        return scores, context, global_features

    block_ids = np.arange(num_memories) // max(1, block_size)
    num_blocks = int(block_ids.max()) + 1
    block_embeddings = np.zeros((num_blocks, memory_embeddings.shape[1]), dtype=np.float32)
    for block_idx in range(num_blocks):
        block_embeddings[block_idx] = memory_embeddings[block_ids == block_idx].mean(axis=0)

    block_embeddings = block_embeddings / (np.linalg.norm(block_embeddings, axis=1, keepdims=True) + 1e-8)
    q_norm = q_np / (np.linalg.norm(q_np) + 1e-8)
    block_scores = block_embeddings @ q_norm
    candidate_blocks = block_ids[candidate_indices]
    candidate_scores = block_scores[candidate_blocks].astype(np.float32)
    scores = torch.tensor(candidate_scores, dtype=torch.float32, device=device)
    context = None
    if return_context:
        context = torch.tensor(block_embeddings[candidate_blocks], dtype=torch.float32, device=device)
    global_features = None
    if return_global_features:
        temp = max(float(temperature), 1e-4)
        shifted = (block_scores - float(block_scores.max())) / temp
        block_probs = np.exp(shifted)
        block_probs = block_probs / (float(block_probs.sum()) + 1e-12)
        ranks = np.empty(num_blocks, dtype=np.float32)
        ranks[np.argsort(-block_scores)] = np.arange(num_blocks, dtype=np.float32)
        rank_quality = 1.0 - ranks / max(1, num_blocks - 1)
        margin = candidate_scores - float(block_scores.max())
        values = np.stack(
            [
                np.log(block_probs[candidate_blocks] + 1e-12),
                rank_quality[candidate_blocks],
                margin.astype(np.float32),
            ],
            axis=1,
        ).astype(np.float32)
        global_features = torch.tensor(values, dtype=torch.float32, device=device)
    return scores, context, global_features


def dca_router_scores(item, candidate_indices, block_size, device, return_context=False):
    scores, context, _ = dca_router_outputs(
        item,
        candidate_indices,
        block_size,
        device,
        return_context=return_context,
    )
    if return_context:
        return scores, context
    return scores


def compressed_dca_features(item, candidate_indices, block_size, device, temperature=0.05):
    memory_embeddings = item["memory_embeddings"].astype(np.float32)
    q_np = item["query_embedding"].astype(np.float32)
    num_memories = memory_embeddings.shape[0]
    dim = memory_embeddings.shape[1]
    if num_memories == 0:
        return torch.zeros((len(candidate_indices), dim * 4), dtype=torch.float32, device=device)

    block_ids = np.arange(num_memories) // max(1, block_size)
    num_blocks = int(block_ids.max()) + 1
    block_mean = np.zeros((num_blocks, dim), dtype=np.float32)
    block_max = np.zeros((num_blocks, dim), dtype=np.float32)
    block_last = np.zeros((num_blocks, dim), dtype=np.float32)
    for block_idx in range(num_blocks):
        mask = block_ids == block_idx
        block = memory_embeddings[mask]
        block_mean[block_idx] = block.mean(axis=0)
        block_max[block_idx] = block.max(axis=0)
        block_last[block_idx] = block[-1]

    norm_mean = block_mean / (np.linalg.norm(block_mean, axis=1, keepdims=True) + 1e-8)
    q_norm = q_np / (np.linalg.norm(q_np) + 1e-8)
    block_scores = norm_mean @ q_norm
    shifted = (block_scores - float(block_scores.max())) / max(float(temperature), 1e-4)
    weights = np.exp(shifted)
    weights = weights / (float(weights.sum()) + 1e-12)
    history = (weights[:, None] * block_mean).sum(axis=0).astype(np.float32)

    candidate_blocks = block_ids[candidate_indices]
    block_stats = np.concatenate(
        [
            block_mean[candidate_blocks],
            block_max[candidate_blocks],
            block_last[candidate_blocks],
        ],
        axis=1,
    ).astype(np.float32)
    history_batch = np.repeat(history[None, :], len(candidate_indices), axis=0)
    features = np.concatenate([block_stats, history_batch], axis=1).astype(np.float32)
    return torch.tensor(features, dtype=torch.float32, device=device)


def parse_router_block_sizes(value):
    if not value:
        return []
    return [int(x.strip()) for x in value.split(",") if x.strip()]


@lru_cache(maxsize=250_000)
def lexical_token_tuple(text):
    text = str(text)
    return tuple(t for t in TOKEN_RE.findall(text.lower()) if len(t) > 1 and t not in STOPWORDS)


@lru_cache(maxsize=250_000)
def lexical_signature(text):
    tokens = lexical_token_tuple(str(text))
    return frozenset(tokens), frozenset(zip(tokens, tokens[1:]))


def lexical_tokens(text):
    return list(lexical_token_tuple(str(text)))


def token_bigrams(tokens):
    return set(zip(tokens, tokens[1:]))


def ensure_lexical_cache(item):
    cache = item.get("_lexical_cache")
    if cache is not None:
        return cache

    query_set, query_bigrams = lexical_signature(item["query"])
    cache = {
        "query_set": query_set,
        "query_bigrams": query_bigrams,
        "memory_signatures": [None for _ in item["memories"]],
    }
    item["_lexical_cache"] = cache
    return cache


def lexical_overlap_features(item, candidate_indices, device):
    cache = ensure_lexical_cache(item)
    query_set = cache["query_set"]
    query_bigrams = cache["query_bigrams"]
    memory_signatures = cache["memory_signatures"]
    rows = []
    for idx in candidate_indices:
        idx = int(idx)
        signature = memory_signatures[idx]
        if signature is None:
            signature = lexical_signature(item["memories"][idx]["text"])
            memory_signatures[idx] = signature
        memory_set, candidate_bigrams = signature
        overlap = query_set & memory_set
        union = query_set | memory_set
        rows.append(
            [
                len(overlap) / max(1, len(query_set)),
                len(overlap) / max(1, len(union)),
                len(query_bigrams & candidate_bigrams) / max(1, len(query_bigrams)),
                np.log1p(len(overlap)) / np.log1p(max(1, len(query_set))),
            ]
        )
    return torch.tensor(rows, dtype=torch.float32, device=device)


def candidate_features(
    model,
    item,
    candidate_indices,
    device,
    raw_scores_all=None,
    window_logits=None,
    memory_to_windows=None,
    dual_window_features=False,
    dca_router_block_size=0,
    dca_router_block_sizes=None,
    dca_router_context=False,
    dca_router_global_features=False,
    dca_router_temperature=0.05,
    dca_compress_block_size=0,
    dca_compress_temperature=0.05,
    lexical_features=False,
):
    q_np = item["query_embedding"].astype(np.float32)
    memory_np = item["memory_embeddings"][candidate_indices].astype(np.float32)
    if raw_scores_all is None:
        raw_scores_all = cosine_scores(q_np, item["memory_embeddings"])
    raw_scores = raw_scores_all[candidate_indices].astype(np.float32)
    raw_norm = normalize_scores(raw_scores)

    if window_logits is None:
        window_logits = window_scores(model, item, device)
    if memory_to_windows is None:
        memory_to_windows = build_memory_to_windows(item["windows"])
    best_window = best_window_scores_for_candidates_fast(
        window_logits,
        memory_to_windows,
        candidate_indices,
        device,
    )
    best_window_norm = (best_window - best_window.mean()) / (best_window.std(unbiased=False) + 1e-6)
    extra_features = []
    if dual_window_features:
        local_logits = local_window_scores(model, item, device)
        local_best = best_window_scores_for_candidates_fast(
            local_logits,
            memory_to_windows,
            candidate_indices,
            device,
        )
        local_norm = (local_best - local_best.mean()) / (local_best.std(unbiased=False) + 1e-6)
        delta = best_window - local_best
        delta_norm = (delta - delta.mean()) / (delta.std(unbiased=False) + 1e-6)
        extra_features.extend([local_norm[:, None], delta_norm[:, None]])
    router_block_sizes = dca_router_block_sizes
    if router_block_sizes is None:
        router_block_sizes = [dca_router_block_size] if dca_router_block_size > 0 else []
    router_context = None
    for router_idx, block_size in enumerate(router_block_sizes):
        need_context = dca_router_context and router_idx == 0
        router_scores, maybe_context, global_features = dca_router_outputs(
            item,
            candidate_indices,
            block_size,
            device,
            return_context=need_context,
            return_global_features=dca_router_global_features,
            temperature=dca_router_temperature,
        )
        router_norm = (router_scores - router_scores.mean()) / (router_scores.std(unbiased=False) + 1e-6)
        extra_features.append(router_norm[:, None])
        if dca_router_global_features and global_features is not None:
            extra_features.append(global_features)
        if need_context:
            router_context = maybe_context
    if dca_compress_block_size > 0:
        extra_features.append(
            compressed_dca_features(
                item,
                candidate_indices,
                dca_compress_block_size,
                device,
                temperature=dca_compress_temperature,
            )
        )
    if lexical_features:
        extra_features.append(lexical_overlap_features(item, candidate_indices, device))

    q = torch.tensor(q_np[None, :], dtype=torch.float32, device=device)
    memories = torch.tensor(memory_np, dtype=torch.float32, device=device)
    q_batch = q.expand(memories.shape[0], -1)
    if router_block_sizes and dca_router_context and router_context is not None:
        extra_features.extend(
            [
                router_context,
                q_batch * router_context,
                torch.abs(q_batch - router_context),
            ]
        )
    raw_rank = torch.linspace(0.0, 1.0, steps=memories.shape[0], dtype=torch.float32, device=device)
    position = torch.tensor(candidate_indices / max(1, len(item["memory_ids"]) - 1), dtype=torch.float32, device=device)
    raw_tensor = torch.tensor(raw_norm, dtype=torch.float32, device=device)

    features = torch.cat(
        [
            q_batch,
            memories,
            q_batch * memories,
            torch.abs(q_batch - memories),
            raw_tensor[:, None],
            best_window_norm[:, None],
            raw_rank[:, None],
            position[:, None],
            *extra_features,
        ],
        dim=-1,
    )
    return features, raw_scores, best_window.detach().cpu().numpy()


def train_one_item(
    model,
    scorer,
    optimizer,
    item,
    teacher_indices,
    teacher_scores,
    device,
    teacher_temperature,
    pairwise_top_k,
    pairwise_bottom_k,
    negative_mode,
    hard_negative_pool,
    gold_weight,
    score_mse_weight,
    gold_weight_single=None,
    gold_weight_multi=None,
    listwise_distill_weight=0.0,
    first_rank_weight=0.0,
    first_rank_target="best_gold",
    gold_first_rank_weight=0.0,
    teacher_first_rank_weight=0.0,
    dual_window_features=False,
    dca_router_block_size=0,
    dca_router_block_sizes=None,
    window_teacher_scores=None,
    window_distill_weight=0.0,
    window_gold_weight=0.0,
    dca_router_context=False,
    dca_router_global_features=False,
    dca_router_temperature=0.05,
    dca_compress_block_size=0,
    dca_compress_temperature=0.05,
    lexical_features=False,
):
    if len(teacher_indices) < 2:
        return 0.0

    features, _, _ = candidate_features(
        model,
        item,
        teacher_indices,
        device,
        dual_window_features=dual_window_features,
        dca_router_block_size=dca_router_block_size,
        dca_router_block_sizes=dca_router_block_sizes,
        dca_router_context=dca_router_context,
        dca_router_global_features=dca_router_global_features,
        dca_router_temperature=dca_router_temperature,
        dca_compress_block_size=dca_compress_block_size,
        dca_compress_temperature=dca_compress_temperature,
        lexical_features=lexical_features,
    )
    logits = scorer(features)
    teacher = torch.tensor(teacher_scores, dtype=torch.float32, device=device)

    top_k = min(pairwise_top_k, teacher.numel())
    bottom_k = min(pairwise_bottom_k, teacher.numel())
    top_idx = torch.topk(teacher, k=top_k).indices
    if negative_mode == "raw_hard":
        pool_size = min(max(hard_negative_pool, bottom_k + top_k), teacher.numel())
        pool_idx = torch.arange(pool_size, dtype=torch.long, device=device)
        top_mask = torch.zeros(teacher.numel(), dtype=torch.bool, device=device)
        top_mask[top_idx] = True
        pool_idx = pool_idx[~top_mask.index_select(0, pool_idx)]
        if pool_idx.numel() == 0:
            bottom_idx = torch.topk(-teacher, k=bottom_k).indices
        else:
            hard_k = min(bottom_k, pool_idx.numel())
            hard_scores = teacher.index_select(0, pool_idx)
            bottom_idx = pool_idx.index_select(0, torch.topk(-hard_scores, k=hard_k).indices)
    elif negative_mode == "mixed":
        global_k = max(1, bottom_k // 2)
        hard_k_target = max(1, bottom_k - global_k)
        global_neg = torch.topk(-teacher, k=min(global_k, teacher.numel())).indices
        pool_size = min(max(hard_negative_pool, hard_k_target + top_k), teacher.numel())
        pool_idx = torch.arange(pool_size, dtype=torch.long, device=device)
        banned = torch.zeros(teacher.numel(), dtype=torch.bool, device=device)
        banned[top_idx] = True
        banned[global_neg] = True
        pool_idx = pool_idx[~banned.index_select(0, pool_idx)]
        if pool_idx.numel() > 0:
            hard_scores = teacher.index_select(0, pool_idx)
            hard_neg = pool_idx.index_select(0, torch.topk(-hard_scores, k=min(hard_k_target, pool_idx.numel())).indices)
            bottom_idx = torch.cat([global_neg, hard_neg], dim=0)
        else:
            bottom_idx = global_neg
    else:
        bottom_idx = torch.topk(-teacher, k=bottom_k).indices
    pos_logits = logits.index_select(0, top_idx)
    neg_logits = logits.index_select(0, bottom_idx)
    pos_teacher = teacher.index_select(0, top_idx)
    neg_teacher = teacher.index_select(0, bottom_idx)
    logit_diff = pos_logits[:, None] - neg_logits[None, :]
    teacher_diff = pos_teacher[:, None] - neg_teacher[None, :]
    weights = torch.sigmoid(teacher_diff / teacher_temperature).detach()
    pairwise_loss = (torch.nn.functional.softplus(-logit_diff) * weights).mean()

    memory_to_candidate = {int(memory_idx): i for i, memory_idx in enumerate(teacher_indices)}
    memory_to_idx = {m: i for i, m in enumerate(item["memory_ids"])}
    gold_positions = [
        memory_to_candidate[memory_to_idx[g]]
        for g in item["gold_memory_ids"]
        if g in memory_to_idx and memory_to_idx[g] in memory_to_candidate
    ]
    if gold_positions:
        gold_idx = torch.tensor(gold_positions, dtype=torch.long, device=device)
        gold_logits = logits.index_select(0, gold_idx)
        gold_loss = torch.logsumexp(logits, dim=0) - torch.logsumexp(gold_logits, dim=0)
    else:
        gold_loss = torch.tensor(0.0, dtype=torch.float32, device=device)

    effective_gold_weight = gold_weight
    if len(item["gold_memory_ids"]) <= 1 and gold_weight_single is not None:
        effective_gold_weight = gold_weight_single
    elif len(item["gold_memory_ids"]) > 1 and gold_weight_multi is not None:
        effective_gold_weight = gold_weight_multi

    losses = [pairwise_loss + effective_gold_weight * gold_loss]

    if first_rank_weight > 0:
        target_idx = None
        if first_rank_target == "best_gold" and gold_positions:
            gold_idx = torch.tensor(gold_positions, dtype=torch.long, device=device)
            gold_teacher = teacher.index_select(0, gold_idx)
            target_idx = gold_idx[torch.argmax(gold_teacher)]
        elif first_rank_target == "teacher":
            target_idx = torch.argmax(teacher)
        if target_idx is not None:
            first_rank_loss = torch.logsumexp(logits, dim=0) - logits[target_idx]
            losses.append(first_rank_weight * first_rank_loss)

    if gold_first_rank_weight > 0 and gold_positions:
        gold_idx = torch.tensor(gold_positions, dtype=torch.long, device=device)
        gold_teacher = teacher.index_select(0, gold_idx)
        gold_target_idx = gold_idx[torch.argmax(gold_teacher)]
        gold_first_rank_loss = torch.logsumexp(logits, dim=0) - logits[gold_target_idx]
        losses.append(gold_first_rank_weight * gold_first_rank_loss)

    if teacher_first_rank_weight > 0:
        teacher_target_idx = torch.argmax(teacher)
        teacher_first_rank_loss = torch.logsumexp(logits, dim=0) - logits[teacher_target_idx]
        losses.append(teacher_first_rank_weight * teacher_first_rank_loss)

    if listwise_distill_weight > 0:
        teacher_probs = torch.softmax(teacher / teacher_temperature, dim=0).detach()
        student_log_probs = torch.log_softmax(logits / teacher_temperature, dim=0)
        listwise_loss = torch.nn.functional.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="batchmean",
        )
        losses.append(listwise_distill_weight * listwise_loss * teacher_temperature * teacher_temperature)

    if score_mse_weight > 0:
        teacher_norm = (teacher - teacher.mean()) / (teacher.std(unbiased=False) + 1e-6)
        logits_norm = (logits - logits.mean()) / (logits.std(unbiased=False) + 1e-6)
        losses.append(score_mse_weight * torch.nn.functional.mse_loss(logits_norm, teacher_norm))

    if window_distill_weight > 0 and window_teacher_scores is not None:
        win_logits = window_scores(model, item, device)
        win_teacher = torch.tensor(window_teacher_scores, dtype=torch.float32, device=device)
        win_top_k = min(pairwise_top_k, win_teacher.numel())
        win_bottom_k = min(pairwise_bottom_k, win_teacher.numel())
        win_top_idx = torch.topk(win_teacher, k=win_top_k).indices
        win_bottom_idx = torch.topk(-win_teacher, k=win_bottom_k).indices
        win_pos = win_logits.index_select(0, win_top_idx)
        win_neg = win_logits.index_select(0, win_bottom_idx)
        win_pos_teacher = win_teacher.index_select(0, win_top_idx)
        win_neg_teacher = win_teacher.index_select(0, win_bottom_idx)
        win_logit_diff = win_pos[:, None] - win_neg[None, :]
        win_teacher_diff = win_pos_teacher[:, None] - win_neg_teacher[None, :]
        win_weights = torch.sigmoid(win_teacher_diff / teacher_temperature).detach()
        win_pairwise_loss = (torch.nn.functional.softplus(-win_logit_diff) * win_weights).mean()

        memory_to_idx = {m: i for i, m in enumerate(item["memory_ids"])}
        gold_indices = {memory_to_idx[g] for g in item["gold_memory_ids"] if g in memory_to_idx}
        positive_windows = [
            i for i, window in enumerate(item["windows"])
            if gold_indices & set(window)
        ]
        if positive_windows:
            pos_idx = torch.tensor(positive_windows, dtype=torch.long, device=device)
            pos_win_logits = win_logits.index_select(0, pos_idx)
            win_gold_loss = torch.logsumexp(win_logits, dim=0) - torch.logsumexp(pos_win_logits, dim=0)
        else:
            win_gold_loss = torch.tensor(0.0, dtype=torch.float32, device=device)

        losses.append(window_distill_weight * (win_pairwise_loss + window_gold_weight * win_gold_loss))

    loss = sum(losses)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


def score_ce_lite(
    model,
    scorer,
    item,
    candidate_indices,
    device,
    raw_scores_all=None,
    window_logits=None,
    memory_to_windows=None,
    dual_window_features=False,
    dca_router_block_size=0,
    dca_router_block_sizes=None,
    dca_router_context=False,
    dca_router_global_features=False,
    dca_router_temperature=0.05,
    dca_compress_block_size=0,
    dca_compress_temperature=0.05,
    lexical_features=False,
):
    with torch.no_grad():
        features, raw_scores, window_values = candidate_features(
            model,
            item,
            candidate_indices,
            device,
            raw_scores_all=raw_scores_all,
            window_logits=window_logits,
            memory_to_windows=memory_to_windows,
            dual_window_features=dual_window_features,
            dca_router_block_size=dca_router_block_size,
            dca_router_block_sizes=dca_router_block_sizes,
            dca_router_context=dca_router_context,
            dca_router_global_features=dca_router_global_features,
            dca_router_temperature=dca_router_temperature,
            dca_compress_block_size=dca_compress_block_size,
            dca_compress_temperature=dca_compress_temperature,
            lexical_features=lexical_features,
        )
        ce_lite_scores = scorer(features).cpu().numpy()
    return raw_scores, window_values, ce_lite_scores


def rerank_candidates(raw_scores_all, candidate_indices, candidate_scores, memory_ids, raw_weight):
    raw_candidate = raw_scores_all[candidate_indices]
    raw_norm = normalize_scores(raw_candidate)
    score_norm = normalize_scores(candidate_scores)
    final = raw_weight * raw_norm + (1.0 - raw_weight) * score_norm
    order = [int(candidate_indices[i]) for i in np.argsort(-final)]
    ranked = [memory_ids[i] for i in order]
    ranked_set = set(ranked)
    raw_order = np.argsort(-raw_scores_all)
    ranked.extend([memory_ids[int(i)] for i in raw_order if memory_ids[int(i)] not in ranked_set])
    return ranked


def dca_block_selector_order(
    item,
    raw_scores,
    block_size=16,
    top_blocks=2,
    neighbor_blocks=0,
    block_weight=0.0,
):
    memory_embeddings = item["memory_embeddings"].astype(np.float32)
    q_np = item["query_embedding"].astype(np.float32)
    num_memories = memory_embeddings.shape[0]
    if num_memories == 0:
        return []

    raw_order = list(np.argsort(-raw_scores))
    block_ids = np.arange(num_memories) // max(1, block_size)
    num_blocks = int(block_ids.max()) + 1
    block_mean = np.zeros((num_blocks, memory_embeddings.shape[1]), dtype=np.float32)
    for block_idx in range(num_blocks):
        block_mean[block_idx] = memory_embeddings[block_ids == block_idx].mean(axis=0)
    block_mean = block_mean / (np.linalg.norm(block_mean, axis=1, keepdims=True) + 1e-8)
    q_norm = q_np / (np.linalg.norm(q_np) + 1e-8)
    block_scores = block_mean @ q_norm

    selected_blocks = set()
    for block_idx in np.argsort(-block_scores)[: min(top_blocks, num_blocks)]:
        for offset in range(-neighbor_blocks, neighbor_blocks + 1):
            b = int(block_idx) + offset
            if 0 <= b < num_blocks:
                selected_blocks.add(b)

    selected_indices = np.asarray(
        [idx for idx in range(num_memories) if int(block_ids[idx]) in selected_blocks],
        dtype=np.int64,
    )
    if selected_indices.size == 0:
        return raw_order

    raw_norm = normalize_scores(raw_scores[selected_indices])
    block_norm = normalize_scores(block_scores[block_ids[selected_indices]])
    final_scores = raw_norm + block_weight * block_norm
    selected_order = [int(selected_indices[i]) for i in np.argsort(-final_scores)]
    selected_set = set(selected_order)
    selected_order.extend([idx for idx in raw_order if idx not in selected_set])
    return selected_order


def build_eval_candidate_indices(
    item,
    raw_scores,
    top_n,
    candidate_source="raw",
    dca_union_raw_frac=0.5,
    dca_union_block_size=16,
    dca_union_top_blocks=2,
    dca_union_neighbor_blocks=0,
    dca_union_block_weight=0.0,
):
    raw_order = list(np.argsort(-raw_scores))
    top_n = min(top_n, len(raw_order))
    if candidate_source in ("raw", "both"):
        return np.asarray(raw_order[:top_n], dtype=np.int64)

    raw_keep = max(1, min(top_n, int(round(top_n * dca_union_raw_frac))))
    dca_order = dca_block_selector_order(
        item,
        raw_scores,
        block_size=dca_union_block_size,
        top_blocks=dca_union_top_blocks,
        neighbor_blocks=dca_union_neighbor_blocks,
        block_weight=dca_union_block_weight,
    )
    out = []
    seen = set()
    for idx in raw_order[:raw_keep]:
        out.append(int(idx))
        seen.add(int(idx))
    for idx in dca_order:
        idx = int(idx)
        if idx not in seen:
            out.append(idx)
            seen.add(idx)
        if len(out) >= top_n:
            break
    return np.asarray(out[:top_n], dtype=np.int64)


def evaluate_item_ce_lite(
    item,
    model,
    scorer,
    device,
    raw_weights,
    rerank_top_ns,
    dual_window_features=False,
    dca_router_block_size=0,
    dca_router_block_sizes=None,
    dca_router_context=False,
    dca_router_global_features=False,
    dca_router_temperature=0.05,
    dca_compress_block_size=0,
    dca_compress_temperature=0.05,
    candidate_source="raw",
    dca_union_raw_frac=0.5,
    dca_union_block_size=16,
    dca_union_top_blocks=2,
    dca_union_neighbor_blocks=0,
    dca_union_block_weight=0.0,
    lexical_features=False,
):
    q = item["query_embedding"]
    memory_ids = item["memory_ids"]
    raw_scores = cosine_scores(q, item["memory_embeddings"])
    raw_order = np.argsort(-raw_scores)

    with torch.no_grad():
        conv_tensor = window_scores(model, item, device)
        conv_scores = conv_tensor.cpu().numpy()

    rankings = {
        "raw_turn": [memory_ids[int(i)] for i in raw_order],
    }

    memory_to_windows = build_memory_to_windows(item["windows"])

    eval_candidate_sources = ["raw", "dca_union"] if candidate_source == "both" else [candidate_source]

    for top_n in rerank_top_ns:
        rankings[f"convmem_rerank_top{top_n}_rw0.5"] = convmem_rerank(
            raw_scores,
            conv_scores,
            item["windows"],
            memory_ids,
            raw_top_n=top_n,
            raw_weight=0.5,
        )
        for source in eval_candidate_sources:
            candidate_indices = build_eval_candidate_indices(
                item,
                raw_scores,
                top_n,
                candidate_source=source,
                dca_union_raw_frac=dca_union_raw_frac,
                dca_union_block_size=dca_union_block_size,
                dca_union_top_blocks=dca_union_top_blocks,
                dca_union_neighbor_blocks=dca_union_neighbor_blocks,
                dca_union_block_weight=dca_union_block_weight,
            )
            _, _, ce_lite_scores = score_ce_lite(
                model,
                scorer,
                item,
                candidate_indices,
                device,
                raw_scores_all=raw_scores,
                window_logits=conv_tensor,
                memory_to_windows=memory_to_windows,
                dual_window_features=dual_window_features,
                dca_router_block_size=dca_router_block_size,
                dca_router_block_sizes=dca_router_block_sizes,
                dca_router_context=dca_router_context,
                dca_router_global_features=dca_router_global_features,
                dca_router_temperature=dca_router_temperature,
                dca_compress_block_size=dca_compress_block_size,
                dca_compress_temperature=dca_compress_temperature,
                lexical_features=lexical_features,
            )
            method_prefix = "celite" if source == "raw" else "celite_dca_union"
            for weight in raw_weights:
                rankings[f"{method_prefix}_rerank_top{top_n}_rw{weight:g}"] = rerank_candidates(
                    raw_scores,
                    candidate_indices,
                    ce_lite_scores,
                    memory_ids,
                    raw_weight=weight,
                )

    rows = []
    for method, ranked in rankings.items():
        rows.append(
            {
                "method": method,
                "question_id": item["question_id"],
                "question_type": item["question_type"],
                "gold_chain_len": len(item["gold_memory_ids"]),
                "recall_at_5": recall_at_k(ranked, item["gold_memory_ids"], 5),
                "recall_at_10": recall_at_k(ranked, item["gold_memory_ids"], 10),
                "recall_at_20": recall_at_k(ranked, item["gold_memory_ids"], 20),
                "hit_at_5": hit_at_k(ranked, item["gold_memory_ids"], 5),
                "hit_at_10": hit_at_k(ranked, item["gold_memory_ids"], 10),
                "hit_at_20": hit_at_k(ranked, item["gold_memory_ids"], 20),
                "mrr": mrr(ranked, item["gold_memory_ids"]),
                "top20_ids": "||".join(ranked[:20]),
                "gold_ids": "||".join(item["gold_memory_ids"]),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/locomo10.json")
    parser.add_argument("--encoder-model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--encoder-batch-size", type=int, default=32)
    parser.add_argument("--embedding-cache", default=None)
    parser.add_argument("--embedding-cache-key", default=None)
    parser.add_argument("--cross-encoder-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--cross-batch-size", type=int, default=64)
    parser.add_argument("--teacher-cache", default=None)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--kernel", type=int, default=3)
    parser.add_argument("--architecture", choices=["mixer", "dca_mixer"], default="mixer")
    parser.add_argument("--mixer-hidden-dim", type=int, default=256)
    parser.add_argument("--mixer-token-dim", type=int, default=32)
    parser.add_argument("--mixer-channel-dim", type=int, default=512)
    parser.add_argument("--dca-chunk-hidden-dim", type=int, default=128)
    parser.add_argument("--dca-gate-init", type=float, default=0.03)
    parser.add_argument("--dca-mode", choices=["correction", "prior"], default="prior")
    parser.add_argument("--dca-gate-mode", choices=["fixed", "confidence"], default="fixed")
    parser.add_argument("--dual-window-features", action="store_true")
    parser.add_argument("--dca-router-block-size", type=int, default=0)
    parser.add_argument("--dca-router-block-sizes", default=None)
    parser.add_argument("--dca-router-context", action="store_true")
    parser.add_argument("--dca-router-global-features", action="store_true")
    parser.add_argument("--dca-router-temperature", type=float, default=0.05)
    parser.add_argument("--dca-compress-block-size", type=int, default=0)
    parser.add_argument("--dca-compress-temperature", type=float, default=0.05)
    parser.add_argument("--dca-compress-summary-dim", type=int, default=64)
    parser.add_argument("--film-scale", type=float, default=0.2)
    parser.add_argument("--candidate-source", choices=["raw", "dca_union", "both"], default="raw")
    parser.add_argument("--train-candidate-source", choices=["raw", "dca_union", "mixed_union"], default="raw")
    parser.add_argument("--train-candidate-top-n", type=int, default=None)
    parser.add_argument("--dca-union-raw-frac", type=float, default=0.5)
    parser.add_argument("--dca-union-block-size", type=int, default=16)
    parser.add_argument("--dca-union-top-blocks", type=int, default=2)
    parser.add_argument("--dca-union-neighbor-blocks", type=int, default=0)
    parser.add_argument("--dca-union-block-weight", type=float, default=0.0)
    parser.add_argument("--lexical-features", action="store_true")
    parser.add_argument(
        "--ce-scorer",
        choices=["mlp", "pairwise", "gated", "film", "compressed_router", "compressed_residual"],
        default="mlp",
    )
    parser.add_argument("--ce-hidden-dim", type=int, default=256)
    parser.add_argument("--interaction-dim", type=int, default=96)
    parser.add_argument("--pretrain-data", default="data/longmemeval_s_cleaned.json")
    parser.add_argument("--pretrain-limit", type=int, default=None)
    parser.add_argument("--pretrain-epochs", type=int, default=1)
    parser.add_argument("--save-conv-state", default=None)
    parser.add_argument("--load-conv-state", default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--raw-top-n", type=int, default=500)
    parser.add_argument("--raw-weights", default="0.2,0.3,0.4,0.5")
    parser.add_argument("--rerank-top-ns", default="100,200,500")
    parser.add_argument("--teacher-temperature", type=float, default=2.0)
    parser.add_argument("--pairwise-top-k", type=int, default=16)
    parser.add_argument("--pairwise-bottom-k", type=int, default=64)
    parser.add_argument("--negative-mode", choices=["bottom", "raw_hard", "mixed"], default="bottom")
    parser.add_argument("--hard-negative-pool", type=int, default=128)
    parser.add_argument("--gold-weight", type=float, default=0.125)
    parser.add_argument("--gold-weight-single", type=float, default=None)
    parser.add_argument("--gold-weight-multi", type=float, default=None)
    parser.add_argument("--listwise-distill-weight", type=float, default=0.0)
    parser.add_argument("--first-rank-weight", type=float, default=0.0)
    parser.add_argument("--first-rank-target", choices=["best_gold", "teacher"], default="best_gold")
    parser.add_argument("--gold-first-rank-weight", type=float, default=0.0)
    parser.add_argument("--teacher-first-rank-weight", type=float, default=0.0)
    parser.add_argument("--score-mse-weight", type=float, default=0.0)
    parser.add_argument("--window-distill-weight", type=float, default=0.0)
    parser.add_argument("--window-gold-weight", type=float, default=0.125)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--out", default="results/locomo/ce_lite")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if args.device == "auto" and device != "cuda":
        device = "cpu"

    all_examples = load_locomo_examples(Path(__file__).parent / args.data)
    train_examples = choose_split(all_examples, "dev", 0.5, args.seed)
    test_examples = choose_split(all_examples, "test", 0.5, args.seed)
    if args.train_limit:
        train_examples = train_examples[: args.train_limit]
    if args.test_limit:
        test_examples = test_examples[: args.test_limit]

    encoder = SentenceTransformerTextEncoder(
        model_name=args.encoder_model,
        device=device,
        batch_size=args.encoder_batch_size,
        cache_path=args.embedding_cache,
        cache_key=args.embedding_cache_key,
    )
    cross_encoder = CrossEncoder(resolve_local_model_path(args.cross_encoder_model), device=device)

    first = prepare_encoded_example(train_examples[0], encoder, args.window_size, args.stride)
    dim = first["memory_embeddings"].shape[1]
    if args.architecture == "dca_mixer":
        model = DCAConvMemoryEncoder(
            dim,
            window_size=args.window_size,
            kernel_size=args.kernel,
            hidden_dim=args.mixer_hidden_dim,
            token_mlp_dim=args.mixer_token_dim,
            channel_mlp_dim=args.mixer_channel_dim,
            chunk_hidden_dim=args.dca_chunk_hidden_dim,
            output_mode="residual",
            output_gate_init=0.1,
            dca_gate_init=args.dca_gate_init,
            dca_mode=args.dca_mode,
            dca_gate_mode=args.dca_gate_mode,
        ).to(device)
    else:
        model = MixerConvMemoryEncoder(
            dim,
            window_size=args.window_size,
            kernel_size=args.kernel,
            hidden_dim=args.mixer_hidden_dim,
            token_mlp_dim=args.mixer_token_dim,
            channel_mlp_dim=args.mixer_channel_dim,
            output_mode="residual",
            output_gate_init=0.1,
            score_mode="cosine",
        ).to(device)
    router_block_sizes = parse_router_block_sizes(args.dca_router_block_sizes)
    if not router_block_sizes and args.dca_router_block_size > 0:
        router_block_sizes = [args.dca_router_block_size]
    extra_scalar_features = 0
    if args.dual_window_features:
        extra_scalar_features += 2
    extra_scalar_features += len(router_block_sizes)
    if args.dca_router_global_features:
        extra_scalar_features += len(router_block_sizes) * 3
    if args.lexical_features:
        extra_scalar_features += 4
    extra_dense_features = 0
    if router_block_sizes and args.dca_router_context:
        extra_dense_features += dim * 3
    if args.dca_compress_block_size > 0:
        extra_dense_features += dim * 4
    scorer_map = {
        "mlp": CELiteScorer,
        "pairwise": PairwiseCELiteScorer,
        "gated": GatedCELiteScorer,
        "film": FiLMCELiteScorer,
        "compressed_router": CompressedRouterCELiteScorer,
        "compressed_residual": CompressedResidualCELiteScorer,
    }
    scorer_cls = scorer_map[args.ce_scorer]
    scorer_kwargs = {
        "hidden_dim": args.ce_hidden_dim,
        "extra_scalar_features": extra_scalar_features,
        "extra_dense_features": extra_dense_features,
    }
    if args.ce_scorer == "gated":
        scorer_kwargs["router_signal_count"] = len(router_block_sizes)
    if args.ce_scorer == "film":
        scorer_kwargs["compressed_summary_dim"] = args.dca_compress_summary_dim
        scorer_kwargs["film_scale"] = args.film_scale
    if args.ce_scorer == "compressed_router":
        scorer_kwargs["compressed_summary_dim"] = args.dca_compress_summary_dim
    if args.ce_scorer == "compressed_residual":
        scorer_kwargs["compressed_summary_dim"] = args.dca_compress_summary_dim
    if args.ce_scorer == "pairwise":
        scorer_kwargs["interaction_dim"] = args.interaction_dim
    scorer = scorer_cls(dim, **scorer_kwargs).to(device)

    pretrain_time = 0.0
    if args.load_conv_state:
        state_path = Path(__file__).parent / args.load_conv_state
        load_result = model.load_state_dict(
            torch.load(state_path, map_location="cpu"),
            strict=args.architecture != "dca_mixer",
        )
        print(f"loaded conv state: {state_path}")
        if args.architecture == "dca_mixer":
            for gate_name in ("score_gate", "bridge_gate", "global_gate", "attention_gate"):
                gate = getattr(model, gate_name, None)
                if gate is not None:
                    gate.data.fill_(float(args.dca_gate_init))
            print(
                "dca partial load: "
                f"missing={len(load_result.missing_keys)} "
                f"unexpected={len(load_result.unexpected_keys)}"
                f" reset_gate={args.dca_gate_init}"
            )
    elif args.pretrain_epochs > 0:
        pretrain_start = time.perf_counter()
        pretrain_examples = load_chain_examples(
            Path(__file__).parent / args.pretrain_data,
            limit=args.pretrain_limit,
            seed=args.seed,
        )
        encoded_pretrain = [
            prepare_encoded_example(example, encoder, args.window_size, args.stride)
            for example in pretrain_examples
        ]
        encoded_pretrain = [x for x in encoded_pretrain if x is not None]
        train_conv(model, encoded_pretrain, epochs=args.pretrain_epochs, device=device)
        pretrain_time = time.perf_counter() - pretrain_start
        if args.save_conv_state:
            state_path = Path(__file__).parent / args.save_conv_state
            state_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), state_path)
            print(f"saved conv state: {state_path}")

    teacher_cache_path = Path(__file__).parent / args.teacher_cache if args.teacher_cache else None
    teacher_cache = load_teacher_cache(teacher_cache_path)
    teacher_cache_misses = 0

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(scorer.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    train_start = time.perf_counter()
    for epoch in range(args.epochs):
        losses = []
        model.train()
        scorer.train()
        for idx, example in enumerate(train_examples, start=1):
            item = first if epoch == 0 and idx == 1 else prepare_encoded_example(
                example,
                encoder,
                args.window_size,
                args.stride,
            )
            if item is None:
                continue
            teacher_indices, teacher_scores, teacher_was_cached = cached_teacher_turn_scores(
                item,
                cross_encoder,
                raw_top_n=args.raw_top_n,
                batch_size=args.cross_batch_size,
                cache=teacher_cache,
                train_candidate_source=args.train_candidate_source,
                train_candidate_top_n=args.train_candidate_top_n,
                dca_union_raw_frac=args.dca_union_raw_frac,
                dca_union_block_size=args.dca_union_block_size,
                dca_union_top_blocks=args.dca_union_top_blocks,
                dca_union_neighbor_blocks=args.dca_union_neighbor_blocks,
                dca_union_block_weight=args.dca_union_block_weight,
            )
            if not teacher_was_cached:
                teacher_cache_misses += 1
                if teacher_cache_misses % 50 == 0:
                    save_teacher_cache(teacher_cache_path, teacher_cache)
            window_teacher = None
            if args.window_distill_weight > 0:
                window_teacher = turn_scores_to_window_scores(item, teacher_indices, teacher_scores)
            loss = train_one_item(
                model,
                scorer,
                optimizer,
                item,
                teacher_indices,
                teacher_scores,
                device=device,
                teacher_temperature=args.teacher_temperature,
                pairwise_top_k=args.pairwise_top_k,
                pairwise_bottom_k=args.pairwise_bottom_k,
                negative_mode=args.negative_mode,
                hard_negative_pool=args.hard_negative_pool,
                gold_weight=args.gold_weight,
                score_mse_weight=args.score_mse_weight,
                gold_weight_single=args.gold_weight_single,
                gold_weight_multi=args.gold_weight_multi,
                listwise_distill_weight=args.listwise_distill_weight,
                first_rank_weight=args.first_rank_weight,
                first_rank_target=args.first_rank_target,
                gold_first_rank_weight=args.gold_first_rank_weight,
                teacher_first_rank_weight=args.teacher_first_rank_weight,
                dual_window_features=args.dual_window_features,
                dca_router_block_size=args.dca_router_block_size,
                dca_router_block_sizes=router_block_sizes,
                dca_router_context=args.dca_router_context,
                dca_router_global_features=args.dca_router_global_features,
                dca_router_temperature=args.dca_router_temperature,
                dca_compress_block_size=args.dca_compress_block_size,
                dca_compress_temperature=args.dca_compress_temperature,
                lexical_features=args.lexical_features,
                window_teacher_scores=window_teacher,
                window_distill_weight=args.window_distill_weight,
                window_gold_weight=args.window_gold_weight,
            )
            losses.append(loss)
            if idx % 100 == 0:
                print(
                    f"epoch {epoch + 1}/{args.epochs} "
                    f"processed {idx}/{len(train_examples)} "
                    f"loss={np.mean(losses[-100:]):.4f}"
                )
    train_time = time.perf_counter() - train_start
    save_teacher_cache(teacher_cache_path, teacher_cache)

    raw_weights = [float(x.strip()) for x in args.raw_weights.split(",") if x.strip()]
    rerank_top_ns = [int(x.strip()) for x in args.rerank_top_ns.split(",") if x.strip()]
    rows = []
    eval_start = time.perf_counter()
    model.eval()
    scorer.eval()
    for example in test_examples:
        item = prepare_encoded_example(example, encoder, args.window_size, args.stride)
        if item is None:
            continue
        rows.extend(
            evaluate_item_ce_lite(
                item,
                model,
                scorer,
                device,
                raw_weights,
                rerank_top_ns,
                dual_window_features=args.dual_window_features,
                dca_router_block_size=args.dca_router_block_size,
                dca_router_block_sizes=router_block_sizes,
                dca_router_context=args.dca_router_context,
                dca_router_global_features=args.dca_router_global_features,
                dca_router_temperature=args.dca_router_temperature,
                dca_compress_block_size=args.dca_compress_block_size,
                dca_compress_temperature=args.dca_compress_temperature,
                candidate_source=args.candidate_source,
                dca_union_raw_frac=args.dca_union_raw_frac,
                dca_union_block_size=args.dca_union_block_size,
                dca_union_top_blocks=args.dca_union_top_blocks,
                dca_union_neighbor_blocks=args.dca_union_neighbor_blocks,
                dca_union_block_weight=args.dca_union_block_weight,
                lexical_features=args.lexical_features,
            )
        )
    eval_time = time.perf_counter() - eval_start

    out_dir = Path(__file__).parent / args.out
    summary = summarize(rows)
    write_csv(out_dir / "ce_lite_detailed_results.csv", rows)
    write_csv(out_dir / "ce_lite_summary_results.csv", summary)

    print("\nCE-lite ConvMem LoCoMo")
    print(f"train/dev questions: {len(train_examples)}")
    print(f"test questions: {len(test_examples)}")
    print(f"device: {device}")
    print(f"architecture: {args.architecture}")
    print(f"ce scorer: {args.ce_scorer}")
    print(f"dual window features: {args.dual_window_features}")
    print(f"lexical features: {args.lexical_features}")
    print(f"dca router block size: {args.dca_router_block_size}")
    print(f"dca router block sizes: {router_block_sizes}")
    print(f"dca router context: {args.dca_router_context}")
    print(f"dca router global features: {args.dca_router_global_features}")
    print(f"dca router temperature: {args.dca_router_temperature}")
    print(f"dca compress block size: {args.dca_compress_block_size}")
    print(f"dca compress summary dim: {args.dca_compress_summary_dim}")
    print(f"film scale: {args.film_scale}")
    print(
        "candidate source: "
        f"{args.candidate_source} "
        f"raw_frac={args.dca_union_raw_frac} "
        f"block={args.dca_union_block_size} "
        f"top_blocks={args.dca_union_top_blocks} "
        f"neighbors={args.dca_union_neighbor_blocks} "
        f"block_weight={args.dca_union_block_weight}"
    )
    print(
        "train candidate source: "
        f"{args.train_candidate_source} "
        f"top_n={args.train_candidate_top_n or args.raw_top_n}"
    )
    if args.architecture == "dca_mixer":
        print(
            "dca: "
            f"mode={args.dca_mode} "
            f"gate={args.dca_gate_init} "
            f"gate_mode={args.dca_gate_mode} "
            f"chunk_hidden={args.dca_chunk_hidden_dim}"
        )
    print(f"pretrain time: {pretrain_time:.1f}s")
    print(f"train time: {train_time:.1f}s")
    print(
        "window distill: "
        f"weight={args.window_distill_weight} "
        f"gold_weight={args.window_gold_weight}"
    )
    print(f"listwise distill weight: {args.listwise_distill_weight}")
    print(f"first-rank loss: weight={args.first_rank_weight} target={args.first_rank_target}")
    print(
        "mixed first-rank loss: "
        f"gold={args.gold_first_rank_weight} teacher={args.teacher_first_rank_weight}"
    )
    print(f"score mse weight: {args.score_mse_weight}")
    print(f"negative sampling: mode={args.negative_mode} hard_pool={args.hard_negative_pool}")
    print(f"eval latency: {1000 * eval_time / max(1, len(test_examples)):.2f}ms/query")
    print("\nmethod                              questions  recall@10 hit@10 mrr")
    for row in summary:
        print(
            f"{row['method']:<35} "
            f"{row['questions']:<9} "
            f"{row['recall_at_10']:.3f}     "
            f"{row['hit_at_10']:.3f}  "
            f"{row['mrr']:.3f}"
        )
    print(f"\nSaved: {out_dir / 'ce_lite_summary_results.csv'}")


if __name__ == "__main__":
    main()

"""Chinese ConvMemory dual-space reranker.

This module packages the v601 Chinese ConvMemory result as a normal user-facing
component. It keeps the ConvMemory inference boundary: query and memories are
encoded separately, then reranked by a lightweight learned scorer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .encoder import MixerConvMemoryEncoder
from .hub import resolve_checkpoint_path
from .reranker import RerankResult, normalize_rows, sliding_windows, window_tensor
from .scoring import (
    build_memory_to_windows,
    cosine_scores,
    dca_router_outputs,
    lexical_overlap_features,
    normalize_scores,
    rerank_candidates,
    window_scores,
)


class PairwiseCELiteScorer(torch.nn.Module):
    """Pairwise CE-lite scorer used by the Chinese dual-space checkpoint."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 512,
        extra_scalar_features: int = 7,
        extra_dense_features: int = 0,
        interaction_dim: int = 128,
    ):
        super().__init__()
        self.dim = int(dim)
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
            [qz, mz, qz * mz, torch.abs(qz - mz), scalars],
            dim=-1,
        )
        h_base = self.base(features)
        h_interaction = self.interaction(interaction_features)
        return self.out(torch.cat([h_base, h_interaction], dim=-1)).squeeze(-1)


class DualSpaceTextEncoder:
    """Concatenate two normalized SentenceTransformer spaces and renormalize."""

    def __init__(
        self,
        base_model: str,
        tuned_model: str,
        *,
        device: str = "cpu",
        batch_size: int = 64,
        trust_remote_code: bool = True,
    ):
        self.base_model_name = str(base_model)
        self.tuned_model_name = str(tuned_model)
        self.device = device
        self.batch_size = int(batch_size)
        self.base = SentenceTransformer(
            self.base_model_name,
            device=device,
            trust_remote_code=trust_remote_code,
        )
        self.tuned = SentenceTransformer(
            self.tuned_model_name,
            device=device,
            trust_remote_code=trust_remote_code,
        )

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        texts = list(texts)
        base = self.base.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)
        tuned = self.tuned.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)
        merged = np.concatenate([base, tuned], axis=1).astype(np.float32)
        return normalize_rows(merged)


def _parse_memories(memories: Iterable):
    memory_ids = []
    memory_texts = []
    for idx, memory in enumerate(memories):
        if isinstance(memory, dict):
            memory_ids.append(str(memory.get("id", idx)))
            memory_texts.append(str(memory.get("text", "")))
        else:
            memory_ids.append(str(idx))
            memory_texts.append(str(memory))
    return memory_ids, memory_texts


def _candidate_local_windows(num_items: int, candidate_indices, window_size: int):
    if num_items <= 0:
        return []
    if num_items <= window_size:
        return [list(range(num_items))]
    windows = []
    seen = set()
    half = window_size // 2
    for idx in candidate_indices:
        idx = int(idx)
        start = idx - half
        end = start + window_size
        if start < 0:
            start = 0
            end = window_size
        if end > num_items:
            end = num_items
            start = max(0, end - window_size)
        window = tuple(range(start, end))
        if window not in seen:
            seen.add(window)
            windows.append(list(window))
    return windows


class ChineseConvMemory:
    """Chinese dual-space ConvMemory reranker.

    Normal usage:

    ```python
    from convmemory import ChineseConvMemory

    model = ChineseConvMemory.from_pretrained("Purdy0228/ConvMemory-ZH-DualSpace-GTE")
    results = model.rerank("他喜欢什么演员?", memories, top_k=5)
    ```
    """

    def __init__(
        self,
        conv_model,
        scorer,
        encoder: DualSpaceTextEncoder,
        config: dict,
        *,
        device: str = "cpu",
    ):
        self.device = device
        self.encoder = encoder
        self.config = config
        self.architecture = config["architecture"]
        self.training = config.get("training", {})
        self.default_raw_weight = float(config.get("default_raw_weight", 0.0))
        self.conv_model = conv_model.to(device).eval()
        self.scorer = scorer.to(device).eval()

    @classmethod
    def from_pretrained(
        cls,
        path_or_hub_id: str | Path,
        *,
        device: str = "cpu",
        base_encoder_model: Optional[str] = None,
        tuned_encoder_model: Optional[str] = None,
        encoder_batch_size: int = 64,
        trust_remote_code: bool = True,
    ):
        """Load a Chinese ConvMemory checkpoint from local disk or Hub.

        A release checkpoint may include a `triplet_encoder/` subfolder. When it
        does, that local folder is used automatically as the tuned encoder.
        Otherwise callers can pass `tuned_encoder_model=...` to point to a local
        or Hub SentenceTransformer model.
        """

        path = resolve_checkpoint_path(path_or_hub_id)
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        encoder_cfg = config["encoder"]
        base_model = base_encoder_model or encoder_cfg["base_model"]
        tuned_model = tuned_encoder_model or encoder_cfg["tuned_model"]
        bundled_tuned = path / "triplet_encoder"
        if tuned_encoder_model is None and bundled_tuned.exists():
            tuned_model = str(bundled_tuned)
        else:
            candidate_tuned = path / str(tuned_model)
            if tuned_encoder_model is None and candidate_tuned.exists():
                tuned_model = str(candidate_tuned)

        arch = config["architecture"]
        conv_model = MixerConvMemoryEncoder(
            int(arch["embedding_dim"]),
            window_size=int(arch["window_size"]),
            kernel_size=int(arch["kernel"]),
            hidden_dim=int(arch["mixer_hidden_dim"]),
            token_mlp_dim=int(arch["mixer_token_dim"]),
            channel_mlp_dim=int(arch["mixer_channel_dim"]),
            output_mode="residual",
            output_gate_init=0.1,
            score_mode="cosine",
        )
        scorer = PairwiseCELiteScorer(
            int(arch["embedding_dim"]),
            hidden_dim=int(arch["ce_hidden_dim"]),
            extra_scalar_features=int(arch["extra_scalar_features"]),
            interaction_dim=int(arch["interaction_dim"]),
        )
        try:
            state = torch.load(path / "student.pt", map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(path / "student.pt", map_location="cpu")
        conv_model.load_state_dict(state["model_state_dict"])
        scorer.load_state_dict(state["scorer_state_dict"])
        encoder = DualSpaceTextEncoder(
            base_model,
            tuned_model,
            device=device,
            batch_size=encoder_batch_size,
            trust_remote_code=trust_remote_code,
        )
        return cls(conv_model, scorer, encoder, config, device=device)

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        return self.encoder.encode(texts)

    def rerank(
        self,
        query: str,
        memories: Iterable,
        *,
        top_k: Optional[int] = None,
        candidate_ids: Optional[Iterable[str]] = None,
        candidate_top_n: Optional[int] = None,
        raw_weight: Optional[float] = None,
        window_mode: str = "full",
    ) -> list[RerankResult]:
        memories = list(memories)
        memory_ids, memory_texts = _parse_memories(memories)
        embeddings = self.encode([query, *memory_texts])
        results = self.rerank_embeddings(
            query_embedding=embeddings[0],
            memory_embeddings=embeddings[1:],
            memory_ids=memory_ids,
            memory_texts=memory_texts,
            query=query,
            candidate_ids=candidate_ids,
            candidate_top_n=candidate_top_n,
            raw_weight=raw_weight,
            window_mode=window_mode,
        )
        return results[:top_k] if top_k is not None else results

    def rerank_embeddings(
        self,
        *,
        query_embedding,
        memory_embeddings,
        memory_ids,
        memory_texts: Optional[Iterable[str]] = None,
        query: str = "",
        candidate_ids: Optional[Iterable[str]] = None,
        candidate_indices=None,
        candidate_top_n: Optional[int] = None,
        raw_weight: Optional[float] = None,
        window_mode: str = "full",
    ) -> list[RerankResult]:
        memory_ids = [str(x) for x in memory_ids]
        memory_embeddings = normalize_rows(np.asarray(memory_embeddings, dtype=np.float32))
        query_embedding = np.asarray(query_embedding, dtype=np.float32)
        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        if memory_texts is None:
            memory_texts = ["" for _ in memory_ids]
        else:
            memory_texts = [str(x) for x in memory_texts]

        if candidate_indices is None and candidate_ids is not None:
            id_to_idx = {memory_id: i for i, memory_id in enumerate(memory_ids)}
            candidate_indices = [
                id_to_idx[str(memory_id)]
                for memory_id in candidate_ids
                if str(memory_id) in id_to_idx
            ]

        windows = sliding_windows(
            len(memory_ids),
            int(self.architecture["window_size"]),
            int(self.architecture.get("stride", 1)),
        )
        item = {
            "question_id": "query",
            "query": str(query),
            "query_embedding": query_embedding.astype(np.float32),
            "memory_embeddings": memory_embeddings.astype(np.float32),
            "memory_ids": memory_ids,
            "memories": [
                {"id": memory_id, "text": text}
                for memory_id, text in zip(memory_ids, memory_texts)
            ],
            "windows": windows,
            "window_tensor": window_tensor(memory_embeddings, windows),
            "gold_memory_ids": [],
        }
        return self.rerank_item(
            item,
            candidate_indices=candidate_indices,
            candidate_top_n=candidate_top_n,
            raw_weight=raw_weight,
            window_mode=window_mode,
        )

    def rerank_item(
        self,
        item: dict,
        *,
        candidate_indices=None,
        candidate_top_n: Optional[int] = None,
        raw_weight: Optional[float] = None,
        window_mode: str = "full",
    ) -> list[RerankResult]:
        raw_scores = cosine_scores(item["query_embedding"], item["memory_embeddings"])
        if candidate_indices is None:
            top_n = candidate_top_n or int(self.training.get("candidate_top_n", 500))
            candidate_indices = np.argsort(-raw_scores)[: min(top_n, len(raw_scores))]
        else:
            candidate_indices = np.asarray(candidate_indices, dtype=np.int64)

        scoring_item = item
        if window_mode == "candidate_local":
            windows = _candidate_local_windows(
                len(item["memory_ids"]),
                candidate_indices,
                int(self.architecture["window_size"]),
            )
            scoring_item = {
                **item,
                "windows": windows,
                "window_tensor": window_tensor(item["memory_embeddings"], windows),
            }
        elif window_mode != "full":
            raise ValueError(f"Unknown window_mode: {window_mode}")

        with torch.no_grad():
            window_logits = window_scores(self.conv_model, scoring_item, self.device)
            memory_to_windows = build_memory_to_windows(scoring_item["windows"])
            candidate_scores = self._score_candidates(
                scoring_item,
                candidate_indices,
                raw_scores,
                window_logits,
                memory_to_windows,
            )

        ranked_ids = rerank_candidates(
            raw_scores,
            candidate_indices,
            candidate_scores,
            item["memory_ids"],
            raw_weight=self.default_raw_weight if raw_weight is None else float(raw_weight),
        )
        score_by_id = {
            item["memory_ids"][int(idx)]: float(score)
            for idx, score in zip(candidate_indices, candidate_scores)
        }
        raw_by_id = {
            item["memory_ids"][idx]: float(score)
            for idx, score in enumerate(raw_scores)
        }
        text_by_id = {
            str(memory.get("id", idx)): memory.get("text")
            for idx, memory in enumerate(item.get("memories", []))
        }
        return [
            RerankResult(
                memory_id=memory_id,
                score=score_by_id.get(memory_id, raw_by_id[memory_id]),
                raw_score=raw_by_id[memory_id],
                rank=rank,
                text=text_by_id.get(memory_id),
            )
            for rank, memory_id in enumerate(ranked_ids, start=1)
        ]

    def _score_candidates(
        self,
        item,
        candidate_indices,
        raw_scores_all,
        window_logits,
        memory_to_windows,
    ) -> np.ndarray:
        features = self._candidate_features(
            item,
            candidate_indices,
            raw_scores_all,
            window_logits,
            memory_to_windows,
        )
        return self.scorer(features).detach().cpu().numpy()

    def _candidate_features(
        self,
        item,
        candidate_indices,
        raw_scores_all,
        window_logits,
        memory_to_windows,
    ):
        q_np = item["query_embedding"].astype(np.float32)
        memory_np = item["memory_embeddings"][candidate_indices].astype(np.float32)
        raw_scores = raw_scores_all[candidate_indices].astype(np.float32)
        raw_norm = normalize_scores(raw_scores)
        best_window = _best_window_scores(window_logits, memory_to_windows, candidate_indices, self.device)
        best_window_norm = (best_window - best_window.mean()) / (
            best_window.std(unbiased=False) + 1e-6
        )

        local_logits = _local_window_scores(self.conv_model, item, self.device)
        local_best = _best_window_scores(local_logits, memory_to_windows, candidate_indices, self.device)
        local_norm = (local_best - local_best.mean()) / (local_best.std(unbiased=False) + 1e-6)
        delta = best_window - local_best
        delta_norm = (delta - delta.mean()) / (delta.std(unbiased=False) + 1e-6)

        router_scores = dca_router_outputs(
            item,
            candidate_indices,
            int(self.architecture["dca_router_block_size"]),
            self.device,
        )
        router_norm = (router_scores - router_scores.mean()) / (
            router_scores.std(unbiased=False) + 1e-6
        )
        lexical = lexical_overlap_features(item, candidate_indices, self.device)

        q = torch.tensor(q_np[None, :], dtype=torch.float32, device=self.device)
        memories = torch.tensor(memory_np, dtype=torch.float32, device=self.device)
        q_batch = q.expand(memories.shape[0], -1)
        raw_rank = torch.linspace(0.0, 1.0, steps=memories.shape[0], dtype=torch.float32, device=self.device)
        position = torch.tensor(
            candidate_indices / max(1, len(item["memory_ids"]) - 1),
            dtype=torch.float32,
            device=self.device,
        )
        raw_tensor = torch.tensor(raw_norm, dtype=torch.float32, device=self.device)
        return torch.cat(
            [
                q_batch,
                memories,
                q_batch * memories,
                torch.abs(q_batch - memories),
                raw_tensor[:, None],
                best_window_norm[:, None],
                raw_rank[:, None],
                position[:, None],
                local_norm[:, None],
                delta_norm[:, None],
                router_norm[:, None],
                lexical,
            ],
            dim=-1,
        )


def _best_window_scores(window_logits, memory_to_windows, candidate_indices, device):
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


def _local_window_scores(model, item, device):
    q = item["query_embedding"]
    query = torch.tensor(q[None, :], dtype=torch.float32, device=device)
    window_batch = item["window_tensor"].to(device)
    query_batch = query.expand(window_batch.shape[0], -1)
    blocks = model(window_batch, query=query_batch)
    return (query @ blocks.T).squeeze(0)


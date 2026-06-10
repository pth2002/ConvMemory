from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import torch

from .scoring import (
    build_memory_to_windows,
    cosine_scores,
    rerank_candidates,
    score_ce_lite,
    window_scores,
)


@dataclass
class RerankConfig:
    window_size: int = 5
    stride: int = 1
    candidate_top_n: int = 500
    raw_weight: float = 0.0
    dca_router_block_size: int = 32
    lexical_features: bool = True
    window_mode: str = "full"


@dataclass
class RerankResult:
    memory_id: str
    score: float
    raw_score: float
    rank: int
    text: Optional[str] = None
    validity: Optional[dict] = None


def sliding_windows(num_items: int, window_size: int, stride: int):
    if num_items <= 0:
        return []
    if num_items <= window_size:
        return [list(range(num_items))]

    windows = []
    for start in range(0, num_items - window_size + 1, stride):
        windows.append(list(range(start, start + window_size)))
    last = list(range(num_items - window_size, num_items))
    if windows[-1] != last:
        windows.append(last)
    return windows


def normalize_rows(matrix):
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8
    return matrix / norms


def window_tensor(memory_embeddings, windows):
    if not windows:
        return torch.zeros((0, 0, memory_embeddings.shape[1]), dtype=torch.float32)
    if all(len(window) == len(windows[0]) for window in windows):
        indices = np.asarray(windows, dtype=np.int64)
        return torch.as_tensor(memory_embeddings[indices], dtype=torch.float32)
    max_len = max(len(window) for window in windows)
    batch = np.zeros((len(windows), max_len, memory_embeddings.shape[1]), dtype=np.float32)
    for i, window in enumerate(windows):
        batch[i, : len(window)] = memory_embeddings[window]
    return torch.tensor(batch, dtype=torch.float32)


def candidate_local_windows(num_items, candidate_indices, window_size):
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


class ConvMemoryReranker:
    """Plug-in reranker over precomputed memory embeddings.

    The class deliberately does not own text embedding. In production, users can
    bring any retriever or embedding model, pass the raw top-k candidates here,
    and get a ConvMemory-enhanced ordering back.
    """

    def __init__(self, conv_model, scorer, config=None, device="cpu"):
        self.conv_model = conv_model
        self.scorer = scorer
        self.config = config or RerankConfig()
        self.device = device

    def make_item(
        self,
        query_embedding,
        memory_embeddings,
        memory_ids,
        memory_texts: Optional[Iterable[str]] = None,
        query: str = "",
    ):
        memory_ids = [str(x) for x in memory_ids]
        memory_embeddings = normalize_rows(memory_embeddings)
        query_embedding = np.asarray(query_embedding, dtype=np.float32)
        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        if memory_texts is None:
            memory_texts = ["" for _ in memory_ids]
        memories = [
            {"id": memory_id, "text": text}
            for memory_id, text in zip(memory_ids, memory_texts)
        ]
        windows = sliding_windows(
            len(memory_ids),
            self.config.window_size,
            self.config.stride,
        )

        return {
            "question_id": "query",
            "question_type": "unknown",
            "query": query,
            "query_embedding": query_embedding.astype(np.float32),
            "memory_embeddings": memory_embeddings.astype(np.float32),
            "memory_ids": memory_ids,
            "memories": memories,
            "windows": windows,
            "window_tensor": None,
            "gold_memory_ids": [],
        }

    def rerank_embeddings(
        self,
        query_embedding,
        memory_embeddings,
        memory_ids,
        memory_texts: Optional[Iterable[str]] = None,
        query: str = "",
        candidate_indices=None,
        candidate_top_n: Optional[int] = None,
        raw_weight: Optional[float] = None,
        window_mode: Optional[str] = None,
    ):
        item = self.make_item(
            query_embedding=query_embedding,
            memory_embeddings=memory_embeddings,
            memory_ids=memory_ids,
            memory_texts=memory_texts,
            query=query,
        )
        return self.rerank_item(
            item,
            candidate_indices=candidate_indices,
            candidate_top_n=candidate_top_n,
            raw_weight=raw_weight,
            window_mode=window_mode,
        )

    def rerank_item(
        self,
        item,
        candidate_indices=None,
        candidate_top_n: Optional[int] = None,
        raw_weight: Optional[float] = None,
        window_mode: Optional[str] = None,
    ):
        raw_scores = cosine_scores(item["query_embedding"], item["memory_embeddings"])
        if candidate_indices is None:
            top_n = candidate_top_n or self.config.candidate_top_n
            candidate_indices = np.argsort(-raw_scores)[: min(top_n, len(raw_scores))]
        else:
            candidate_indices = np.asarray(candidate_indices, dtype=np.int64)

        scoring_item = item
        selected_window_mode = window_mode or self.config.window_mode
        if selected_window_mode == "candidate_local":
            local_windows = candidate_local_windows(
                len(item["memory_ids"]),
                candidate_indices,
                self.config.window_size,
            )
            scoring_item = {
                **item,
                "windows": local_windows,
                "window_tensor": window_tensor(item["memory_embeddings"], local_windows),
            }
        elif selected_window_mode != "full":
            raise ValueError(f"Unknown window_mode: {selected_window_mode}")
        elif scoring_item.get("window_tensor") is None:
            scoring_item = {
                **item,
                "window_tensor": window_tensor(item["memory_embeddings"], item["windows"]),
            }

        with torch.no_grad():
            conv_tensor = window_scores(self.conv_model, scoring_item, self.device)
        memory_to_windows = build_memory_to_windows(scoring_item["windows"])
        _, _, ce_lite_scores = score_ce_lite(
            self.conv_model,
            self.scorer,
            scoring_item,
            candidate_indices,
            self.device,
            raw_scores_all=raw_scores,
            window_logits=conv_tensor,
            memory_to_windows=memory_to_windows,
            dca_router_block_size=self.config.dca_router_block_size,
            lexical_features=self.config.lexical_features,
        )
        ranked_ids = rerank_candidates(
            raw_scores,
            candidate_indices,
            ce_lite_scores,
            item["memory_ids"],
            raw_weight=self.config.raw_weight if raw_weight is None else raw_weight,
        )
        score_by_id = {
            item["memory_ids"][int(idx)]: float(score)
            for idx, score in zip(candidate_indices, ce_lite_scores)
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

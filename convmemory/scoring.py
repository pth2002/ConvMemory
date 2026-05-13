import re
from functools import lru_cache

import numpy as np
import torch


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


def build_memory_to_windows(windows):
    memory_to_windows = {}
    for window_idx, window in enumerate(windows):
        for memory_idx in window:
            memory_to_windows.setdefault(int(memory_idx), []).append(window_idx)
    return memory_to_windows


def best_window_scores_for_candidates(window_logits, memory_to_windows, candidate_indices, device):
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


def dca_router_outputs(item, candidate_indices, block_size, device):
    memory_embeddings = item["memory_embeddings"].astype(np.float32)
    q_np = item["query_embedding"].astype(np.float32)
    num_memories = memory_embeddings.shape[0]
    if num_memories == 0:
        return torch.zeros(len(candidate_indices), dtype=torch.float32, device=device)

    block_ids = np.arange(num_memories) // max(1, block_size)
    num_blocks = int(block_ids.max()) + 1
    block_embeddings = np.zeros((num_blocks, memory_embeddings.shape[1]), dtype=np.float32)
    for block_idx in range(num_blocks):
        block_embeddings[block_idx] = memory_embeddings[block_ids == block_idx].mean(axis=0)

    block_embeddings = block_embeddings / (
        np.linalg.norm(block_embeddings, axis=1, keepdims=True) + 1e-8
    )
    q_norm = q_np / (np.linalg.norm(q_np) + 1e-8)
    block_scores = block_embeddings @ q_norm
    candidate_blocks = block_ids[candidate_indices]
    candidate_scores = block_scores[candidate_blocks].astype(np.float32)
    return torch.tensor(candidate_scores, dtype=torch.float32, device=device)


@lru_cache(maxsize=250_000)
def lexical_token_tuple(text):
    text = str(text)
    return tuple(t for t in TOKEN_RE.findall(text.lower()) if len(t) > 1 and t not in STOPWORDS)


@lru_cache(maxsize=250_000)
def lexical_signature(text):
    tokens = lexical_token_tuple(str(text))
    return frozenset(tokens), frozenset(zip(tokens, tokens[1:]))


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
    dca_router_block_size=0,
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
    best_window = best_window_scores_for_candidates(
        window_logits,
        memory_to_windows,
        candidate_indices,
        device,
    )
    best_window_norm = (best_window - best_window.mean()) / (best_window.std(unbiased=False) + 1e-6)

    extra_features = []
    if dca_router_block_size > 0:
        router_scores = dca_router_outputs(item, candidate_indices, dca_router_block_size, device)
        router_norm = (router_scores - router_scores.mean()) / (
            router_scores.std(unbiased=False) + 1e-6
        )
        extra_features.append(router_norm[:, None])
    if lexical_features:
        extra_features.append(lexical_overlap_features(item, candidate_indices, device))

    q = torch.tensor(q_np[None, :], dtype=torch.float32, device=device)
    memories = torch.tensor(memory_np, dtype=torch.float32, device=device)
    q_batch = q.expand(memories.shape[0], -1)
    raw_rank = torch.linspace(0.0, 1.0, steps=memories.shape[0], dtype=torch.float32, device=device)
    position = torch.tensor(
        candidate_indices / max(1, len(item["memory_ids"]) - 1),
        dtype=torch.float32,
        device=device,
    )
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


def score_ce_lite(
    model,
    scorer,
    item,
    candidate_indices,
    device,
    raw_scores_all=None,
    window_logits=None,
    memory_to_windows=None,
    dca_router_block_size=0,
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
            dca_router_block_size=dca_router_block_size,
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

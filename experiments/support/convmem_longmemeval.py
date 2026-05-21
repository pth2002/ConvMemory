import argparse
import csv
import hashlib
import json
import random
import sqlite3
import time
from pathlib import Path
from os.path import expanduser

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer


def l2_normalize(x, eps=1e-8):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def cosine_scores(query, matrix):
    return matrix @ query


def recall_at_k(ranked_ids, gold_ids, k):
    return len(set(ranked_ids[:k]) & set(gold_ids)) / max(1, len(gold_ids))


def hit_at_k(ranked_ids, gold_ids, k):
    return float(bool(set(ranked_ids[:k]) & set(gold_ids)))


def mrr(ranked_ids, gold_ids):
    gold = set(gold_ids)
    for rank, item_id in enumerate(ranked_ids, start=1):
        if item_id in gold:
            return 1.0 / rank
    return 0.0


def session_to_text(session):
    parts = []
    for turn in session:
        role = turn.get("role", "speaker")
        content = turn.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def load_longmemeval(path, limit=None, skip_abstention=True):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    examples = []
    for item in data:
        gold_ids = item.get("answer_session_ids", [])
        if skip_abstention and not gold_ids:
            continue

        sessions = item["haystack_sessions"]
        session_ids = item["haystack_session_ids"]
        memories = [
            {
                "id": str(session_id),
                "text": session_to_text(session),
            }
            for session_id, session in zip(session_ids, sessions)
        ]
        examples.append(
            {
                "question_id": item["question_id"],
                "question_type": item.get("question_type", "unknown"),
                "query": item["question"],
                "answer": item.get("answer", ""),
                "memories": memories,
                "gold_memory_ids": [str(x) for x in gold_ids],
            }
        )
        if limit and len(examples) >= limit:
            break
    return examples


def make_windows(num_items, window_size=5, stride=2):
    if window_size <= 1:
        return [[i] for i in range(num_items)]
    windows = []
    for start in range(0, num_items, stride):
        end = min(start + window_size, num_items)
        if end - start < 2:
            break
        windows.append(list(range(start, end)))
        if end == num_items:
            break
    return windows


class TfidfTextEncoder:
    def __init__(self, max_features=4096):
        self.cache = {}
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=max_features,
            min_df=1,
            stop_words="english",
        )

    def fit(self, texts):
        self.vectorizer.fit(texts)

    def transform(self, texts):
        missing = [text for text in texts if text not in self.cache]
        if missing:
            x = self.vectorizer.transform(missing).astype(np.float32).toarray()
            x = l2_normalize(x)
            for text, embedding in zip(missing, x):
                self.cache[text] = embedding
        return np.array([self.cache[text] for text in texts], dtype=np.float32)


class SQLiteEmbeddingCache:
    def __init__(self, path, model_name):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.conn = sqlite3.connect(str(self.path))
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                model_name TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                text TEXT NOT NULL,
                dim INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                PRIMARY KEY (model_name, text_hash)
            )
            """
        )
        self.conn.commit()

    def key(self, text):
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def get(self, text):
        row = self.conn.execute(
            """
            SELECT text, dim, embedding
            FROM embeddings
            WHERE model_name = ? AND text_hash = ?
            """,
            (self.model_name, self.key(text)),
        ).fetchone()
        if not row:
            return None
        stored_text, dim, blob = row
        if stored_text != text:
            return None
        return np.frombuffer(blob, dtype=np.float32).copy().reshape(dim)

    def put_many(self, texts, embeddings):
        rows = []
        for text, embedding in zip(texts, embeddings):
            value = np.asarray(embedding, dtype=np.float32)
            rows.append(
                (
                    self.model_name,
                    self.key(text),
                    text,
                    int(value.shape[0]),
                    value.tobytes(),
                )
            )
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO embeddings
            (model_name, text_hash, text, dim, embedding)
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )
        self.conn.commit()


class SentenceTransformerTextEncoder:
    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
        batch_size=32,
        cache_path=None,
        cache_key=None,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.cache = {}
        self.disk_cache = SQLiteEmbeddingCache(cache_path, cache_key or model_name) if cache_path else None
        self.model = SentenceTransformer(resolve_local_model_path(model_name), device=device)

    def fit(self, texts):
        return None

    def transform(self, texts):
        unique_texts = list(dict.fromkeys(texts))
        missing = []
        for text in unique_texts:
            if text in self.cache:
                continue
            cached = self.disk_cache.get(text) if self.disk_cache else None
            if cached is not None:
                self.cache[text] = cached
            else:
                missing.append(text)
        if missing:
            x = self.model.encode(
                missing,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype(np.float32)
            for text, embedding in zip(missing, x):
                self.cache[text] = embedding
            if self.disk_cache:
                self.disk_cache.put_many(missing, x)
        return np.array([self.cache[text] for text in texts], dtype=np.float32)


def resolve_local_model_path(model_name):
    if Path(model_name).exists():
        return model_name

    cache_name = "models--" + model_name.replace("/", "--")
    snapshots = Path(expanduser("~")) / ".cache" / "huggingface" / "hub" / cache_name / "snapshots"
    if snapshots.exists():
        candidates = sorted([p for p in snapshots.iterdir() if p.is_dir()], reverse=True)
        for candidate in candidates:
            has_sentence_transformer_files = (candidate / "modules.json").exists()
            has_hf_transformer_files = (
                (candidate / "tokenizer_config.json").exists()
                or (candidate / "tokenizer.json").exists()
                or (candidate / "vocab.txt").exists()
            )
            if (candidate / "config.json").exists() and (
                has_sentence_transformer_files or has_hf_transformer_files
            ):
                return str(candidate)
    return model_name


class ConvMemoryEncoder(torch.nn.Module):
    def __init__(
        self,
        dim,
        kernel_size=3,
        layers=1,
        pooling="mean",
        multi_scale_kernels=None,
        type_vocab_size=0,
        projection="none",
    ):
        super().__init__()
        self.pooling = pooling
        self.projection = projection
        self.gate = torch.nn.Parameter(torch.tensor(0.1))
        self.multi_scale_kernels = multi_scale_kernels or []
        self.type_embedding = None
        if type_vocab_size:
            self.type_embedding = torch.nn.Embedding(type_vocab_size, dim)
        if self.multi_scale_kernels:
            self.encoders = torch.nn.ModuleList(
                [self._make_encoder(dim, kernel, layers) for kernel in self.multi_scale_kernels]
            )
        else:
            self.encoder = self._make_encoder(dim, kernel_size, layers)
        if projection == "linear":
            self.projection_head = torch.nn.Linear(dim, dim)
        elif projection == "mlp":
            self.projection_head = torch.nn.Sequential(
                torch.nn.Linear(dim, dim),
                torch.nn.ReLU(),
                torch.nn.Linear(dim, dim),
            )
        else:
            self.projection_head = None

    def _make_encoder(self, dim, kernel_size, layers):
        blocks = []
        for _ in range(layers):
            blocks.append(
                torch.nn.Conv1d(
                    in_channels=dim,
                    out_channels=dim,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
            blocks.append(torch.nn.ReLU())
        return torch.nn.Sequential(*blocks)

    def _pool(self, h, query=None):
        if self.pooling == "query_attention" and query is not None:
            h_time = h.transpose(1, 2)
            query = torch.nn.functional.normalize(query, dim=-1)
            h_norm = torch.nn.functional.normalize(h_time, dim=-1)
            scores = (h_norm * query[:, None, :]).sum(dim=-1)
            weights = torch.softmax(scores, dim=-1)
            return (h_time * weights[:, :, None]).sum(dim=1)
        return h.mean(dim=-1)

    def forward(self, x, query=None, type_ids=None):
        if self.type_embedding is not None and type_ids is not None:
            x = x + self.type_embedding(type_ids)
        x = x.transpose(1, 2)
        if self.multi_scale_kernels:
            pooled = []
            for encoder in self.encoders:
                h = x + self.gate * encoder(x)
                pooled.append(self._pool(h, query=query))
            scale_stack = torch.stack(pooled, dim=1)
            if query is not None:
                query = torch.nn.functional.normalize(query, dim=-1)
                scale_norm = torch.nn.functional.normalize(scale_stack, dim=-1)
                scale_scores = (scale_norm * query[:, None, :]).sum(dim=-1)
                scale_weights = torch.softmax(scale_scores, dim=-1)
                h = (scale_stack * scale_weights[:, :, None]).sum(dim=1)
            else:
                h = scale_stack.mean(dim=1)
        else:
            h = x + self.gate * self.encoder(x)
            h = self._pool(h, query=query)
        if self.projection_head is not None:
            h = h + self.projection_head(h)
        return torch.nn.functional.normalize(h, dim=-1)


class MixerConvMemoryEncoder(torch.nn.Module):
    def __init__(
        self,
        dim,
        window_size=5,
        kernel_size=3,
        hidden_dim=256,
        token_mlp_dim=32,
        channel_mlp_dim=512,
        type_vocab_size=0,
        output_mode="residual",
        output_gate_init=0.1,
        score_mode="cosine",
        score_gate_init=0.1,
    ):
        super().__init__()
        self.pooling = "query_attention"
        self.window_size = window_size
        self.output_mode = output_mode
        self.score_mode = score_mode
        self.type_embedding = None
        if type_vocab_size:
            self.type_embedding = torch.nn.Embedding(type_vocab_size, dim)

        self.input_proj = torch.nn.Sequential(
            torch.nn.Linear(dim * 3, hidden_dim),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden_dim),
        )
        self.conv_norm = torch.nn.LayerNorm(hidden_dim)
        self.depthwise_conv = torch.nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=hidden_dim,
        )
        self.pointwise = torch.nn.Linear(hidden_dim, hidden_dim)
        self.conv_gate = torch.nn.Parameter(torch.tensor(0.1))

        self.token_norm = torch.nn.LayerNorm(window_size)
        self.token_mlp = torch.nn.Sequential(
            torch.nn.Linear(window_size, token_mlp_dim),
            torch.nn.GELU(),
            torch.nn.Linear(token_mlp_dim, window_size),
        )
        self.token_gate = torch.nn.Parameter(torch.tensor(0.1))

        self.channel_norm = torch.nn.LayerNorm(hidden_dim)
        self.channel_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, channel_mlp_dim),
            torch.nn.GELU(),
            torch.nn.Linear(channel_mlp_dim, hidden_dim),
        )
        self.channel_gate = torch.nn.Parameter(torch.tensor(0.1))

        self.query_proj = torch.nn.Linear(dim, hidden_dim)
        self.attn_x = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_q = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_v = torch.nn.Linear(hidden_dim, 1, bias=False)
        self.output_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 4, dim),
            torch.nn.LayerNorm(dim),
        )
        self.output_gate = torch.nn.Parameter(torch.tensor(float(output_gate_init)))
        self.score_head = torch.nn.Sequential(
            torch.nn.Linear(dim * 4, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )
        self.score_gate = torch.nn.Parameter(torch.tensor(float(score_gate_init)))

    def _token_mix(self, h):
        length = h.shape[1]
        if length < self.window_size:
            pad = torch.zeros(
                h.shape[0],
                self.window_size - length,
                h.shape[2],
                dtype=h.dtype,
                device=h.device,
            )
            h_for_mix = torch.cat([h, pad], dim=1)
        else:
            h_for_mix = h[:, : self.window_size]

        mixed = h_for_mix.transpose(1, 2)
        mixed = self.token_mlp(self.token_norm(mixed)).transpose(1, 2)
        return mixed[:, :length]

    def forward(self, x, query=None, type_ids=None):
        base_x = x
        if self.type_embedding is not None and type_ids is not None:
            x = x + self.type_embedding(type_ids)
            base_x = x
        if query is None:
            query = x.mean(dim=1)

        query_norm = torch.nn.functional.normalize(query, dim=-1)
        base_norm = torch.nn.functional.normalize(base_x, dim=-1)
        base_scores = (base_norm * query_norm[:, None, :]).sum(dim=-1)
        base_weights = torch.softmax(base_scores, dim=1)
        base = (base_x * base_weights[:, :, None]).sum(dim=1)

        query_per_turn = query[:, None, :].expand(-1, x.shape[1], -1)
        features = torch.cat([x, x * query_per_turn, torch.abs(x - query_per_turn)], dim=-1)
        h = self.input_proj(features)

        conv_in = self.conv_norm(h).transpose(1, 2)
        conv_out = self.depthwise_conv(conv_in).transpose(1, 2)
        h = h + self.conv_gate * self.pointwise(torch.nn.functional.gelu(conv_out))

        h = h + self.token_gate * self._token_mix(h)
        h = h + self.channel_gate * self.channel_mlp(self.channel_norm(h))

        qh = self.query_proj(query)
        attn = self.attn_v(torch.tanh(self.attn_x(h) + self.attn_q(qh)[:, None, :])).squeeze(-1)
        weights = torch.softmax(attn, dim=1)
        pooled = (h * weights[:, :, None]).sum(dim=1)

        out = self.output_head(
            torch.cat([pooled, qh, pooled * qh, torch.abs(pooled - qh)], dim=-1)
        )
        if self.output_mode == "residual":
            out = base + self.output_gate * out
        return torch.nn.functional.normalize(out, dim=-1)

    def score_windows(self, x, query=None, type_ids=None):
        vectors = self.forward(x, query=query, type_ids=type_ids)
        if query is None:
            query = x.mean(dim=1)
        query_norm = torch.nn.functional.normalize(query, dim=-1)
        cosine = (vectors * query_norm).sum(dim=-1)
        if self.score_mode == "cosine":
            return cosine

        features = torch.cat(
            [vectors, query_norm, vectors * query_norm, torch.abs(vectors - query_norm)],
            dim=-1,
        )
        correction = torch.tanh(self.score_head(features).squeeze(-1))
        return cosine + self.score_gate * correction


class DCAConvMemoryEncoder(MixerConvMemoryEncoder):
    def __init__(
        self,
        dim,
        window_size=5,
        kernel_size=3,
        hidden_dim=256,
        token_mlp_dim=32,
        channel_mlp_dim=512,
        chunk_hidden_dim=128,
        type_vocab_size=0,
        output_mode="residual",
        output_gate_init=0.1,
        dca_gate_init=0.1,
        dca_mode="correction",
        dca_gate_mode="fixed",
    ):
        super().__init__(
            dim,
            window_size=window_size,
            kernel_size=kernel_size,
            hidden_dim=hidden_dim,
            token_mlp_dim=token_mlp_dim,
            channel_mlp_dim=channel_mlp_dim,
            type_vocab_size=type_vocab_size,
            output_mode=output_mode,
            output_gate_init=output_gate_init,
            score_mode="dca",
            score_gate_init=dca_gate_init,
        )
        self.score_mode = "dca"
        self.dca_mode = dca_mode
        self.dca_gate_mode = dca_gate_mode
        self.chunk_norm = torch.nn.LayerNorm(dim)
        self.chunk_query = torch.nn.Linear(dim, chunk_hidden_dim, bias=False)
        self.chunk_key = torch.nn.Linear(dim, chunk_hidden_dim, bias=False)
        self.chunk_value = torch.nn.Linear(dim, dim, bias=False)
        self.chunk_out = torch.nn.Linear(dim, dim, bias=False)
        self.chunk_bridge = torch.nn.Conv1d(
            dim,
            dim,
            kernel_size=3,
            padding=1,
            groups=dim,
        )
        self.bridge_gate = torch.nn.Parameter(torch.tensor(float(dca_gate_init)))
        self.global_gate = torch.nn.Parameter(torch.tensor(float(dca_gate_init)))
        self.attention_gate = torch.nn.Parameter(torch.tensor(float(dca_gate_init)))
        self.dca_score_head = torch.nn.Sequential(
            torch.nn.Linear(dim * 4, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def score_windows(self, x, query=None, type_ids=None):
        vectors = self.forward(x, query=query, type_ids=type_ids)
        if query is None:
            query = x.mean(dim=1)

        query_norm = torch.nn.functional.normalize(query, dim=-1)
        local_cosine = (vectors * query_norm).sum(dim=-1)
        if vectors.shape[0] <= 1:
            return local_cosine

        chunk_features = self.chunk_norm(vectors)
        qh = torch.nn.functional.normalize(self.chunk_query(query_norm), dim=-1)
        kh = torch.nn.functional.normalize(self.chunk_key(chunk_features), dim=-1)
        attention_logits = (qh * kh).sum(dim=-1)

        if self.dca_mode == "prior":
            prior = attention_logits - attention_logits.mean()
            prior = prior / (prior.std(unbiased=False) + 1e-6)
            gate = self.score_gate
            if self.dca_gate_mode == "confidence":
                attention = torch.softmax(attention_logits, dim=0)
                entropy = -(attention * torch.log(attention + 1e-8)).sum()
                max_entropy = torch.log(torch.tensor(float(attention.numel()), device=attention.device))
                confidence = 1.0 - entropy / (max_entropy + 1e-8)
                gate = gate * (0.25 + confidence.clamp(0.0, 1.0))
            return local_cosine + gate * prior

        bridge_delta = self.chunk_bridge(vectors[None, :, :].transpose(1, 2))
        bridge_delta = bridge_delta.transpose(1, 2).squeeze(0)
        bridged = vectors + self.bridge_gate * bridge_delta
        bridged = torch.nn.functional.normalize(bridged, dim=-1)

        chunk_features = self.chunk_norm(bridged)
        kh = torch.nn.functional.normalize(self.chunk_key(chunk_features), dim=-1)
        attention_logits = (qh * kh).sum(dim=-1)
        attention = torch.softmax(attention_logits, dim=0)

        global_context = (attention[:, None] * self.chunk_value(bridged)).sum(dim=0, keepdim=True)
        global_context = global_context.expand_as(vectors)
        fused = vectors + self.global_gate * self.chunk_out(global_context)
        fused = torch.nn.functional.normalize(fused, dim=-1)
        dca_cosine = (fused * query_norm).sum(dim=-1)

        features = torch.cat(
            [fused, query_norm, fused * query_norm, torch.abs(fused - query_norm)],
            dim=-1,
        )
        correction = torch.tanh(self.dca_score_head(features).squeeze(-1))
        centered_attention = attention - attention.mean()
        return (
            local_cosine
            + self.score_gate * (dca_cosine - local_cosine)
            + self.score_gate * correction
            + self.attention_gate * centered_attention
        )


def window_tensor(memory_embeddings, windows):
    max_len = max(len(w) for w in windows)
    dim = memory_embeddings.shape[1]
    batch = np.zeros((len(windows), max_len, dim), dtype=np.float32)
    for i, window in enumerate(windows):
        values = memory_embeddings[window]
        batch[i, : len(window)] = values
    return torch.tensor(batch, dtype=torch.float32)


def prepare_encoded_example(example, encoder, window_size, stride):
    memory_ids = [m["id"] for m in example["memories"]]
    memory_texts = [m["text"] for m in example["memories"]]
    memory_embeddings = encoder.transform(memory_texts)
    query_embedding = encoder.transform([example["query"]])[0]
    windows = make_windows(len(memory_ids), window_size, stride)
    if not windows:
        return None
    return {
        **example,
        "memory_ids": memory_ids,
        "memory_embeddings": memory_embeddings,
        "query_embedding": query_embedding,
        "windows": windows,
        "window_tensor": window_tensor(memory_embeddings, windows),
    }


def add_distractors(examples, target_sessions, seed=7):
    if not target_sessions:
        return examples

    rng = random.Random(seed)
    all_memories = []
    for item in examples:
        for memory in item["memories"]:
            all_memories.append(memory)

    expanded = []
    for item in examples:
        existing_ids = {m["id"] for m in item["memories"]}
        new_memories = list(item["memories"])
        candidates = [m for m in all_memories if m["id"] not in existing_ids]
        rng.shuffle(candidates)

        for memory in candidates:
            if len(new_memories) >= target_sessions:
                break
            copied = {
                "id": f"distractor::{memory['id']}",
                "text": memory["text"],
            }
            new_memories.append(copied)

        expanded.append({**item, "memories": new_memories})
    return expanded


def positive_window_index(encoded):
    positives = positive_window_indices(encoded)
    return positives[0] if positives else None


def positive_window_indices(encoded):
    if "_positive_window_indices" in encoded:
        return encoded["_positive_window_indices"]
    memory_to_idx = {m: i for i, m in enumerate(encoded["memory_ids"])}
    gold_indices = {memory_to_idx[g] for g in encoded["gold_memory_ids"] if g in memory_to_idx}
    positives = []
    for i, window in enumerate(encoded["windows"]):
        if gold_indices & set(window):
            positives.append(i)
    return positives


def hard_negative_window_indices(encoded, max_negatives=8):
    cache_key = f"_hard_negative_window_indices_{max_negatives}"
    if cache_key in encoded:
        return encoded[cache_key]
    positives = set(positive_window_indices(encoded))
    if not positives:
        return []

    q = encoded["query_embedding"]
    memory_embeddings = encoded["memory_embeddings"]
    windows = encoded["windows"]
    memory_to_idx = {m: i for i, m in enumerate(encoded["memory_ids"])}
    gold_indices = {memory_to_idx[g] for g in encoded["gold_memory_ids"] if g in memory_to_idx}

    scores = []
    for i, window in enumerate(windows):
        if i in positives:
            continue
        if gold_indices & set(window):
            continue
        window_embedding = memory_embeddings[window].mean(axis=0, keepdims=True)
        window_embedding = l2_normalize(window_embedding)[0]
        score = float(window_embedding @ q)
        scores.append((score, i))

    scores.sort(reverse=True)
    negatives = [i for _, i in scores[:max_negatives]]
    encoded[cache_key] = negatives
    return negatives


def cache_training_windows(encoded_items, hard_negatives=0):
    for item in encoded_items:
        item["_positive_window_indices"] = positive_window_indices({k: v for k, v in item.items() if k != "_positive_window_indices"})
        if hard_negatives > 0:
            hard_negative_window_indices(item, max_negatives=hard_negatives)
    return encoded_items


def train_conv(model, encoded_train, epochs=3, lr=3e-3, device="cpu", hard_negatives=0):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    encoded_train = cache_training_windows(encoded_train, hard_negatives=hard_negatives)
    usable = [x for x in encoded_train if positive_window_indices(x)]

    model.train()
    for _ in range(epochs):
        random.shuffle(usable)
        for item in usable:
            positives = positive_window_indices(item)
            if hard_negatives > 0:
                negatives = hard_negative_window_indices(item, max_negatives=hard_negatives)
                selected = positives + negatives
                positive_positions = list(range(len(positives)))
            else:
                selected = list(range(len(item["windows"])))
                positive_positions = positives

            query = torch.tensor(item["query_embedding"][None, :], dtype=torch.float32, device=device)
            window_batch = item["window_tensor"]
            if hard_negatives > 0:
                window_batch = window_batch[selected]
            if getattr(model, "pooling", "mean") == "query_attention":
                query_batch = query.expand(window_batch.shape[0], -1)
                kwargs = {}
                if "window_type_tensor" in item:
                    kwargs["type_ids"] = item["window_type_tensor"].to(device)
                    if hard_negatives > 0:
                        kwargs["type_ids"] = kwargs["type_ids"][selected]
                blocks = model(window_batch.to(device), query=query_batch, **kwargs)
            else:
                kwargs = {}
                if "window_type_tensor" in item:
                    kwargs["type_ids"] = item["window_type_tensor"].to(device)
                    if hard_negatives > 0:
                        kwargs["type_ids"] = kwargs["type_ids"][selected]
                blocks = model(window_batch.to(device), **kwargs)
            logits = query @ blocks.T / 0.08
            logits = logits.squeeze(0)
            positive_positions = torch.tensor(positive_positions, dtype=torch.long, device=device)
            positive_logits = logits.index_select(0, positive_positions)
            loss = torch.logsumexp(logits, dim=0) - torch.logsumexp(positive_logits, dim=0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def expand_window_ranking(block_scores, windows, memory_ids):
    ranked_windows = np.argsort(-block_scores)
    ranked = []
    seen = set()
    for window_idx in ranked_windows:
        for memory_idx in windows[window_idx]:
            memory_id = memory_ids[memory_idx]
            if memory_id not in seen:
                ranked.append(memory_id)
                seen.add(memory_id)
    return ranked


def hybrid_ranking(raw_scores, block_scores, windows, memory_ids, raw_weight=0.55):
    scores = {memory_id: raw_weight * raw_scores[i] for i, memory_id in enumerate(memory_ids)}
    for block_idx, block_score in enumerate(block_scores):
        share = (1.0 - raw_weight) * block_score
        for memory_idx in windows[block_idx]:
            memory_id = memory_ids[memory_idx]
            scores[memory_id] = scores.get(memory_id, 0.0) + share
    return [m for m, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def normalize_vector(scores, mode):
    scores = np.asarray(scores, dtype=np.float32)
    if mode == "none":
        return scores
    if mode == "zscore":
        std = float(scores.std())
        if std < 1e-8:
            return scores - float(scores.mean())
        return (scores - float(scores.mean())) / std
    if mode == "minmax":
        lo = float(scores.min())
        hi = float(scores.max())
        if hi - lo < 1e-8:
            return scores - lo
        return (scores - lo) / (hi - lo)
    raise ValueError(f"Unknown score normalization mode: {mode}")


def convmem_rerank(
    raw_scores,
    block_scores,
    windows,
    memory_ids,
    raw_top_n=100,
    raw_weight=0.8,
    score_norm="none",
):
    raw_scores = normalize_vector(raw_scores, score_norm)
    block_scores = normalize_vector(block_scores, score_norm)
    raw_order = np.argsort(-raw_scores)
    candidate_indices = raw_order[: min(raw_top_n, len(raw_order))]
    candidate_set = set(int(i) for i in candidate_indices)

    best_window_score = {int(i): 0.0 for i in candidate_indices}
    for block_idx, block_score in enumerate(block_scores):
        for memory_idx in windows[block_idx]:
            if memory_idx in candidate_set:
                best_window_score[memory_idx] = max(best_window_score[memory_idx], float(block_score))

    rerank_scores = {}
    for memory_idx in candidate_indices:
        memory_idx = int(memory_idx)
        rerank_scores[memory_ids[memory_idx]] = (
            raw_weight * float(raw_scores[memory_idx])
            + (1.0 - raw_weight) * best_window_score[memory_idx]
        )

    reranked = [m for m, _ in sorted(rerank_scores.items(), key=lambda x: x[1], reverse=True)]
    remaining = [memory_ids[int(i)] for i in raw_order if memory_ids[int(i)] not in rerank_scores]
    return reranked + remaining


def adaptive_raw_weight(item, mode):
    if mode == "oracle":
        question_type = item.get("question_type", "")
        if question_type in {"knowledge-update", "multi-session", "temporal-reasoning"}:
            return 0.80
        return 0.95

    query = item.get("query", "").lower()
    temporal_terms = [
        "before",
        "after",
        "later",
        "previous",
        "previously",
        "last time",
        "changed",
        "update",
        "updated",
        "new",
        "recent",
        "again",
        "now",
        "then",
    ]
    multi_terms = [
        "both",
        "across",
        "between",
        "compare",
        "relationship",
        "same",
        "different",
        "mentioned",
        "remember",
    ]
    if any(term in query for term in temporal_terms):
        return 0.80
    if any(term in query for term in multi_terms):
        return 0.85
    return 0.95


def evaluate_item(item, model, raw_weights, rerank_top_ns, adaptive_rerank_top_ns, device="cpu", top_k=5):
    q = item["query_embedding"]
    memory_ids = item["memory_ids"]
    memory_embeddings = item["memory_embeddings"]
    windows = item["windows"]

    raw_scores = cosine_scores(q, memory_embeddings)
    mean_blocks = l2_normalize(np.array([memory_embeddings[w].mean(axis=0) for w in windows]))

    with torch.no_grad():
        if getattr(model, "pooling", "mean") == "query_attention":
            query_tensor = torch.tensor(q[None, :], dtype=torch.float32, device=device)
            query_batch = query_tensor.expand(item["window_tensor"].shape[0], -1)
            window_batch = item["window_tensor"].to(device)
            conv_blocks_tensor = model(window_batch, query=query_batch)
            conv_blocks = conv_blocks_tensor.cpu().numpy()
            if hasattr(model, "score_windows") and getattr(model, "score_mode", "cosine") != "cosine":
                conv_scores = model.score_windows(window_batch, query=query_batch).cpu().numpy()
            else:
                conv_scores = cosine_scores(q, conv_blocks)
        else:
            conv_blocks = model(item["window_tensor"].to(device)).cpu().numpy()
            conv_scores = cosine_scores(q, conv_blocks)
    mean_scores = cosine_scores(q, mean_blocks)

    rankings = {
        "raw_session": [memory_ids[i] for i in np.argsort(-raw_scores)],
        "mean_window": expand_window_ranking(mean_scores, windows, memory_ids),
        "conv1d_window": expand_window_ranking(conv_scores, windows, memory_ids),
    }
    for weight in raw_weights:
        if len(raw_weights) == 1:
            method = "hybrid_raw_conv"
        else:
            method = f"hybrid_raw_conv_rw{weight:g}"
        rankings[method] = hybrid_ranking(raw_scores, conv_scores, windows, memory_ids, weight)

    for top_n, score_norm in rerank_top_ns:
        for weight in raw_weights:
            suffix = "" if score_norm == "none" else f"_{score_norm}"
            method = f"convmem_rerank_top{top_n}_rw{weight:g}{suffix}"
            rankings[method] = convmem_rerank(
                raw_scores,
                conv_scores,
                windows,
                memory_ids,
                raw_top_n=top_n,
                raw_weight=weight,
                score_norm=score_norm,
            )

    for top_n in adaptive_rerank_top_ns:
        for mode in ["heuristic", "oracle"]:
            weight = adaptive_raw_weight(item, mode)
            method = f"adaptive_{mode}_rerank_top{top_n}"
            rankings[method] = convmem_rerank(
                raw_scores,
                conv_scores,
                windows,
                memory_ids,
                raw_top_n=top_n,
                raw_weight=weight,
            )

    rows = []
    for method, ranked in rankings.items():
        rows.append(
            {
                "method": method,
                "question_id": item["question_id"],
                "question_type": item["question_type"],
                "recall_at_5": recall_at_k(ranked, item["gold_memory_ids"], top_k),
                "hit_at_5": hit_at_k(ranked, item["gold_memory_ids"], top_k),
                "mrr": mrr(ranked, item["gold_memory_ids"]),
            }
        )
    return rows


def summarize(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["method"], []).append(row)
    out = []
    for method, items in grouped.items():
        out.append(
            {
                "method": method,
                "questions": len(items),
                "recall_at_5": np.mean([float(x["recall_at_5"]) for x in items]),
                "hit_at_5": np.mean([float(x["hit_at_5"]) for x in items]),
                "mrr": np.mean([float(x["mrr"]) for x in items]),
            }
        )
    return sorted(out, key=lambda x: x["recall_at_5"], reverse=True)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/longmemeval_s_cleaned.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--conv-layers", type=int, default=1)
    parser.add_argument("--pooling", choices=["mean", "query_attention"], default="mean")
    parser.add_argument("--multi-scale-kernels", default=None)
    parser.add_argument("--projection", choices=["none", "linear", "mlp"], default="none")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--raw-weight", type=float, default=0.55)
    parser.add_argument("--raw-weights", default=None)
    parser.add_argument("--rerank-top-ns", default=None)
    parser.add_argument("--rerank-score-norms", default="none")
    parser.add_argument("--adaptive-rerank-top-ns", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--encoder", choices=["tfidf", "sbert"], default="tfidf")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--embedding-cache", default=None)
    parser.add_argument("--distractor-sessions", type=int, default=0)
    parser.add_argument("--hard-negatives", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="results/longmemeval")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if args.raw_weights:
        raw_weights = [float(x.strip()) for x in args.raw_weights.split(",") if x.strip()]
    else:
        raw_weights = [args.raw_weight]
    if args.rerank_top_ns:
        rerank_top_values = [int(x.strip()) for x in args.rerank_top_ns.split(",") if x.strip()]
        score_norms = [x.strip() for x in args.rerank_score_norms.split(",") if x.strip()]
        rerank_top_ns = [(top_n, norm) for top_n in rerank_top_values for norm in score_norms]
    else:
        rerank_top_ns = []
    if args.adaptive_rerank_top_ns:
        adaptive_rerank_top_ns = [
            int(x.strip()) for x in args.adaptive_rerank_top_ns.split(",") if x.strip()
        ]
    else:
        adaptive_rerank_top_ns = []

    base_dir = Path(__file__).parent
    data_path = base_dir / args.data
    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing {data_path}. Download LongMemEval first, then rerun this script."
        )

    examples = load_longmemeval(data_path, limit=args.limit)
    examples = add_distractors(examples, args.distractor_sessions, seed=args.seed)
    random.shuffle(examples)
    split = int(len(examples) * args.train_ratio)
    train_examples = examples[:split]
    test_examples = examples[split:]

    texts = []
    for item in train_examples:
        texts.append(item["query"])
        texts.extend(m["text"] for m in item["memories"])
    if args.encoder == "tfidf":
        encoder = TfidfTextEncoder()
    else:
        encoder = SentenceTransformerTextEncoder(
            model_name=args.embedding_model,
            device=device,
            batch_size=args.embedding_batch_size,
            cache_path=args.embedding_cache,
        )
    encoder.fit(texts)

    encoded_train = [
        prepare_encoded_example(x, encoder, args.window_size, args.stride)
        for x in train_examples
    ]
    encoded_test = [
        prepare_encoded_example(x, encoder, args.window_size, args.stride)
        for x in test_examples
    ]
    encoded_train = [x for x in encoded_train if x is not None]
    encoded_test = [x for x in encoded_test if x is not None]

    dim = encoded_train[0]["memory_embeddings"].shape[1]
    multi_scale_kernels = None
    if args.multi_scale_kernels:
        multi_scale_kernels = [
            int(x.strip()) for x in args.multi_scale_kernels.split(",") if x.strip()
        ]
    model = ConvMemoryEncoder(
        dim,
        args.kernel_size,
        args.conv_layers,
        pooling=args.pooling,
        multi_scale_kernels=multi_scale_kernels,
        projection=args.projection,
    ).to(device)

    start = time.perf_counter()
    train_conv(
        model,
        encoded_train,
        epochs=args.epochs,
        device=device,
        hard_negatives=args.hard_negatives,
    )
    train_time = time.perf_counter() - start

    rows = []
    start = time.perf_counter()
    for item in encoded_test:
        rows.extend(
            evaluate_item(
                item,
                model,
                raw_weights=raw_weights,
                rerank_top_ns=rerank_top_ns,
                adaptive_rerank_top_ns=adaptive_rerank_top_ns,
                device=device,
            )
        )
    eval_time = time.perf_counter() - start

    out_dir = base_dir / args.out
    summary = summarize(rows)
    write_csv(out_dir / "longmemeval_detailed_results.csv", rows)
    write_csv(out_dir / "longmemeval_summary_results.csv", summary)

    avg_sessions = np.mean([len(x["memories"]) for x in encoded_test])
    avg_windows = np.mean([len(x["windows"]) for x in encoded_test])
    print("\nConvMem LongMemEval retrieval experiment")
    print(f"data: {data_path}")
    print(f"train questions: {len(encoded_train)}")
    print(f"test questions: {len(encoded_test)}")
    print(f"device: {device}")
    print(f"encoder: {args.encoder}")
    if args.encoder == "sbert":
        print(f"embedding model: {args.embedding_model}")
    print(f"hard negatives: {args.hard_negatives}")
    print(f"pooling: {args.pooling}")
    if multi_scale_kernels:
        print(f"multi-scale kernels: {multi_scale_kernels}")
    print(f"projection: {args.projection}")
    print(f"avg sessions per test question: {avg_sessions:.1f}")
    print(f"avg windows per test question: {avg_windows:.1f}")
    print(f"compression ratio: {avg_windows / avg_sessions:.3f}")
    print(f"train time: {train_time:.1f}s")
    print(f"eval latency: {1000 * eval_time / max(1, len(encoded_test)):.2f}ms/query")
    print("\nmethod            questions  recall@5  hit@5  mrr")
    for row in summary:
        print(
            f"{row['method']:<17} "
            f"{row['questions']:<9} "
            f"{row['recall_at_5']:.3f}     "
            f"{row['hit_at_5']:.3f}  "
            f"{row['mrr']:.3f}"
        )
    print(f"\nSaved: {out_dir / 'longmemeval_summary_results.csv'}")
    print(f"Saved: {out_dir / 'longmemeval_detailed_results.csv'}")


if __name__ == "__main__":
    main()

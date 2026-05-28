import json
from pathlib import Path
import re

import numpy as np
import pytest

from convmemory import ConvMemory, EvidenceReranker, EvidenceRerankerConfig
from convmemory.evidence_reranker import FORBIDDEN_FIELDS


def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def tiny_model(device="cpu"):
    return ConvMemory.from_config(
        embedding_dim=24,
        hidden_dim=24,
        token_mlp_dim=8,
        channel_mlp_dim=48,
        device=device,
    )


def tiny_inputs(n=12, dim=24):
    rng = np.random.default_rng(361)
    query = normalize(rng.normal(size=(1, dim)))[0]
    memories = normalize(rng.normal(size=(n, dim)))
    ids = [f"mem-{idx}" for idx in range(n)]
    texts = [f"candidate memory evidence {idx}" for idx in range(n)]
    return query, memories, ids, texts


def patch_encoder(model, query_embedding, memory_embeddings):
    encoded = np.vstack([query_embedding[None, :], memory_embeddings]).astype(np.float32)

    def fake_encode(texts):
        assert len(texts) == encoded.shape[0]
        return encoded

    model.encode = fake_encode


def memories_from(ids, texts):
    return [{"id": memory_id, "text": text} for memory_id, text in zip(ids, texts)]


def result_tuple(results):
    return [
        (item.memory_id, item.score, item.raw_score, item.rank, item.text)
        for item in results
    ]


class FakeCrossEncoder:
    def __init__(self, model_name_or_path=None, num_labels=1, max_length=256, device=None):
        self.model_name_or_path = str(model_name_or_path)
        self.offset = 0.0
        state_path = Path(self.model_name_or_path) / "fake_state.json"
        if state_path.exists():
            self.offset = float(json.loads(state_path.read_text(encoding="utf-8"))["offset"])

    def predict(self, pairs, batch_size=32, show_progress_bar=False):
        scores = []
        for _, candidate in pairs:
            matches = re.findall(r"-?\d+(?:\.\d+)?", candidate)
            value = float(matches[-1]) if matches else float(len(candidate))
            scores.append(self.offset + value / 100.0)
        return np.asarray(scores, dtype=np.float32)

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "fake_state.json").write_text(
            json.dumps({"offset": self.offset}),
            encoding="utf-8",
        )


def test_default_behavior_unchanged(device):
    model = tiny_model(device)
    query, memories, ids, texts = tiny_inputs()

    before = model.rerank_embeddings(query, memories, ids, texts, query="compat")
    after = model.rerank_embeddings(
        query,
        memories,
        ids,
        texts,
        query="compat",
        evidence_reranker=None,
    )

    assert result_tuple(after) == result_tuple(before)


def test_evidence_reranker_rejects_forbidden_fields():
    reranker = EvidenceReranker(
        EvidenceRerankerConfig(top_k=10),
        cross_encoder=FakeCrossEncoder(),
    )
    for field in FORBIDDEN_FIELDS:
        candidate = {"id": "m1", "text": "safe text", field: "forbidden"}
        with pytest.raises(ValueError, match=f"field '{field}' is not allowed"):
            reranker.score("query", [candidate])


def test_evidence_reranker_recall_preserving(device):
    model = tiny_model(device)
    model.attach_evidence_reranker(
        EvidenceReranker(
            EvidenceRerankerConfig(top_k=10),
            cross_encoder=FakeCrossEncoder(),
        )
    )
    query, embeddings, ids, texts = tiny_inputs(n=14)
    patch_encoder(model, query, embeddings)

    base = model.retrieve("query", memories_from(ids, texts), top_k=10)
    reranked = model.retrieve(
        "query",
        memories_from(ids, texts),
        top_k=10,
        evidence_reranker="v2",
    )

    assert {item.memory_id for item in reranked} == {item.memory_id for item in base}
    assert len(reranked) == len(base) == 10


def test_evidence_reranker_save_load_roundtrip(monkeypatch, tmp_checkpoint_dir):
    import convmemory.evidence_reranker as evidence_module

    monkeypatch.setattr(evidence_module, "CrossEncoder", FakeCrossEncoder)
    reranker = EvidenceReranker(
        EvidenceRerankerConfig(top_k=3, max_length=64),
        cross_encoder=FakeCrossEncoder(),
    )
    reranker.cross_encoder.offset = 1.5
    candidates = [
        {"id": "m1", "text": "memory one", "position": 1},
        {"id": "m2", "text": "memory two", "position": 2},
        {"id": "m3", "text": "memory three", "position": 3},
    ]
    before = reranker.score("query", candidates)

    reranker.save_pretrained(tmp_checkpoint_dir)
    loaded = EvidenceReranker.from_pretrained(tmp_checkpoint_dir)
    after = loaded.score("query", candidates)

    assert np.allclose(before, after, atol=1e-6)

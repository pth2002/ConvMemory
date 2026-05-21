import numpy as np
import pytest

from convmemory import (
    CCGELowAmplitudeEditor,
    ConvMemory,
    build_ccge_features,
)
from convmemory.ccge import FEATURE_NAMES


class DummyEncoder:
    def __init__(self, dim=32):
        self.dim = dim

    def encode(
        self,
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ):
        rows = []
        for text in texts:
            seed = sum(ord(ch) for ch in str(text)) % 997
            rng = np.random.default_rng(seed)
            rows.append(rng.normal(size=self.dim))
        x = np.asarray(rows, dtype=np.float32)
        if normalize_embeddings:
            x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
        return x


def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def feature_batch():
    embeddings = normalize(np.eye(4, dtype=np.float32))
    return build_ccge_features(
        candidate_ids=["a", "b", "c", "d"],
        convmemory_scores=[0.4, 0.3, 0.2, 0.1],
        dense_scores=[0.1, 0.4, 0.2, 0.3],
        positions=[0, 1, 2, 3],
        candidate_embeddings=embeddings,
        query="current job",
        candidate_texts=[
            "old job was analyst",
            "current job is engineer",
            "likes tea",
            "assistant summary",
        ],
    )


def tiny_model(device="cpu"):
    model = ConvMemory.from_config(
        embedding_dim=32,
        hidden_dim=32,
        token_mlp_dim=8,
        channel_mlp_dim=64,
        device=device,
    )
    model.embedding_model = DummyEncoder(dim=32)
    model.embedding_model_name = "dummy-local"
    return model


def tiny_memories():
    return [
        {"id": "m0", "text": "old job was analyst"},
        {"id": "m1", "text": "current job is engineer"},
        {"id": "m2", "text": "likes tea"},
        {"id": "m3", "text": "assistant summary"},
    ]


def test_build_ccge_features_basic():
    batch = feature_batch()

    assert batch.features.shape == (4, len(FEATURE_NAMES))


def test_build_ccge_features_empty_raises():
    with pytest.raises(ValueError):
        build_ccge_features(candidate_ids=[], convmemory_scores=[])


def test_build_ccge_features_shape_mismatch_raises():
    with pytest.raises(ValueError):
        build_ccge_features(candidate_ids=["a", "b"], convmemory_scores=[0.1])


def test_editor_save_load_roundtrip(device, tmp_checkpoint_dir):
    batch = feature_batch()
    editor = CCGELowAmplitudeEditor(model_dim=32, num_heads=4, layers=1).to(device)
    before, _ = editor.edit_batch(batch, device=device)

    editor.save_pretrained(tmp_checkpoint_dir)
    loaded = CCGELowAmplitudeEditor.from_pretrained(tmp_checkpoint_dir, device=device)
    after, _ = loaded.edit_batch(batch, device=device)

    assert np.allclose(before, after, atol=1e-6)


def test_untrained_editor_low_amplitude(device):
    batch = feature_batch()
    editor = CCGELowAmplitudeEditor(model_dim=32, num_heads=4, layers=1).to(device)

    scores, _ = editor.edit_batch(batch, device=device)
    base_top3 = set(np.argsort(-batch.features[:, 0])[:3])
    edited_top3 = set(np.argsort(-scores)[:3])

    assert len(base_top3 & edited_top3) >= 2


def test_resolve_editor_strict(device):
    model = tiny_model(device)

    with pytest.raises(ValueError, match="editor"):
        model.retrieve(
            query="current job?",
            memories=tiny_memories(),
            editor="not_a_known_string",
        )


def test_resolve_editor_accepts_canonical(device):
    model = tiny_model(device)
    model.attach_ccge_editor(CCGELowAmplitudeEditor(model_dim=32, layers=1, num_heads=4))

    results = model.retrieve(
        query="current job?",
        memories=tiny_memories(),
        editor="ccge_la",
    )

    assert results

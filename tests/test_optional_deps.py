"""The core reranker must work without the encoder stack installed.

`pip install convmemory` gives you numpy + torch. Stores that already hold
embeddings call `rerank_embeddings` and never need sentence-transformers. These
tests block the import outright rather than trusting that it is unused.
"""

import subprocess
import sys

import numpy as np
import pytest

from convmemory import ConvMemory


class _BlockSentenceTransformers:
    """Meta-path hook that makes `import sentence_transformers` fail."""

    def find_spec(self, name, path=None, target=None):
        if name == "sentence_transformers" or name.startswith("sentence_transformers."):
            raise ImportError("sentence_transformers is blocked for this test")
        return None


@pytest.fixture
def without_sentence_transformers(monkeypatch):
    for name in [n for n in sys.modules if n.startswith("sentence_transformers")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    blocker = _BlockSentenceTransformers()
    monkeypatch.setattr(sys, "meta_path", [blocker, *sys.meta_path])
    return blocker


def tiny_model():
    return ConvMemory.from_config(
        embedding_dim=32, hidden_dim=32, token_mlp_dim=8, channel_mlp_dim=64, device="cpu"
    )


def unit_rows(n, dim=32, seed=7):
    rng = np.random.default_rng(seed)
    rows = rng.normal(size=(n, dim)).astype(np.float32)
    return rows / (np.linalg.norm(rows, axis=-1, keepdims=True) + 1e-8)


def test_import_does_not_pull_sentence_transformers():
    """A bare `import convmemory` must not drag in the encoder stack.

    Runs in a subprocess: reimporting convmemory in-process would swap module
    identities out from under the rest of the suite.
    """
    probe = (
        "import sys, convmemory;"
        "print(any(n.startswith('sentence_transformers') for n in sys.modules),"
        "      'transformers' in sys.modules)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert completed.stdout.split() == ["False", "False"], completed.stdout


def test_rerank_embeddings_works_without_encoder(without_sentence_transformers):
    model = tiny_model()
    memories = unit_rows(8)
    query = unit_rows(1)[0]

    results = model.rerank_embeddings(
        query_embedding=query,
        memory_embeddings=memories,
        memory_ids=[f"m{i}" for i in range(8)],
        memory_texts=[f"memory {i}" for i in range(8)],
        query="test",
        top_k=3,
    )

    assert [item.rank for item in results] == [1, 2, 3]


def test_encode_without_encoder_raises_actionable_error(without_sentence_transformers):
    from convmemory._optional import load_sentence_transformer

    with pytest.raises(ImportError) as error:
        load_sentence_transformer()

    message = str(error.value)
    assert "convmemory[encode]" in message
    assert "rerank_embeddings" in message


def test_cross_encoder_error_names_the_extra(without_sentence_transformers):
    from convmemory._optional import load_cross_encoder

    with pytest.raises(ImportError) as error:
        load_cross_encoder()

    assert "convmemory[encode]" in str(error.value)


def test_encode_without_attached_model_still_raises_value_error():
    """A config-built model has no encoder; that path must stay a ValueError."""
    model = tiny_model()
    with pytest.raises(ValueError, match="rerank_embeddings"):
        model.encode(["some text"])

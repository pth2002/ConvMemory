import numpy as np
import pytest

from convmemory import ConvMemory


def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def tiny_model(device="cpu"):
    return ConvMemory.from_config(
        embedding_dim=32,
        hidden_dim=32,
        token_mlp_dim=8,
        channel_mlp_dim=64,
        device=device,
    )


def tiny_inputs(n=8, dim=32):
    rng = np.random.default_rng(7)
    query = normalize(rng.normal(size=(1, dim)))[0]
    memories = normalize(rng.normal(size=(n, dim)))
    ids = [f"m{i}" for i in range(n)]
    texts = [f"memory {i}" for i in range(n)]
    return query, memories, ids, texts


def test_from_config_rerank_embeddings_roundtrip(device):
    model = tiny_model(device)
    query, memories, ids, texts = tiny_inputs()

    results = model.rerank_embeddings(query, memories, ids, texts, query="test")

    assert len(results) == 8
    assert [item.rank for item in results] == list(range(1, 9))
    assert all(isinstance(item.score, float) for item in results)


def test_save_load_roundtrip(device, tmp_checkpoint_dir):
    model = tiny_model(device)
    query, memories, ids, texts = tiny_inputs()
    before = model.rerank_embeddings(query, memories, ids, texts, query="test")

    model.save_pretrained(tmp_checkpoint_dir)
    loaded = ConvMemory.from_pretrained(
        tmp_checkpoint_dir,
        device=device,
        embedding_model=False,
    )
    after = loaded.rerank_embeddings(query, memories, ids, texts, query="test")

    assert [item.memory_id for item in after] == [item.memory_id for item in before]


def test_empty_memories_raises(device):
    model = tiny_model(device)
    query = normalize(np.ones((1, 32), dtype=np.float32))[0]

    with pytest.raises(RuntimeError):
        model.rerank_embeddings(
            query,
            np.zeros((0, 32), dtype=np.float32),
            [],
            [],
            query="empty",
        )


def test_single_memory(device):
    model = tiny_model(device)
    query, memories, ids, texts = tiny_inputs(n=1)

    results = model.rerank_embeddings(query, memories, ids, texts, query="single")

    assert len(results) == 1
    assert results[0].rank == 1


def test_invalid_window_mode_raises(device):
    model = tiny_model(device)
    query, memories, ids, texts = tiny_inputs()

    with pytest.raises(ValueError, match="window_mode"):
        model.rerank_embeddings(
            query,
            memories,
            ids,
            texts,
            query="test",
            window_mode="xxx",
        )


def test_expand_context_budget_edge_cases(device):
    model = tiny_model(device)
    query, memories, ids, texts = tiny_inputs()

    assert (
        model.expand_context_embeddings(
            query,
            memories,
            ids,
            texts,
            query="test",
            context_budget=0,
        )
        == []
    )
    results = model.expand_context_embeddings(
        query,
        memories,
        ids,
        texts,
        query="test",
        protected_k=10,
        context_budget=3,
    )
    assert len(results) == 3
    assert [item.rank for item in results] == [1, 2, 3]

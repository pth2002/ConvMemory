import numpy as np
import pytest

from convmemory import ConvMemory, MemoryMLAConfig, MemoryMLAExpander, RerankResult


def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def tiny_inputs(n=10, dim=32):
    rng = np.random.default_rng(19)
    query = normalize(rng.normal(size=(1, dim)))[0]
    memories = normalize(rng.normal(size=(n, dim)))
    ids = [f"m{i}" for i in range(n)]
    texts = [f"memory text {i}" for i in range(n)]
    return query, memories, ids, texts


def tiny_model(device="cpu"):
    return ConvMemory.from_config(
        embedding_dim=32,
        hidden_dim=32,
        token_mlp_dim=8,
        channel_mlp_dim=64,
        device=device,
    )


def tiny_expander(device="cpu"):
    config = MemoryMLAConfig(
        embedding_dim=32,
        latent_count=4,
        code_dim=16,
        model_dim=32,
        hidden_dim=64,
        setwise_layers=0,
        heads=4,
        query_slots=2,
        pair_top_k=8,
        cluster_top_k=4,
        candidate_top_n=8,
        protect_top_k=2,
        expand_window=6,
    )
    return MemoryMLAExpander(config).to(device).eval()


def patch_encoder(model, query_embedding, memory_embeddings):
    encoded = np.vstack([query_embedding[None, :], memory_embeddings]).astype(np.float32)

    def fake_encode(texts):
        assert len(texts) == encoded.shape[0]
        return encoded

    model.encode = fake_encode


def text_memories(ids, texts):
    return [{"id": memory_id, "text": text} for memory_id, text in zip(ids, texts)]


def ranked_results(ids, texts):
    return [
        RerankResult(
            memory_id=memory_id,
            score=float(0.5 - rank * 0.1),
            raw_score=float(1.0 - rank * 0.1),
            rank=rank + 1,
            text=text,
        )
        for rank, (memory_id, text) in enumerate(zip(ids, texts))
    ]


def test_expander_build_and_forward(device):
    expander = tiny_expander(device)
    query, memories, ids, texts = tiny_inputs(n=6)
    results = ranked_results(ids, texts)
    codes = expander.build_codes_from_embeddings(memories)
    features = expander.build_features(
        query="who changed jobs",
        candidate_results=results,
        candidate_indices=list(range(len(ids))),
        candidate_texts=texts,
    )

    scores = expander.score_batch(query_embedding=query, candidate_codes=codes, features=features, device=device)

    assert scores.shape == (6,)
    assert np.isfinite(scores).all()


def test_expander_save_load_roundtrip(device, tmp_checkpoint_dir):
    expander = tiny_expander(device)
    query, memories, ids, texts = tiny_inputs(n=5)
    results = ranked_results(ids, texts)
    codes = expander.build_codes_from_embeddings(memories)
    features = expander.build_features(
        query="current project",
        candidate_results=results,
        candidate_indices=list(range(len(texts))),
        candidate_texts=texts,
    )
    before = expander.score_batch(query_embedding=query, candidate_codes=codes, features=features, device=device)

    expander.save_pretrained(tmp_checkpoint_dir)
    loaded = MemoryMLAExpander.from_pretrained(tmp_checkpoint_dir, device=device)
    after = loaded.score_batch(query_embedding=query, candidate_codes=codes, features=features, device=device)

    assert np.allclose(before, after, atol=1e-6)


def test_expander_hub_id_roundtrip(monkeypatch, device, tmp_checkpoint_dir):
    expander = tiny_expander(device)
    expander.save_pretrained(tmp_checkpoint_dir)

    import convmemory.hub as hub

    monkeypatch.setattr(hub, "_hf_snapshot_download", lambda repo_id, repo_type="model": str(tmp_checkpoint_dir))
    loaded = MemoryMLAExpander.from_pretrained("Purdy0228/ConvMemory-Memory-MLA", device=device)

    assert isinstance(loaded, MemoryMLAExpander)


def test_v1_backward_compat(device):
    model = tiny_model(device)
    query, memories, ids, texts = tiny_inputs(n=9)

    before = model.rerank_embeddings(query, memories, ids, texts, query="compat")
    after = model.rerank_embeddings(query, memories, ids, texts, query="compat")

    assert [item.memory_id for item in after] == [item.memory_id for item in before]
    assert [item.rank for item in after] == [item.rank for item in before]
    assert np.allclose([item.score for item in after], [item.score for item in before])


def test_prefix_protection(device):
    model = tiny_model(device)
    model.attach_expander(tiny_expander(device))
    query, memories, ids, texts = tiny_inputs(n=10)
    patch_encoder(model, query, memories)

    base = model.retrieve("query", text_memories(ids, texts), top_k=None)
    expanded = model.retrieve(
        "query",
        text_memories(ids, texts),
        top_k=None,
        expander="memory_mla",
        protect_top_k=3,
        expand_window=8,
    )

    assert [item.memory_id for item in expanded[:3]] == [item.memory_id for item in base[:3]]


def test_resolve_expander_strict(device):
    model = tiny_model(device)
    query, memories, ids, texts = tiny_inputs(n=6)
    patch_encoder(model, query, memories)

    with pytest.raises(ValueError, match="expander"):
        model.retrieve("query", text_memories(ids, texts), expander="bogus")


def test_attach_and_use(device):
    model = tiny_model(device)
    model.attach_expander(tiny_expander(device))
    query, memories, ids, texts = tiny_inputs(n=6)
    patch_encoder(model, query, memories)

    results = model.retrieve(
        "query",
        text_memories(ids, texts),
        expander="memory_mla",
        protect_top_k=2,
        expand_window=5,
    )

    assert results
    assert [item.rank for item in results] == list(range(1, len(results) + 1))

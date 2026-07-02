import json

import numpy as np
import torch

import convmemory.chinese as zh_module
from convmemory import ChineseConvMemory
from convmemory.chinese import PairwiseCELiteScorer
from convmemory.encoder import MixerConvMemoryEncoder


def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


class FakeDualEncoder:
    def __init__(self, dim=16):
        self.dim = dim

    def encode(self, texts):
        rows = []
        for text in texts:
            seed = abs(hash(str(text))) % (2**32)
            rng = np.random.default_rng(seed)
            rows.append(rng.normal(size=self.dim))
        return normalize(np.asarray(rows, dtype=np.float32))


def make_checkpoint(path, dim=16):
    path.mkdir(parents=True, exist_ok=True)
    conv = MixerConvMemoryEncoder(
        dim,
        window_size=3,
        kernel_size=3,
        hidden_dim=16,
        token_mlp_dim=4,
        channel_mlp_dim=32,
        output_mode="residual",
        output_gate_init=0.1,
        score_mode="cosine",
    )
    scorer = PairwiseCELiteScorer(
        dim,
        hidden_dim=16,
        extra_scalar_features=7,
        interaction_dim=8,
    )
    torch.save(
        {
            "model_state_dict": conv.state_dict(),
            "scorer_state_dict": scorer.state_dict(),
        },
        path / "student.pt",
    )
    config = {
        "encoder": {
            "base_model": "fake-base",
            "tuned_model": "fake-tuned",
        },
        "architecture": {
            "embedding_dim": dim,
            "window_size": 3,
            "stride": 1,
            "kernel": 3,
            "mixer_hidden_dim": 16,
            "mixer_token_dim": 4,
            "mixer_channel_dim": 32,
            "ce_hidden_dim": 16,
            "interaction_dim": 8,
            "extra_scalar_features": 7,
            "dca_router_block_size": 2,
            "dual_window_features": True,
            "lexical_features": True,
        },
        "training": {
            "candidate_top_n": 8,
        },
        "default_raw_weight": 0.0,
        "student_sha256": "test",
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")


def test_chinese_convmemory_from_pretrained_with_injected_encoder(tmp_path, monkeypatch):
    ckpt = tmp_path / "zh"
    make_checkpoint(ckpt)
    monkeypatch.setattr(
        zh_module,
        "DualSpaceTextEncoder",
        lambda *args, **kwargs: FakeDualEncoder(dim=16),
    )

    model = ChineseConvMemory.from_pretrained(
        ckpt,
        device="cpu",
        base_encoder_model="fake-base",
        tuned_encoder_model="fake-tuned",
    )

    memories = [
        {"id": "m0", "text": "用户喜欢周杰伦的音乐。"},
        {"id": "m1", "text": "用户最近在学习法语。"},
        {"id": "m2", "text": "用户计划周末去上海。"},
        {"id": "m3", "text": "用户不吃辣。"},
    ]
    results = model.rerank("用户喜欢什么音乐？", memories, top_k=3)

    assert len(results) == 3
    assert {item.memory_id for item in results}.issubset({"m0", "m1", "m2", "m3"})
    assert [item.rank for item in results] == [1, 2, 3]
    assert all(isinstance(item.score, float) for item in results)


def test_chinese_convmemory_rerank_embeddings(tmp_path, monkeypatch):
    ckpt = tmp_path / "zh"
    make_checkpoint(ckpt)
    monkeypatch.setattr(
        zh_module,
        "DualSpaceTextEncoder",
        lambda *args, **kwargs: FakeDualEncoder(dim=16),
    )
    model = ChineseConvMemory.from_pretrained(
        ckpt,
        device="cpu",
        base_encoder_model="fake-base",
        tuned_encoder_model="fake-tuned",
    )
    rng = np.random.default_rng(13)
    query = normalize(rng.normal(size=(1, 16)))[0]
    memories = normalize(rng.normal(size=(6, 16)))
    ids = [f"m{i}" for i in range(6)]

    results = model.rerank_embeddings(
        query_embedding=query,
        memory_embeddings=memories,
        memory_ids=ids,
        memory_texts=[f"中文记忆 {i}" for i in range(6)],
        query="测试查询",
    )

    assert len(results) == 6
    assert sorted(item.memory_id for item in results) == ids

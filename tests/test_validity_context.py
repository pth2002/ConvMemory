import numpy as np
import pytest
import json
from pathlib import Path

from convmemory import ConvMemory, ValidityEvidenceConfig, ValidityEvidenceModule
from convmemory.validity import FORBIDDEN_FIELDS


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


def tiny_inputs(n=8, dim=24):
    rng = np.random.default_rng(555)
    query = normalize(rng.normal(size=(1, dim)))[0]
    memories = normalize(rng.normal(size=(n, dim)))
    ids = [f"mem-{idx}" for idx in range(n)]
    texts = [
        "Paris travel plan",
        "Later update says Paris plan moved to Lyon",
        "Coffee preference",
        "Book club note",
        "Gym schedule",
        "Project reminder",
        "Music preference",
        "Restaurant idea",
    ][:n]
    return query, memories, ids, texts


def result_tuple(results, include_validity=True):
    rows = []
    for item in results:
        row = (item.memory_id, item.score, item.raw_score, item.rank, item.text)
        if include_validity:
            row = (*row, item.validity)
        rows.append(row)
    return rows


def scorer(query, target, source):
    del query
    source_text = str(source.get("text", "")).lower()
    target_text = str(target.get("text", "")).lower()
    if "later update" in source_text and "paris" in target_text:
        return 0.9
    if "later update" in target_text and "paris" in source_text:
        return 0.2
    return 0.0


class FakeCrossEncoder:
    def __init__(self, model_name_or_path=None, num_labels=1, max_length=192, device=None):
        del num_labels, max_length, device
        self.model_name_or_path = str(model_name_or_path)
        self.last_pairs = []
        self.calls = []
        self.offset = 0.0
        state_path = Path(self.model_name_or_path) / "fake_state.json"
        if state_path.exists():
            self.offset = float(json.loads(state_path.read_text(encoding="utf-8"))["offset"])

    def predict(self, pairs, batch_size=32, show_progress_bar=False):
        del show_progress_bar
        self.last_pairs = list(pairs)
        self.calls.append({"n_pairs": len(pairs), "batch_size": batch_size})
        out = []
        for left, right in pairs:
            score = self.offset
            if "SOURCE_EVIDENCE:" in left and "TARGET_MEMORY:" in right:
                score += 0.8
            out.append(score)
        return np.asarray(out, dtype=np.float32)

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "fake_state.json").write_text(
            json.dumps({"offset": self.offset}),
            encoding="utf-8",
        )


def memories_from(ids, texts):
    return [
        {"id": memory_id, "text": text, "position": idx}
        for idx, (memory_id, text) in enumerate(zip(ids, texts))
    ]


def test_off_mode_byte_identical(device):
    model = tiny_model(device)
    model.attach_validity_module(ValidityEvidenceModule(scorer=scorer))
    query, memories, ids, texts = tiny_inputs()

    base = model.rerank_embeddings(query, memories, ids, texts, query="trip")
    none_mode = model.rerank_embeddings(
        query,
        memories,
        ids,
        texts,
        query="trip",
        validity_mode=None,
    )
    off_mode = model.rerank_embeddings(
        query,
        memories,
        ids,
        texts,
        query="trip",
        validity_mode="off",
    )

    assert result_tuple(none_mode) == result_tuple(base)
    assert result_tuple(off_mode) == result_tuple(base)


def test_context_mode_preserves_ranking(device):
    model = tiny_model(device)
    model.attach_validity_module(ValidityEvidenceModule(scorer=scorer))
    query, memories, ids, texts = tiny_inputs()

    base = model.rerank_embeddings(query, memories, ids, texts, query="trip")
    context = model.rerank_embeddings(
        query,
        memories,
        ids,
        texts,
        query="trip",
        validity_mode="context",
    )

    assert result_tuple(context, include_validity=False) == result_tuple(
        base,
        include_validity=False,
    )
    assert all(item.validity is not None for item in context)


def test_demote_preserves_candidate_set(device):
    model = tiny_model(device)
    model.attach_validity_module(
        ValidityEvidenceModule(
            ValidityEvidenceConfig(demote_threshold=0.1, demote_score_scale=10.0),
            scorer=scorer,
        )
    )
    query, memories, ids, texts = tiny_inputs()

    base = model.rerank_embeddings(query, memories, ids, texts, query="trip")
    demoted = model.rerank_embeddings(
        query,
        memories,
        ids,
        texts,
        query="trip",
        validity_mode="demote",
    )

    assert {item.memory_id for item in demoted} == {item.memory_id for item in base}
    assert len(demoted) == len(base)
    assert any((item.validity or {}).get("action") == "demote" for item in demoted)


def test_demote_is_opt_in(device):
    model = tiny_model(device)
    model.attach_validity_module(
        ValidityEvidenceModule(
            ValidityEvidenceConfig(demote_threshold=0.1, demote_score_scale=10.0),
            scorer=scorer,
        )
    )
    query, memories, ids, texts = tiny_inputs()

    base = model.rerank_embeddings(query, memories, ids, texts, query="trip")
    context = model.rerank_embeddings(
        query,
        memories,
        ids,
        texts,
        query="trip",
        validity_mode="context",
    )

    assert result_tuple(context, include_validity=False) == result_tuple(
        base,
        include_validity=False,
    )
    assert all((item.validity or {}).get("action") != "demote" for item in context)


def test_forbidden_fields_rejected(device):
    model = tiny_model(device)
    model.attach_validity_module(ValidityEvidenceModule(scorer=scorer))
    query, embeddings, ids, texts = tiny_inputs()

    def fake_encode(all_texts):
        assert len(all_texts) == len(texts) + 1
        return np.vstack([query[None, :], embeddings]).astype(np.float32)

    model.encode = fake_encode
    for field in FORBIDDEN_FIELDS:
        memories = memories_from(ids, texts)
        memories[0][field] = "forbidden"
        with pytest.raises(ValueError, match=f"field '{field}' is not allowed"):
            model.retrieve("trip", memories, validity_mode="context")


def test_context_evidence_has_no_forbidden_fields(device):
    model = tiny_model(device)
    model.attach_validity_module(ValidityEvidenceModule(scorer=scorer))
    query, memories, ids, texts = tiny_inputs()

    context = model.rerank_embeddings(
        query,
        memories,
        ids,
        texts,
        query="trip",
        validity_mode="context",
    )

    for item in context:
        for evidence in (item.validity or {}).get("source_evidence", []):
            assert not FORBIDDEN_FIELDS.intersection(evidence.keys())


def test_validity_save_load_roundtrip(tmp_checkpoint_dir):
    module = ValidityEvidenceModule(ValidityEvidenceConfig(context_threshold=0.0))
    memories = [
        {"id": "m0", "text": "Paris travel plan", "position": 0},
        {"id": "m1", "text": "Later Paris update", "position": 1},
    ]
    results = [
        type("Result", (), {"memory_id": "m0", "text": "Paris travel plan"})(),
        type("Result", (), {"memory_id": "m1", "text": "Later Paris update"})(),
    ]
    before = module.annotate(query="trip", results=results, memories=memories)

    module.save_pretrained(tmp_checkpoint_dir)
    loaded = ValidityEvidenceModule.from_pretrained(tmp_checkpoint_dir)
    after = loaded.annotate(query="trip", results=results, memories=memories)

    assert [item.to_dict() for item in after] == [item.to_dict() for item in before]


def test_validity_cross_encoder_uses_query_source_target_format():
    cross_encoder = FakeCrossEncoder()
    module = ValidityEvidenceModule(
        ValidityEvidenceConfig(cross_encoder_model="fake-model"),
        cross_encoder=cross_encoder,
    )
    memories = [
        {"id": "m0", "text": "old Paris plan", "position": 0},
        {"id": "m1", "text": "new Paris update", "position": 1},
    ]
    results = [
        type("Result", (), {"memory_id": "m0", "text": "old Paris plan"})(),
    ]

    annotations = module.annotate(query="what is current?", results=results, memories=memories)

    assert annotations[0].confidence == pytest.approx(0.8, abs=1e-6)
    assert cross_encoder.last_pairs
    left, right = cross_encoder.last_pairs[0]
    assert "USER_QUERY:" in left
    assert "SOURCE_EVIDENCE:" in left
    assert "TASK: Decide whether the target memory should be demoted" in left
    assert "TARGET_MEMORY:" in right


def test_validity_cross_encoder_batches_apply_pairs():
    cross_encoder = FakeCrossEncoder()
    module = ValidityEvidenceModule(
        ValidityEvidenceConfig(cross_encoder_model="fake-model", cross_encoder_batch_size=16),
        cross_encoder=cross_encoder,
    )
    memories = [
        {"id": "m0", "text": "old Paris plan", "position": 0},
        {"id": "m1", "text": "old Rome plan", "position": 1},
        {"id": "m2", "text": "new Paris and Rome updates", "position": 2},
    ]
    results = [
        type("Result", (), {"memory_id": "m0", "text": "old Paris plan"})(),
        type("Result", (), {"memory_id": "m1", "text": "old Rome plan"})(),
    ]

    module.annotate(query="what is current?", results=results, memories=memories)

    assert len(cross_encoder.calls) == 1
    assert cross_encoder.calls[0] == {"n_pairs": 2, "batch_size": 16}


def test_validity_auto_source_policy_scores_at_most_one_source_per_target():
    cross_encoder = FakeCrossEncoder()
    module = ValidityEvidenceModule(
        ValidityEvidenceConfig(cross_encoder_model="fake-model", source_policy="top1"),
        cross_encoder=cross_encoder,
    )
    memories = [
        {"id": "m0", "text": "old Paris plan", "position": 0},
        {"id": "m1", "text": "new Paris update", "position": 1},
        {"id": "m2", "text": "another Paris update", "position": 2},
        {"id": "m3", "text": "unrelated note", "position": 3},
    ]
    results = [
        type("Result", (), {"memory_id": memory["id"], "text": memory["text"]})()
        for memory in memories
    ]

    module.annotate(query="what is current?", results=results, memories=memories)

    assert cross_encoder.calls == [{"n_pairs": 3, "batch_size": 32}]


def test_validity_explicit_source_map_skips_internal_source_search():
    cross_encoder = FakeCrossEncoder()
    module = ValidityEvidenceModule(
        ValidityEvidenceConfig(cross_encoder_model="fake-model"),
        cross_encoder=cross_encoder,
    )
    memories = [
        {"id": "m0", "text": "old Paris plan", "position": 0},
        {"id": "m1", "text": "irrelevant note", "position": 1},
        {"id": "m2", "text": "new Paris update", "position": 2},
    ]
    results = [type("Result", (), {"memory_id": "m0", "text": "old Paris plan"})()]

    module.annotate(
        query="what is current?",
        results=results,
        memories=memories,
        source_evidence={"m0": "m2"},
    )

    assert cross_encoder.calls == [{"n_pairs": 1, "batch_size": 32}]
    left, right = cross_encoder.last_pairs[0]
    assert "new Paris update" in left
    assert "old Paris plan" in right


def test_validity_auto_source_policy_supports_chinese_overlap():
    cross_encoder = FakeCrossEncoder()
    module = ValidityEvidenceModule(
        ValidityEvidenceConfig(cross_encoder_model="fake-model"),
        cross_encoder=cross_encoder,
    )
    memories = [
        {"id": "m0", "text": "旧定价方案是基础版99元", "position": 0},
        {"id": "m1", "text": "后来开始学习吉他", "position": 1},
        {"id": "m2", "text": "定价方案后来更新为基础版129元", "position": 2},
    ]
    results = [type("Result", (), {"memory_id": "m0", "text": memories[0]["text"]})()]

    module.annotate(query="现在的定价方案是什么", results=results, memories=memories)

    left, _ = cross_encoder.last_pairs[0]
    assert "定价方案后来更新" in left


def test_validity_rejects_earlier_explicit_source():
    cross_encoder = FakeCrossEncoder()
    module = ValidityEvidenceModule(
        ValidityEvidenceConfig(cross_encoder_model="fake-model", require_later_source=True),
        cross_encoder=cross_encoder,
    )
    memories = [
        {"id": "m0", "text": "earlier Paris note", "position": 0},
        {"id": "m1", "text": "current Paris plan", "position": 1},
    ]
    results = [type("Result", (), {"memory_id": "m1", "text": "current Paris plan"})()]

    annotations = module.annotate(
        query="what is current?",
        results=results,
        memories=memories,
        source_evidence={"m1": "m0"},
    )

    assert cross_encoder.calls == []
    assert annotations[0].status == "unknown"


def test_validity_explicit_source_rejects_forbidden_fields():
    module = ValidityEvidenceModule(scorer=scorer)
    memories = [{"id": "m0", "text": "old Paris plan", "position": 0}]
    results = [type("Result", (), {"memory_id": "m0", "text": "old Paris plan"})()]

    with pytest.raises(ValueError, match="field 'teacher_score' is not allowed"):
        module.annotate(
            query="what is current?",
            results=results,
            memories=memories,
            source_evidence={
                "m0": {"id": "external", "text": "new Paris update", "teacher_score": 1.0}
            },
        )


def test_demote_preserves_requested_top_k_set(device):
    model = tiny_model(device)
    model.attach_validity_module(
        ValidityEvidenceModule(
            ValidityEvidenceConfig(demote_threshold=0.1, demote_score_scale=10.0),
            scorer=lambda query, target, source: 0.9,
        )
    )
    query, memories, ids, texts = tiny_inputs()
    base = model.rerank_embeddings(query, memories, ids, texts, query="trip", top_k=3)
    source_map = {
        result.memory_id: {"id": f"source-{idx}", "text": "later update evidence"}
        for idx, result in enumerate(base)
    }

    demoted = model.rerank_embeddings(
        query,
        memories,
        ids,
        texts,
        query="trip",
        top_k=3,
        validity_mode="demote",
        validity_source_map=source_map,
    )

    assert {item.memory_id for item in demoted} == {item.memory_id for item in base}


def test_integrated_validity_limits_targets_when_top_k_is_omitted(device):
    calls = []

    def counting_scorer(query, target, source):
        calls.append((query, target["id"], source["id"]))
        return 0.9

    model = tiny_model(device)
    model.attach_validity_module(
        ValidityEvidenceModule(
            ValidityEvidenceConfig(candidate_top_k=3),
            scorer=counting_scorer,
        )
    )
    query, memories, ids, texts = tiny_inputs()
    source_map = {
        memory_id: {"id": f"source-{idx}", "text": "later update evidence"}
        for idx, memory_id in enumerate(ids)
    }

    context = model.rerank_embeddings(
        query,
        memories,
        ids,
        texts,
        query="trip",
        validity_mode="context",
        validity_source_map=source_map,
    )

    assert len(calls) == 3
    assert all(item.validity is not None for item in context[:3])
    assert all(item.validity is None for item in context[3:])


def test_expand_context_applies_validity_after_final_selection(device):
    model = tiny_model(device)
    model.attach_validity_module(
        ValidityEvidenceModule(scorer=lambda query, target, source: 0.9)
    )
    query, memories, ids, texts = tiny_inputs()
    source_map = {
        memory_id: {"id": f"source-{idx}", "text": "later update evidence"}
        for idx, memory_id in enumerate(ids)
    }

    context = model.expand_context_embeddings(
        query_embedding=query,
        memory_embeddings=memories,
        memory_ids=ids,
        memory_texts=texts,
        query="trip",
        context_budget=5,
        validity_mode="context",
        validity_source_map=source_map,
    )

    assert len(context) == 5
    assert all(item.validity is not None for item in context)


def test_validity_score_evidence_pairs_batches_explicit_pairs():
    cross_encoder = FakeCrossEncoder()
    module = ValidityEvidenceModule(
        ValidityEvidenceConfig(cross_encoder_model="fake-model", cross_encoder_batch_size=8),
        cross_encoder=cross_encoder,
    )
    scores = module.score_evidence_pairs(
        [
            {"query": "current?", "source": "new update", "target": "old plan"},
            {
                "query": "current?",
                "source": {"text": "another update"},
                "target": {"text": "another old plan"},
            },
        ]
    )

    assert scores == pytest.approx([0.8, 0.8], abs=1e-6)
    assert cross_encoder.calls == [{"n_pairs": 2, "batch_size": 8}]


def test_validity_cross_encoder_save_load_roundtrip(monkeypatch, tmp_checkpoint_dir):
    import convmemory.validity as validity_module

    monkeypatch.setattr(validity_module, "load_cross_encoder", lambda: FakeCrossEncoder)
    module = ValidityEvidenceModule(
        ValidityEvidenceConfig(cross_encoder_model="fake-model"),
        cross_encoder=FakeCrossEncoder(),
    )
    module.cross_encoder.offset = 0.1
    memories = [
        {"id": "m0", "text": "old Paris plan", "position": 0},
        {"id": "m1", "text": "new Paris update", "position": 1},
    ]
    results = [
        type("Result", (), {"memory_id": "m0", "text": "old Paris plan"})(),
    ]
    before = module.annotate(query="what is current?", results=results, memories=memories)

    module.save_pretrained(tmp_checkpoint_dir)
    loaded = ValidityEvidenceModule.from_pretrained(tmp_checkpoint_dir)
    after = loaded.annotate(query="what is current?", results=results, memories=memories)

    assert [item.to_dict() for item in after] == [item.to_dict() for item in before]


def test_invalid_validity_mode_raises(device):
    model = tiny_model(device)
    model.attach_validity_module(ValidityEvidenceModule(scorer=scorer))
    query, memories, ids, texts = tiny_inputs()

    with pytest.raises(ValueError, match="validity_mode"):
        model.rerank_embeddings(
            query,
            memories,
            ids,
            texts,
            query="trip",
            validity_mode="auto",
        )

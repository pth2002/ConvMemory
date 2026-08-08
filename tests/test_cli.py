import json

import numpy as np
import pytest

from convmemory import cli
from convmemory.reranker import RerankResult


def write_jsonl(path, records):
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )
    return path


class StubModel:
    """Stands in for a loaded checkpoint: no hub, no encoder download.

    Ranks memories by how many query words they contain, so the reranked order
    is deterministic and different from the embedding order.
    """

    def __init__(self):
        self.prewarmed = 0

    def encode(self, texts):
        rng = np.random.default_rng(0)
        vectors = rng.normal(size=(len(list(texts)), 8)).astype(np.float32)
        return vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)

    def prewarm_lexical(self, memories):
        self.prewarmed += 1

    def rerank_embeddings(self, *, query, memory_ids, memory_texts, top_k=None, **kwargs):
        words = set(query.lower().split())
        scored = sorted(
            zip(memory_ids, memory_texts),
            key=lambda pair: -len(words & set(pair[1].lower().split())),
        )
        results = [
            RerankResult(memory_id=mid, score=1.0 / rank, raw_score=1.0 / rank, rank=rank, text=text)
            for rank, (mid, text) in enumerate(scored, start=1)
        ]
        return results[:top_k] if top_k else results


@pytest.fixture
def stub_checkpoint(monkeypatch):
    from convmemory import api

    monkeypatch.setattr(api.ConvMemory, "from_pretrained", classmethod(lambda cls, *a, **k: StubModel()))


@pytest.fixture
def dataset(tmp_path):
    memories = write_jsonl(
        tmp_path / "memories.jsonl",
        [
            {"id": "m1", "text": "the analytics store moved to clickhouse", "group": "g1"},
            {"id": "m2", "text": "the offsite is in porto", "group": "g1"},
            {"id": "m3", "text": "unrelated chatter about coffee", "group": "g1"},
        ],
    )
    queries = write_jsonl(
        tmp_path / "queries.jsonl",
        [{"query": "analytics store clickhouse", "gold_ids": ["m1"], "group": "g1"}],
    )
    return queries, memories


def run(argv):
    return cli.main(argv)


def test_benchmark_end_to_end(stub_checkpoint, dataset, tmp_path, capsys):
    queries, memories = dataset
    out = tmp_path / "summary.json"

    assert run(
        [
            "benchmark",
            "--queries", str(queries),
            "--memories", str(memories),
            "--json", str(out),
        ]
    ) == 0

    printed = capsys.readouterr().out
    assert "Before ConvMemory" in printed
    assert "After ConvMemory" in printed

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["queries"] == 1
    assert payload["memory_groups"] == 1
    # The stub ranks the gold memory first, so post-rerank MRR is perfect.
    assert payload["after"]["mrr"] == pytest.approx(1.0)
    assert "recall@5" in payload["after"] and "recall@10" in payload["after"]
    assert payload["after"]["ms_per_query"] >= 0.0


def test_custom_cutoffs_and_limit(stub_checkpoint, dataset, tmp_path):
    queries, memories = dataset
    out = tmp_path / "summary.json"

    assert run(
        [
            "benchmark",
            "--queries", str(queries),
            "--memories", str(memories),
            "--k", "1", "3",
            "--limit", "1",
            "--json", str(out),
        ]
    ) == 0

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert set(payload["after"]) >= {"recall@1", "recall@3", "hit@1", "hit@3"}


def test_missing_file_exits(stub_checkpoint, dataset, tmp_path):
    queries, _ = dataset
    with pytest.raises(SystemExit):
        run(["benchmark", "--queries", str(queries), "--memories", str(tmp_path / "nope.jsonl")])


def test_unknown_group_exits(stub_checkpoint, dataset, tmp_path):
    _, memories = dataset
    queries = write_jsonl(
        tmp_path / "bad.jsonl", [{"query": "x", "gold_ids": ["m1"], "group": "ghost"}]
    )
    with pytest.raises(SystemExit) as error:
        run(["benchmark", "--queries", str(queries), "--memories", str(memories)])
    assert "ghost" in str(error.value)


def test_missing_gold_field_exits(stub_checkpoint, dataset, tmp_path):
    _, memories = dataset
    queries = write_jsonl(tmp_path / "bad.jsonl", [{"query": "x", "group": "g1"}])
    with pytest.raises(SystemExit) as error:
        run(["benchmark", "--queries", str(queries), "--memories", str(memories)])
    assert "gold_ids" in str(error.value)


def test_duplicate_memory_id_exits(stub_checkpoint, dataset, tmp_path):
    queries, _ = dataset
    memories = write_jsonl(
        tmp_path / "dupe.jsonl",
        [{"id": "m1", "text": "a", "group": "g1"}, {"id": "m1", "text": "b", "group": "g1"}],
    )
    with pytest.raises(SystemExit) as error:
        run(["benchmark", "--queries", str(queries), "--memories", str(memories)])
    assert "duplicate" in str(error.value)


def test_invalid_json_reports_line_number(stub_checkpoint, dataset, tmp_path):
    queries, _ = dataset
    memories = tmp_path / "broken.jsonl"
    memories.write_text('{"id": "m1", "text": "ok"}\nnot json\n', encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        run(["benchmark", "--queries", str(queries), "--memories", str(memories)])
    assert ":2:" in str(error.value)


def test_gold_id_singular_accepted(tmp_path):
    path = write_jsonl(tmp_path / "q.jsonl", [{"query": "x", "gold_id": "m1"}])
    assert cli._load_queries(path)[0]["gold_ids"] == ["m1"]


def test_no_command_prints_help(capsys):
    assert cli.main([]) == 1
    assert "benchmark" in capsys.readouterr().out

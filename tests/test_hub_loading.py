import numpy as np

from convmemory import CCGELowAmplitudeEditor, ConvMemory
from convmemory import hub


def test_from_pretrained_accepts_hub_id(monkeypatch, tmp_checkpoint_dir, device):
    model = ConvMemory.from_config(
        embedding_dim=16,
        hidden_dim=16,
        token_mlp_dim=8,
        channel_mlp_dim=32,
        device=device,
    )
    model.save_pretrained(tmp_checkpoint_dir)
    monkeypatch.setattr(hub, "_hf_snapshot_download", lambda repo_id, repo_type: str(tmp_checkpoint_dir))

    loaded = ConvMemory.from_pretrained(
        "Purdy0228/ConvMemory-LoCoMo-MPNet",
        device=device,
        embedding_model=False,
    )

    q = np.ones(16, dtype=np.float32)
    q = q / np.linalg.norm(q)
    memories = np.eye(16, dtype=np.float32)[:3]
    results = loaded.rerank_embeddings(q, memories, ["a", "b", "c"], query="test")
    assert len(results) == 3


def test_load_ccge_editor_accepts_hub_id(monkeypatch, tmp_checkpoint_dir, device):
    editor = CCGELowAmplitudeEditor(model_dim=32, layers=1, num_heads=4)
    editor.save_pretrained(tmp_checkpoint_dir)
    monkeypatch.setattr(hub, "_hf_snapshot_download", lambda repo_id, repo_type: str(tmp_checkpoint_dir))
    model = ConvMemory.from_config(
        embedding_dim=16,
        hidden_dim=16,
        token_mlp_dim=8,
        channel_mlp_dim=32,
        device=device,
    )

    model.load_ccge_editor("Purdy0228/ConvMemory-CCGE-LA")

    assert model.ccge_editor is not None

"""Minimal CCGE-LA API example.

This example demonstrates how to attach and call the conflict editor. The editor
created here is randomly initialized, so it is only an API smoke test. Real
applications should load a trained CCGE-LA checkpoint with
``model.load_ccge_editor(path)``.
"""

import numpy as np

from convmemory import CCGELowAmplitudeEditor, ConvMemory


def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def main():
    rng = np.random.default_rng(11)
    embedding_dim = 32
    query_embedding = normalize(rng.normal(size=(1, embedding_dim)))[0]
    memory_embeddings = normalize(rng.normal(size=(8, embedding_dim)))
    memory_ids = [f"m{i}" for i in range(len(memory_embeddings))]
    memory_texts = [
        "User said the hiking trip was on Saturday.",
        "Assistant recommended a rain jacket.",
        "User later moved the hiking trip to Sunday.",
        "User asked about a dinner reservation.",
        "Assistant summarized the calendar.",
        "User said the old Saturday plan was cancelled.",
        "User likes quiet cafes.",
        "Assistant noted a study deadline.",
    ]

    model = ConvMemory.from_config(
        embedding_dim=embedding_dim,
        hidden_dim=32,
        token_mlp_dim=8,
        channel_mlp_dim=64,
        device="cpu",
    )
    model.attach_ccge_editor(
        CCGELowAmplitudeEditor(model_dim=32, layers=1, num_heads=4)
    )

    results = model.rerank_embeddings(
        query_embedding=query_embedding,
        memory_embeddings=memory_embeddings,
        memory_ids=memory_ids,
        memory_texts=memory_texts,
        query="When is the hiking trip?",
        editor="ccge_la",
        top_k=5,
    )

    for item in results:
        print(f"{item.rank:02d} {item.memory_id} score={item.score:.4f}")


if __name__ == "__main__":
    main()

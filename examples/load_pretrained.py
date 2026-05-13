from pathlib import Path

import numpy as np

from convmemory import ConvMemory


def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def main():
    checkpoint = Path("checkpoints/convmemory-locomo-mpnet")
    if not (checkpoint / "config.json").exists() or not (checkpoint / "model.pt").exists():
        raise SystemExit(
            "Checkpoint not found. Download or place a ConvMemory checkpoint at "
            "checkpoints/convmemory-locomo-mpnet/ before running this example."
        )

    model = ConvMemory.from_pretrained(
        checkpoint,
        device="cpu",
        embedding_model=False,
    )

    rng = np.random.default_rng(11)
    query_embedding = normalize(rng.normal(size=(1, 768)))[0]
    memory_embeddings = normalize(rng.normal(size=(3, 768)))
    memory_ids = ["m1", "m2", "m3"]
    memory_texts = [
        "The user said their hiking trip moved to Sunday.",
        "The assistant recommended bringing extra water.",
        "The user has an exam next Friday.",
    ]

    results = model.rerank_embeddings(
        query_embedding=query_embedding,
        memory_embeddings=memory_embeddings,
        memory_ids=memory_ids,
        memory_texts=memory_texts,
        query="When is the hiking trip?",
        top_k=2,
    )
    for item in results:
        print(item.rank, item.memory_id, item.text)


if __name__ == "__main__":
    main()

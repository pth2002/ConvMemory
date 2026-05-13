import numpy as np

from convmemory import ConvMemory


def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def main():
    rng = np.random.default_rng(7)
    embedding_dim = 32
    query_embedding = normalize(rng.normal(size=(1, embedding_dim)))[0]
    memory_embeddings = normalize(rng.normal(size=(8, embedding_dim)))
    memory_ids = [f"m{i}" for i in range(len(memory_embeddings))]
    memory_texts = [
        "User mentioned a hiking trip.",
        "Assistant suggested packing water.",
        "User said the exam is next Friday.",
        "User asked about dinner plans.",
        "Assistant noted the restaurant reservation.",
        "User changed the hiking trip to Sunday.",
        "User prefers quiet cafes.",
        "Assistant summarized the schedule.",
    ]

    model = ConvMemory.from_config(
        embedding_dim=embedding_dim,
        hidden_dim=32,
        token_mlp_dim=8,
        channel_mlp_dim=64,
        device="cpu",
    )

    results = model.rerank_embeddings(
        query_embedding=query_embedding,
        memory_embeddings=memory_embeddings,
        memory_ids=memory_ids,
        memory_texts=memory_texts,
        query="When is the hiking trip?",
    )

    for item in results[:5]:
        print(f"{item.rank:02d} {item.memory_id} score={item.score:.4f}")


if __name__ == "__main__":
    main()

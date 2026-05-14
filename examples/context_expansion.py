import numpy as np

from convmemory import ConvMemory


def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def main():
    rng = np.random.default_rng(13)
    embedding_dim = 32
    query_embedding = normalize(rng.normal(size=(1, embedding_dim)))[0]
    memory_embeddings = normalize(rng.normal(size=(12, embedding_dim)))
    memory_ids = [f"m{i}" for i in range(len(memory_embeddings))]
    memory_texts = [
        "User said they were comparing roles in Shanghai and Chengdu.",
        "User mentioned that family is based in Chengdu.",
        "Assistant summarized the job-search criteria.",
        "User said Shanghai has good opportunities but higher pressure.",
        "User prefers avoiding long commutes.",
        "User planned to revisit the decision after interviews.",
        "Assistant suggested ranking cities by lifestyle fit.",
        "User noted that remote work would change the decision.",
        "User said salary is important but not the only factor.",
        "Assistant recorded the user's preference for stability.",
        "User asked for a reminder to compare living costs.",
        "Assistant connected the city choice to career goals.",
    ]

    model = ConvMemory.from_config(
        embedding_dim=embedding_dim,
        hidden_dim=32,
        token_mlp_dim=8,
        channel_mlp_dim=64,
        device="cpu",
    )

    context = model.expand_context_embeddings(
        query_embedding=query_embedding,
        memory_embeddings=memory_embeddings,
        memory_ids=memory_ids,
        memory_texts=memory_texts,
        query="Why was I unsure about working in Shanghai?",
        protected_k=3,
        context_budget=6,
    )

    for item in context:
        print(f"{item.rank:02d} {item.memory_id} score={item.score:.4f}")


if __name__ == "__main__":
    main()

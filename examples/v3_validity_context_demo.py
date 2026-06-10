"""Minimal ConvMemory v3 validity-context demo.

This example uses precomputed toy embeddings and a tiny deterministic scorer so
it runs without downloading a checkpoint. Replace the scorer by
`load_validity_module("path-or-hub-id")` for a trained v3 validity module.
"""

import numpy as np

from convmemory import ConvMemory, ValidityEvidenceModule


def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def update_scorer(query, target, source):
    del query
    target_text = str(target.get("text", "")).lower()
    source_text = str(source.get("text", "")).lower()
    if "paris" in target_text and "moved to lyon" in source_text:
        return 0.91
    return 0.0


def main():
    model = ConvMemory.from_config(
        embedding_dim=24,
        hidden_dim=24,
        token_mlp_dim=8,
        channel_mlp_dim=48,
    )
    model.attach_validity_module(ValidityEvidenceModule(scorer=update_scorer))

    rng = np.random.default_rng(3)
    query_embedding = normalize(rng.normal(size=(24,)))
    memory_embeddings = normalize(rng.normal(size=(4, 24)))
    memory_ids = ["m0", "m1", "m2", "m3"]
    memory_texts = [
        "I planned to visit Paris next month.",
        "Later update: the trip moved to Lyon instead.",
        "I like dark roast coffee.",
        "The book club meets on Fridays.",
    ]

    ranked = model.rerank_embeddings(
        query_embedding=query_embedding,
        memory_embeddings=memory_embeddings,
        memory_ids=memory_ids,
        memory_texts=memory_texts,
        query="Where is my current trip?",
        top_k=4,
        validity_mode="context",
    )

    for item in ranked:
        print(item.rank, item.memory_id, item.text)
        if item.validity:
            print("  validity:", item.validity["status"], item.validity["context_note"])


if __name__ == "__main__":
    main()

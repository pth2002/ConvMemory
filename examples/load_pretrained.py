from convmemory import ConvMemory


def main():
    model = ConvMemory.from_pretrained(
        "checkpoints/convmemory-locomo-mpnet",
        device="cpu",
    )
    memories = [
        {"id": "m1", "text": "The user said their hiking trip moved to Sunday."},
        {"id": "m2", "text": "The assistant recommended bringing extra water."},
        {"id": "m3", "text": "The user has an exam next Friday."},
    ]
    results = model.rerank("When is the hiking trip?", memories, top_k=2)
    for item in results:
        print(item.rank, item.memory_id, item.text)


if __name__ == "__main__":
    main()

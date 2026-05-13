from pathlib import Path

from convmemory import ConvMemory


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

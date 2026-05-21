"""Run CCGE-LA with real checkpoints on a small stale/current scenario.

This is a checkpoint demonstration, not a smoke test. By default it loads the
public Hugging Face Hub checkpoints. If local checkpoint folders exist under
``checkpoints/``, those are used instead.
"""

from pathlib import Path

from convmemory import ConvMemory


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "checkpoints" / "convmemory-locomo-mpnet"
CCGE = ROOT / "checkpoints" / "convmemory-ccge-la-locomo-mpnet-seed23-alpha"
BASE_HUB = "Purdy0228/ConvMemory-LoCoMo-MPNet"
CCGE_HUB = "Purdy0228/ConvMemory-CCGE-LA"


def print_ranking(title, results):
    print(f"\n{title}")
    for item in results:
        print(f"{item.rank:02d} {item.memory_id} score={item.score:.4f} {item.text}")


def main():
    base = BASE if BASE.exists() else BASE_HUB
    ccge = CCGE if CCGE.exists() else CCGE_HUB

    print(f"Loading ConvMemory from {base}")
    print(f"Loading CCGE-LA from {ccge}")
    model = ConvMemory.from_pretrained(base)
    model.load_ccge_editor(ccge)

    memories = [
        {
            "id": "m0",
            "text": "In January, the user said their job was a data analyst.",
        },
        {
            "id": "m1",
            "text": "The assistant noted that the user prefers concise summaries.",
        },
        {
            "id": "m2",
            "text": "In March, the user changed jobs and became a machine learning engineer.",
        },
        {
            "id": "m3",
            "text": "The user asked for interview prep help for engineering roles.",
        },
        {
            "id": "m4",
            "text": "The old data analyst role is no longer the user's current job.",
        },
    ]
    query = "What is the user's current job?"

    base_results = model.retrieve(query=query, memories=memories, editor=None, top_k=5)
    edited_results = model.retrieve(query=query, memories=memories, editor="ccge_la", top_k=5)

    print_ranking("ConvMemory only", base_results)
    print_ranking("ConvMemory + CCGE-LA", edited_results)


if __name__ == "__main__":
    main()

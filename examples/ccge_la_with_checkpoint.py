"""Run CCGE-LA with real checkpoints on a small stale/current scenario.

This is a checkpoint demonstration, not a smoke test. Download and extract both
the base ConvMemory checkpoint and the CCGE-LA alpha checkpoint before running:

- checkpoints/convmemory-locomo-mpnet/
- checkpoints/convmemory-ccge-la-locomo-mpnet-seed23-alpha/
"""

from pathlib import Path

from convmemory import ConvMemory


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "checkpoints" / "convmemory-locomo-mpnet"
CCGE = ROOT / "checkpoints" / "convmemory-ccge-la-locomo-mpnet-seed23-alpha"


def print_ranking(title, results):
    print(f"\n{title}")
    for item in results:
        print(f"{item.rank:02d} {item.memory_id} score={item.score:.4f} {item.text}")


def main():
    if not BASE.exists():
        print(f"Missing base checkpoint: {BASE}")
        print("Download convmemory-locomo-mpnet.zip from the GitHub release page.")
        return
    if not CCGE.exists():
        print(f"Missing CCGE-LA checkpoint: {CCGE}")
        print("Download convmemory-ccge-la-locomo-mpnet-seed23-alpha.zip first.")
        return

    model = ConvMemory.from_pretrained(BASE)
    model.load_ccge_editor(CCGE)

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

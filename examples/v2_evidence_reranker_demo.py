"""Minimal ConvMemory v2 evidence-reranker API demo.

Usage:
  python examples/v2_evidence_reranker_demo.py \
    --base Purdy0228/ConvMemory-LoCoMo-MPNet \
    --evidence path-or-hub-id-tbd

Checkpoint distribution for the canonical v2 evidence reranker is TBD. If
`--evidence` is omitted, this script attaches an untrained default CrossEncoder
only to demonstrate API shape; do not use that mode for quality evaluation.
"""

from __future__ import annotations

import argparse

from convmemory import ConvMemory, EvidenceReranker, EvidenceRerankerConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Purdy0228/ConvMemory-LoCoMo-MPNet")
    parser.add_argument("--evidence", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model = ConvMemory.from_pretrained(args.base, device=args.device)
    if args.evidence:
        model.load_evidence_reranker(args.evidence, device=args.device)
    else:
        print("No evidence checkpoint supplied; attaching an untrained API-demo reranker.")
        model.attach_evidence_reranker(
            EvidenceReranker(EvidenceRerankerConfig(), device=args.device)
        )

    memories = [
        {"id": "turn-1", "text": "Alex used to work as a backend engineer."},
        {"id": "turn-2", "text": "Alex later moved into product management."},
        {"id": "turn-3", "text": "Alex currently leads the memory platform team."},
        {"id": "turn-4", "text": "Alex likes hiking on weekends."},
    ]
    ranked = model.retrieve(
        query="What is Alex's current job?",
        memories=memories,
        evidence_reranker="v2",
        top_k=4,
    )
    for item in ranked:
        print(f"{item.rank:02d} {item.memory_id} score={item.score:.4f} {item.text}")


if __name__ == "__main__":
    main()

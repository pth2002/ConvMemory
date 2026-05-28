"""Train a ConvMemory v2 evidence reranker from user-provided records.

Input JSON/JSONL records must look like:

{
  "query": "What is the user's current role?",
  "candidates": [
    {"id": "m1", "text": "Earlier memory...", "position": 1},
    {"id": "m2", "text": "Current memory...", "position": 2}
  ],
  "gold_ids": ["m2"],
  "teacher_scores": [0.1, 1.7]   # optional
}

Gold ids and optional teacher scores are training targets only. They are not
accepted by the public inference API.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Sequence

import numpy as np
import torch
from sentence_transformers import CrossEncoder
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from convmemory import EvidenceReranker, EvidenceRerankerConfig


def load_records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, dict):
        data = data.get("records", [])
    return list(data)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def candidate_position(candidate: dict, fallback: int) -> float:
    for key in ("position", "time"):
        if key in candidate:
            return float(candidate[key])
    return float(fallback)


def format_pairs(record: dict) -> list[tuple[str, str]]:
    candidates = list(record["candidates"])
    positions = [candidate_position(candidate, idx) for idx, candidate in enumerate(candidates)]
    query_time = max(positions) if positions else 0.0
    query = f"QUERY_TIME: {query_time:.0f}. {record['query']}"
    return [
        (query, f"MEMORY_TIME: {position:.0f}. {candidate['text']}")
        for candidate, position in zip(candidates, positions)
    ]


def collate_records(model: CrossEncoder, records: Sequence[dict], device: torch.device):
    left: list[str] = []
    right: list[str] = []
    gold_targets: list[np.ndarray] = []
    teacher_targets: list[np.ndarray] = []
    has_gold: list[float] = []
    for record in records:
        candidates = list(record["candidates"])
        pairs = format_pairs(record)
        for query, memory in pairs:
            left.append(query)
            right.append(memory)
        gold_ids = {str(x) for x in record.get("gold_ids", [])}
        gold = np.asarray(
            [1.0 if str(candidate["id"]) in gold_ids else 0.0 for candidate in candidates],
            dtype=np.float32,
        )
        if float(gold.sum()) > 0:
            gold = gold / float(gold.sum())
            has_gold.append(1.0)
        else:
            gold = np.full(len(candidates), 1.0 / max(1, len(candidates)), dtype=np.float32)
            has_gold.append(0.0)
        teacher = np.asarray(record.get("teacher_scores", np.zeros(len(candidates))), dtype=np.float32)
        teacher = teacher - float(teacher.max()) if teacher.size else teacher
        teacher = np.exp(teacher / 2.0)
        teacher = teacher / max(float(teacher.sum()), 1.0e-8)
        gold_targets.append(gold)
        teacher_targets.append(teacher.astype(np.float32))

    features = model.tokenizer(
        left,
        right,
        padding=True,
        truncation=True,
        max_length=model.max_length,
        return_tensors="pt",
    )
    return (
        {key: value.to(device) for key, value in features.items()},
        torch.tensor(np.stack(gold_targets), dtype=torch.float32, device=device),
        torch.tensor(np.stack(teacher_targets), dtype=torch.float32, device=device),
        torch.tensor(has_gold, dtype=torch.float32, device=device),
    )


def train_one_seed(args, records: list[dict], seed: int) -> EvidenceReranker:
    set_seed(seed)
    device = torch.device(args.device)
    model = CrossEncoder(
        args.cross_encoder_model,
        num_labels=1,
        max_length=args.max_length,
        device=args.device,
    )
    random.shuffle(records)
    split_at = max(1, int(len(records) * (1.0 - args.dev_ratio)))
    train_records = records[:split_at]
    loader = DataLoader(
        train_records,
        shuffle=True,
        batch_size=args.train_batch_size,
        collate_fn=lambda batch: collate_records(model, batch, device),
    )
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(args.warmup_steps, max(0, total_steps // 10)),
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.use_amp and device.type == "cuda"))
    model.model.train()
    for epoch in range(args.epochs):
        for features, gold_target, teacher_target, has_gold in loader:
            optimizer.zero_grad(set_to_none=True)
            batch_size = gold_target.shape[0]
            with torch.cuda.amp.autocast(enabled=bool(args.use_amp and device.type == "cuda")):
                logits = model.model(**features, return_dict=True).logits.reshape(batch_size, -1)
                log_probs = torch.log_softmax(logits, dim=1)
                gold_loss = (-(gold_target * log_probs).sum(dim=1) * has_gold).sum()
                gold_loss = gold_loss / has_gold.sum().clamp_min(1.0)
                teacher_loss = -(teacher_target * log_probs).sum(dim=1).mean()
                loss = args.gold_weight * gold_loss + args.teacher_weight * teacher_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        print(f"seed={seed} epoch={epoch + 1}/{args.epochs} loss={float(loss.detach().cpu()):.4f}")
    return EvidenceReranker(
        EvidenceRerankerConfig(
            top_k=args.top_k,
            max_length=args.max_length,
            cross_encoder_model=args.cross_encoder_model,
        ),
        cross_encoder=model,
        device=args.device,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSON or JSONL training records")
    parser.add_argument("--output", required=True, help="Output checkpoint directory")
    parser.add_argument("--cross-encoder-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 11, 23, 31, 47])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--dev-ratio", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2.0e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--gold-weight", type=float, default=1.0)
    parser.add_argument("--teacher-weight", type=float, default=0.0)
    parser.add_argument("--use-amp", action="store_true")
    args = parser.parse_args()

    records = load_records(Path(args.input))
    if not records:
        raise ValueError("no training records found")
    # The last seed's checkpoint is saved for simplicity. For production,
    # select by held-out validation MRR or average checkpoints explicitly.
    reranker = None
    for seed in args.seeds:
        reranker = train_one_seed(args, list(records), seed)
    reranker.save_pretrained(args.output)
    print(f"saved evidence reranker to {args.output}")


if __name__ == "__main__":
    main()

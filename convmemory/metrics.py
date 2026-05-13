def recall_at_k(ranked_ids, gold_ids, k):
    return len(set(ranked_ids[:k]) & set(gold_ids)) / max(1, len(gold_ids))


def hit_at_k(ranked_ids, gold_ids, k):
    return float(bool(set(ranked_ids[:k]) & set(gold_ids)))


def mrr(ranked_ids, gold_ids):
    gold = set(gold_ids)
    for rank, item_id in enumerate(ranked_ids, start=1):
        if item_id in gold:
            return 1.0 / rank
    return 0.0

"""Two explicitly separate evaluation protocols. Default evaluate matches ADRec.

Optional evaluate_user_weighted scores use columns 0..max_item_id.

Padding is excluded. Previously seen products are retained by default.
predict receives ONLY histories and masks, never labels or held-out targets.
"""
import math
from .metrics import RankingMetrics

def evaluate_user_weighted(dataset, predict, num_items, batch_size=128, ks=(5, 10, 20)):
    if type(num_items) is not int or num_items < 1:
        raise ValueError("num_items must be the maximum positive product ID")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be positive")
    metric = RankingMetrics(ks)
    if max(metric.ks) > num_items:
        raise ValueError("Cutoff exceeds catalog size")
    for start in range(0, len(dataset), batch_size):
        rows = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
        if any(r["target_id"] > num_items for r in rows):
            raise ValueError("Target outside candidate vocabulary")
        scores = predict([r["input_ids"] for r in rows],
                         [r["attention_mask"] for r in rows])
        if hasattr(scores, "detach"):
            scores = scores.detach().cpu().tolist()
        elif hasattr(scores, "tolist"):
            scores = scores.tolist()
        if len(scores) != len(rows):
            raise ValueError("Prediction batch size mismatch")
        rankings = []
        for score in scores:
            if len(score) != num_items + 1:
                raise ValueError("Scores must include column 0 plus every product ID")
            if any(not math.isfinite(float(v)) for v in score[1:]):
                raise ValueError("Nonfinite candidate scores")
            # Stable ties: smaller product ID first.
            rankings.append(sorted(range(1, num_items + 1),
                                   key=lambda i: (-float(score[i]), i))[:max(metric.ks)])
        metric.update(rankings, [r["target_id"] for r in rows])
    return {"users": metric.count, "metrics": metric.compute()}


def evaluate(dataset, predict, num_items, batch_size=512, ks=(5, 10, 20)):
    """Original ADRec protocol: CPU torch.topk, batch-mean %, four decimals.

    No extra padding/seen-item filtering; prediction score dtype is preserved.
    For exact parity, use the same batching and score tensors as upstream.
    """
    import numpy as np
    import torch
    from .upstream_metrics import hrs_and_ndcgs_k
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be positive")
    if type(num_items) is not int or num_items < 1:
        raise ValueError("num_items must be positive")
    if not ks or any(type(k) is not int or k < 1 or k > num_items + 1 for k in ks):
        raise ValueError("Invalid metric cutoffs")
    if not len(dataset):
        raise ValueError("Cannot evaluate zero users")
    totals = {f"{metric}@{k}": [] for k in ks for metric in ("HR", "NDCG")}
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            rows = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
            scores = predict([r["input_ids"] for r in rows], [r["attention_mask"] for r in rows])
            scores = torch.as_tensor(scores).detach()
            if scores.shape != (len(rows), num_items + 1):
                raise ValueError("Scores must have shape [batch, num_items+1]")
            if not torch.isfinite(scores).all():
                raise ValueError("Nonfinite scores")
            labels = torch.tensor([[r["target_id"]] for r in rows], dtype=torch.long)
            if (labels < 1).any() or (labels > num_items).any():
                raise ValueError("Target outside vocabulary")
            batch_metrics = hrs_and_ndcgs_k(scores, labels, ks)
            for key, value in batch_metrics.items():
                totals[key].append(value)
    return {"users": len(dataset), "protocol": "adrec_upstream", "units": "percent",
            "metrics": {k: round(float(np.mean(v)) * 100, 4) for k,v in totals.items()}}

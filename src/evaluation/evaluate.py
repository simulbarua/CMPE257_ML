"""Model-independent evaluator. Scores use columns 0..max_item_id.

Padding is excluded. Previously seen products are retained by default.
predict receives ONLY histories and masks, never labels or held-out targets.
"""
import math
from .metrics import RankingMetrics

def evaluate(dataset, predict, num_items, batch_size=128, ks=(5, 10, 20)):
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

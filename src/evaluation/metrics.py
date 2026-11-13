"""HR/NDCG for one held-out item; accumulate per user, not per batch."""
import math

class RankingMetrics:
    def __init__(self, ks=(5, 10, 20)):
        if not ks or any(type(k) is not int or k < 1 for k in ks):
            raise ValueError("Cutoffs must be positive integers")
        self.ks = tuple(sorted(set(ks)))
        self.count = 0
        self.sums = {f"{metric}@{k}": 0.0 for k in self.ks
                     for metric in ("HR", "NDCG")}

    def update(self, rankings, targets):
        if len(rankings) != len(targets):
            raise ValueError("Rankings and targets must align")
        for ranked, target in zip(rankings, targets):
            if type(target) is not int or target < 1:
                raise ValueError("Target must be a positive item ID")
            if len(set(ranked)) != len(ranked) or any(type(i) is not int or i < 1 for i in ranked):
                raise ValueError("Rankings require unique positive item IDs")
            if len(ranked) < max(self.ks):
                raise ValueError("Not enough ranked candidates for requested cutoffs")
            rank = ranked.index(target) + 1 if target in ranked else None
            for k in self.ks:
                if rank is not None and rank <= k:
                    self.sums[f"HR@{k}"] += 1
                    self.sums[f"NDCG@{k}"] += 1 / math.log2(rank + 1)
            self.count += 1

    def compute(self):
        if not self.count:
            raise ValueError("Cannot evaluate zero users")
        return {key: value / self.count for key, value in self.sums.items()}

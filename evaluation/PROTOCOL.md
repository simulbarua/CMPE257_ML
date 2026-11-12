# Evaluation agreement — pending verification

Before full runs, all three members must agree on the following and record the implemented behavior:

- Exact dataset version, checksums, ID mapping, and padding token.
- Actual split semantics in supplied data and code, checked against the paper.
- Sequence maximum length, truncation, padding and attention masks.
- Target construction, eligible users/items, and leakage checks.
- Metric formulas and cutoffs from the paper, tested on a tiny known example.
- Candidate set and treatment of previously seen items, consistently across models.
- Seeds, validation-based model selection, and comparable tuning budgets.
- Timing hardware, batch size, warm-up, and whether loading/preprocessing is included.

Do not compare results obtained under incompatible settings as if they were equivalent. Document paper/code differences explicitly. Optional shortened-history analysis uses fixed users, targets, checkpoints, and candidate sets.

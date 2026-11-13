# Evaluation protocols

## Default: original ADRec behavior (`evaluate`)

Checked against ADRec commit `0a30ef764cd85efa44f884a387198e9bf28c7d4c`.

- Preserve released per-user holdouts: validation input=train; test input=train+val.
- Latest 50 items, left zero-padding, original product IDs.
- Score columns correspond to IDs 0..4731. No additional filtering of padding or previously seen products (same as inspected upstream score/metric path).
- Use the upstream HR/NDCG functions and CPU `torch.topk`; no custom tie-breaking.
- Compute batch metrics, take unweighted mean of batch means, multiply by 100 and round to four decimals.
- Outputs are **percentages**, not fractions. Default evaluation batch size is 512, matching upstream config. Identical batching and score tensors are required for comparison.
- Model prediction callback receives histories and masks only. ADRec adapter builds shifted visible history with a dummy final ID, masked to match original label padding; smoke check confirms identical scores with a fixed random seed versus upstream calls that receive labels.

Validation tests compare directly to the retained original functions, including ties, padding winning rank, float32/float64, and a short final batch. Real-model scores are also checked.

## Optional alternative (`evaluate_user_weighted`)

Prior team behavior retained under an explicit name: exclude padding, deterministic item-ID ties, per-user averages, fractions. It is not the reproduction evaluator. `RankingMetrics` supports this alternative only.

## Scope and limitations

CPU smoke runs establish interfaces and evaluation parity, not paper-level accuracy, convergence, or GPU reproducibility. ADRec smoke uses short stages and reduced hidden/diffusion sizes. The full upstream training schedule, validation checkpoint selection and GPU run still require validation. GRU4Rec uses a credited RecBole v1.2.0 CE adapter; the paper's exact GRU implementation/settings are not established.

The paper describes an 8:1:1 user split, but the released loader/file uses per-user item holdouts. Follow and disclose the released protocol. Record software, batch size, random seed, configs, data checksum and checkpoint provenance.

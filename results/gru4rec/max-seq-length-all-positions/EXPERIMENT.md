# Experiment record: GRU4Rec maximum sequence length on ml-100k

- Model: GRU4Rec, full mode. Maximum history lengths: [10, 25]. Seeds: 42-46.
- What changes: the maximum sequence length, in BOTH training and testing. This is not an evaluation-only truncation of a trained model.
- Dataset: `ml-100k`, SHA-256 `dfc5dfaa7950d1a85af4fd6bb168c38efc8213092a9f72563c6d75ab00cef8d0`. Released train/validation/test splits retained.
- Everything else matches the shared settings (hidden size 128, batch size 512, up to 500 epochs, validation every 5 epochs, patience 4).
- Selection: improvement in any tracked validation metric (the shared runner's rule). Test is evaluated once, at the end.
- Evaluation: `adrec_upstream`, HR/NDCG percentages, no seen-item filtering. Mean and sample standard deviation (n-1) over seeds.
- Notebook: this file. Weights are stored separately from these result files.

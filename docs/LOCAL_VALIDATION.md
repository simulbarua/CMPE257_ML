# Local reproduction checks

Branch: `fix/upstream-evaluation-checks`. Kept separate from `main` for team review.

## Re-run

From the repository root, install `requirements-shared.txt` in an isolated environment, then:

```bash
python -m unittest discover -s tests -v
python -m scripts.check_shared_pipeline --dataset data/baby/dataset.pkl
python -m scripts.check_upstream_models
```

The smoke runner requires the pinned Baby dataset in that location. It runs on CPU, uses nine users from the real file, preserves all 4,731 candidate products, and uses three batches (4,4,1). It performs two optimizer steps per stage/model, 16 hidden dimensions, and four ADRec diffusion steps. These are interface tests, not meaningful recommendation-quality results.

## Verified

- Shared training/validation/test records match the original loader for the smoke users.
- Original HR/NDCG results match the default evaluator on crafted rankings, random scores, ties, padding candidates and uneven batch sizes.
- SASRec+ forward/loss/backward/update and validation/test scoring.
- ADRec embedding pretraining; original checkpoint loading; frozen-embedding warm-up; full-parameter fine-tuning; gradients and weight changes; validation/test scoring.
- RecBole-based GRU4Rec CE adapter forward/loss/update and validation/test scoring. Padding conversion checked independently.
- All model checkpoints save/reload with identical parameters.
- Fixed-seed upstream inference with labels equals inference constructed from history only for the smoke cases.

Machine-readable results: `results/local_model_smoke.json`. Local package versions: `results/local_smoke_environment.txt`.

## Before full reproduction

Review changes before merging. Full model notebooks and training schedules are not yet implemented as team launchers. Confirm full-run GPU environment and model configs. The GRU4Rec adapter is based on RecBole, not established as the ADRec paper's exact implementation. No full training or reported-paper-score reproduction is claimed.

Vendored ADRec files are unchanged from the recorded commit. RecBole source/license is retained with adaptation notes. Tiny temporary checkpoints are not committed.

## Additional GRU4Rec validation

Run `python -m scripts.check_gru4rec` from the repository root. This compares the adapted model against the retained RecBole class using a minimal framework-base harness: score, loss, gradient, and optimizer-step parity for one and two layers, in training and evaluation modes. It also intentionally overfits 32 training histories and verifies checkpoint prediction equality. Results are in `results/gru4rec_validation.json`. These checks do not establish paper-level accuracy or validate the entire RecBole training pipeline.

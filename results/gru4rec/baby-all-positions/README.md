# GRU4Rec on Baby, trained on all positions (extra experiment)

Seeds 42 to 46, full mode, `cfg["gru4rec_targets"] = "all"`.

The main GRU4Rec results (`results/gru4rec/baby/`) train on one target per user: the last
training item. This experiment trains on every position in each history, like SASRec+ and
ADRec. For a GRU this matches RecBole-style prefix augmentation, because each output depends
only on the items before it. On Baby that is 57,540 training targets instead of 11,761.
Model, data, evaluation and all other settings are unchanged.

| Training targets | HR@10 | HR@20 | NDCG@10 | NDCG@20 | Best epochs |
|---|---|---|---|---|---|
| Last only (main runs) | 1.88 ± 0.16 | 3.42 ± 0.15 | 0.93 ± 0.06 | 1.31 ± 0.05 | 5–15 |
| All positions (this run) | 3.52 ± 0.06 | 5.72 ± 0.21 | 1.83 ± 0.03 | 2.39 ± 0.06 | 30–45 |
| ADRec paper, GRU4Rec | | 5.46 | | | |

Mean ± sample standard deviation over five seeds. Training on all positions brings GRU4Rec's
HR@20 from 3.42 to 5.72, close to the paper's 5.46. This supports the explanation that the gap
to the paper came from the number of training targets, not from the model code.
Code: `src/models/gru4rec.py` (`calculate_loss_all_positions`) and `src/training/kaggle.py`
(`gru4rec_targets`). The default stays `"last"`, so existing results are unchanged.

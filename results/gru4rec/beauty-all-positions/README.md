# GRU4Rec on Beauty, trained on all positions (extra experiment)

Seeds 42 to 46, full mode, `cfg["gru4rec_targets"] = "all"`. Same setup as
`results/gru4rec/baby-all-positions/`; only the dataset differs.

| Training targets | HR@10 | HR@20 | NDCG@20 | Best epochs |
|---|---|---|---|---|
| Last only (main runs) | 4.55 ± 0.20 | 7.01 | 3.05 | early |
| All positions (this run) | 8.81 ± 0.26 | 12.67 ± 0.22 | 6.05 | 70–85 |
| ADRec paper, GRU4Rec | | 12.65 | | |

Mean ± sample standard deviation over five seeds. With all-position training, GRU4Rec's
HR@20 matches the paper's value on Beauty.

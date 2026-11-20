# GRU4Rec on MovieLens-100K, trained on all positions (extra experiment)

Seeds 42 to 46, full mode, `cfg["gru4rec_targets"] = "all"`. Same setup as
`results/gru4rec/baby-all-positions/`; only the dataset differs.

| Training targets | HR@10 | HR@20 | NDCG@20 | Best epochs |
|---|---|---|---|---|
| Last only (main runs) | 4.52 ± 0.27 | 8.31 | 3.07 | early |
| All positions (this run) | 12.74 ± 0.20 | 19.89 ± 0.54 | 8.14 | 155–260 |
| ADRec (main runs) | 11.74 | 21.43 | 8.04 | |
| SASRec+ (main runs) | 11.45 | 19.72 | 7.74 | |

Mean ± sample standard deviation over five seeds.

Note: histories are truncated to the last 50 items (MovieLens users average about 58), so
training uses at most the last 50 positions per user. This is close to, but not exactly,
RecBole's full prefix augmentation, which would also train on earlier prefixes.

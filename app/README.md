# Web demo

A small web page that recommends the next items from a shopper's history. Pick an example shopper (or type your own item IDs), choose a model (ADRec, SASRec+ or GRU4Rec) or **Compare all**, press **Recommend**, and see the top-k items with their scores and the model time in milliseconds. In compare mode the lists appear side by side, and items recommended by more than one model are highlighted.

## Run it

From the repository root:

1. **Try the page without a model (test mode).** Needs only Python 3 and the dataset file.
   ```bash
   python3 -m app.server --fake
   ```
   Open http://localhost:8000. All three models appear, with random scores.

2. **Run with trained models.** Needs the shared requirements (PyTorch), so use the same Python environment you train with. Pass one `--model NAME=PATH` per model:
   ```bash
   python3 -m app.server \
     --model adrec=checkpoints/baby/best.pt \
     --model sasrec=checkpoints/baby/sasrec_best.pt \
     --model gru4rec=checkpoints/baby/gru4rec_best.pt
   ```
   With no `--model`, it serves ADRec from `checkpoints/baby/best.pt`. The server checks that each checkpoint holds the model it is named as, and that it was trained on the selected dataset.

Options: `--dataset-name baby|beauty|ml-100k` (default `baby`), `--port 8000`.

### Files you need locally (never committed)

| File | Where to put it | Where to get it |
|---|---|---|
| Dataset | `data/<dataset>/dataset.pkl` | Released ADRec data (`configs/datasets.json` lists the URL and SHA-256) |
| Model weights | `checkpoints/<dataset>/…best.pt` | Seed 42 of each model: ADRec from the team shared Drive; SASRec+ and GRU4Rec from their owners' Kaggle exports |

Both folders are ignored by Git. The server checks the dataset's SHA-256 and refuses a checkpoint trained on a different dataset.

## How it works

- `app/server.py` loads the model **once** at startup (`load_for_inference`) and ranks the full item catalog with `predict_scores`, the same functions used in evaluation. Padding (item 0) is never recommended.
- `recommend(history_item_ids, k)` returns ranked item IDs and model scores (not probabilities).
- Example shoppers show the visible history only (train + validation items). The held-out test target is never included. These five examples are placeholders until the team's validated example list is added.
- Input is validated: the history must be non-empty whole numbers from 1 to the number of items, and k must be 1 to 50. Invalid input returns a clear error message.
- Each response reports the model time in milliseconds for latency testing.
- The web server uses only the Python standard library; no extra web framework is needed.

## Notes

- Item IDs come from the released dataset and do not map to product names or images. Add metadata only after verifying the mapping.
- Keep padding, masks, vocabulary and candidate filtering consistent with evaluation.
- Use seed 42 for every model in the demo, so the comparison is consistent (see `evaluation/PROTOCOL.md`).

Ownership: Simul — ADRec inference; Anusha — input examples/validation; Mahalakshmi — UI; all — integration/testing.

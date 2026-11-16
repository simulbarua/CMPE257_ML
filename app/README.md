# Web demo

The web demo is planned; trained ADRec checkpoints are available locally under `checkpoints/`. Load the selected model once at startup and use inference mode. Accept validated example histories in that model's dataset-specific item-ID vocabulary; never include the target in the history.

Proposed shared interface: `recommend(history_item_ids, k)` returns ranked item IDs and scores. Treat scores as model scores, not probabilities unless explicitly calibrated.

Keep padding, masks, vocabulary, and candidate filtering consistent with evaluation. Do not assume encoded IDs map to product names or images. Add metadata only after verifying the mapping. Baseline comparison is optional.

Ownership: Simul — ADRec inference; Anusha — input examples/validation; Mahalakshmi — UI; all — integration/testing.

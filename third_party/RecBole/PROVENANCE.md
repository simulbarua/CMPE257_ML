Source: https://github.com/RUCAIBox/RecBole/blob/v1.2.0/recbole/model/sequential_recommender/gru4rec.py
Version: v1.2.0
Original source and license retained. Team standalone CE adapter is src/models/gru4rec.py.

Changes: remove RecBole framework configuration/Interaction dependencies, accept the team's left-padded input, move real items to the left internally for length-based recurrent output selection, expose forward/calculate_loss/calculate_score for smoke integration. Preserve embedding, bias-free GRU, dense projection and full-catalog CE architecture. BPR path omitted. This is an adapted GRU4Rec baseline, NOT verified to be the exact implementation/settings used for the ADRec paper's GRU4Rec numbers.

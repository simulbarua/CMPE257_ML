"""Standalone CE adapter based on RecBole v1.2.0 GRU4Rec.

See third_party/RecBole/{LICENSE,PROVENANCE.md,gru4rec.py}.
"""
import torch
from torch import nn

class GRU4Rec(nn.Module):
    def __init__(self, num_items, embedding_size=128, hidden_size=128,
                 num_layers=1, dropout=0.3):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items+1, embedding_size, padding_idx=0)
        self.emb_dropout = nn.Dropout(dropout)
        self.gru_layers = nn.GRU(embedding_size, hidden_size, num_layers,
                                 bias=False, batch_first=True)
        self.dense = nn.Linear(hidden_size, embedding_size)
        self.loss_fct = nn.CrossEntropyLoss()
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight)
        elif isinstance(module, nn.GRU):
            nn.init.xavier_uniform_(module.weight_hh_l0)
            nn.init.xavier_uniform_(module.weight_ih_l0)

    def forward(self, item_seq, targets=None, train_flag=True):
        lengths = (item_seq > 0).sum(1)
        if (lengths == 0).any():
            raise ValueError('GRU4Rec requires nonempty histories')
        # RecBole gathers index length-1 from right-padded sequences. Convert
        # the team's left padding, without introducing padding as recurrent history.
        packed_ids = torch.zeros_like(item_seq)
        for i, row in enumerate(item_seq):
            real = row[row > 0]
            packed_ids[i, :len(real)] = real
        out, _ = self.gru_layers(self.emb_dropout(self.item_embedding(packed_ids)))
        out = self.dense(out)
        last = out[torch.arange(len(out),device=out.device), lengths-1]
        return out, last

    def calculate_score(self, representation):
        return representation @ self.item_embedding.weight.T

    def calculate_loss(self, representation, targets):
        # RecBole CE trains the final next-item target, not per-token CE.
        return self.loss_fct(self.calculate_score(representation), targets[:, -1])

    def align_to_input(self, packed_out, item_seq):
        """Move outputs from the packed (real items first) order back to the
        left-padded input positions, so position t lines up with labels[:, t].
        A GRU reads left to right, so each output depends only on its item and
        the items before it: training every position = training every prefix."""
        seq_len = item_seq.size(1)
        lengths = (item_seq > 0).sum(1)
        positions = torch.arange(seq_len, device=item_seq.device).unsqueeze(0)
        index = (positions - (seq_len - lengths).unsqueeze(1)).clamp(min=0)
        index = index.unsqueeze(-1).expand(-1, -1, packed_out.size(-1))
        return packed_out.gather(1, index)  # padding positions are masked by labels == 0

    def calculate_loss_all_positions(self, packed_out, item_seq, labels):
        """Extra experiment: CE on every real position (RecBole-style prefix
        augmentation) instead of only the final target."""
        aligned = self.align_to_input(packed_out, item_seq)
        mask = labels > 0
        return self.loss_fct(self.calculate_score(aligned[mask]), labels[mask])

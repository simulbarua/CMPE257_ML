import unittest
import torch
from src.models.gru4rec import GRU4Rec

class GRUAdapterTests(unittest.TestCase):
    def test_padding_does_not_change_recurrent_history(self):
        torch.manual_seed(10)
        model=GRU4Rec(20,8,8,dropout=0).eval()
        a=model(torch.tensor([[0,0,2,3]]))[1]
        b=model(torch.tensor([[2,3]]))[1]
        torch.testing.assert_close(a,b)
        self.assertEqual(tuple(model.calculate_score(a).shape),(1,21))
    def test_no_history_rejected(self):
        with self.assertRaises(ValueError):GRU4Rec(20,8,8)(torch.zeros((1,3),dtype=torch.long))

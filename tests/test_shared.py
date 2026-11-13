import math
import pickle
import tempfile
import unittest
from pathlib import Path
from src.data.load import load_dataset, validate_dataset
from src.data.datasets import SequenceDataset
from src.evaluation.metrics import RankingMetrics
from src.evaluation.evaluate import evaluate

DATA = {'train': [[1, 2, 3], [4, 2]], 'val': [[4], [3]], 'test': [[5], [1]]}

class SharedTests(unittest.TestCase):
    def test_train_shift_and_padding(self):
        row = SequenceDataset(DATA, 'train', 4)[0]
        self.assertEqual(row['input_ids'], [0, 0, 1, 2])
        self.assertEqual(row['labels'], [0, 0, 2, 3])
        self.assertEqual(row['attention_mask'], [False, False, True, True])
        self.assertEqual(SequenceDataset(DATA,'train',4,'last')[0]['labels'], [0,0,0,3])

    def test_holdouts_and_truncation(self):
        val = SequenceDataset(DATA,'val',2)[0]
        test = SequenceDataset(DATA,'test',2)[0]
        self.assertEqual(val['input_ids'],[2,3])
        self.assertEqual(val['labels'],[3,4])
        self.assertEqual(test['input_ids'],[3,4])
        self.assertEqual(test['labels'],[4,5])
        self.assertEqual(SequenceDataset(DATA,'test',1)[0]['labels'],[5])
        self.assertEqual(DATA['train'][0],[1,2,3])

    def test_validation_and_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'data.pkl';p.write_bytes(pickle.dumps(DATA))
            self.assertEqual(load_dataset(p),DATA)
            with self.assertRaises(ValueError):load_dataset(p,'wrong')
            p.write_bytes(pickle.dumps(Path('not-data')))
            with self.assertRaises(ValueError):load_dataset(p)
        with self.assertRaises(ValueError):validate_dataset({'train':[[0,1]],'val':[[2]],'test':[[3]]})
        with self.assertRaises(ValueError):validate_dataset({'train':[[1,2]],'val':[],'test':[[3]]})

    def test_metrics_hand_calculation_and_batch_weight(self):
        m=RankingMetrics((1,3))
        m.update([[1,2,3],[1,2,3]],[1,2])
        m.update([[1,2,3]],[4])
        self.assertAlmostEqual(m.compute()['HR@1'],1/3)
        self.assertAlmostEqual(m.compute()['HR@3'],2/3)
        self.assertAlmostEqual(m.compute()['NDCG@3'],(1+1/math.log2(3))/3)
        with self.assertRaises(ValueError):RankingMetrics().compute()
        with self.assertRaises(ValueError):m.update([[1,1,2]],[1])

    def test_evaluation_no_target_passed_and_padding_excluded(self):
        ds=SequenceDataset(DATA,'test',3)
        received=[]
        def predict(histories,masks):
            received.extend(histories)
            # Padding's huge score must not influence rankings.
            return [[999,5,4,3,2,1] for h in histories]
        a=evaluate(ds,predict,5,batch_size=1,ks=(1,5))
        b=evaluate(ds,predict,5,batch_size=2,ks=(1,5))
        self.assertEqual(a,b)
        self.assertEqual(a['metrics']['HR@1'],.5)
        self.assertEqual(a['metrics']['HR@5'],1)
        self.assertEqual(received[0],[2,3,4])
        self.assertEqual(received[1],[4,2,3])

    def test_torch_adapter(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch optional dependency not installed")
        from src.data.torch_adapter import make_loader
        ds=SequenceDataset(DATA,'train',4)
        batch=next(iter(make_loader(ds,batch_size=2)))
        self.assertEqual(tuple(batch['input_ids'].shape),(2,4))
        self.assertEqual(batch['input_ids'].dtype,torch.long)
        self.assertEqual(batch['attention_mask'].dtype,torch.bool)
        self.assertEqual(batch['labels'][0].tolist(),[0,0,2,3])
        a=list(make_loader(ds,1,True,seed=7))
        b=list(make_loader(ds,1,True,seed=7))
        self.assertEqual([x['user_index'].item() for x in a],[x['user_index'].item() for x in b])

    def test_score_validation(self):
        ds=SequenceDataset(DATA,'test')
        with self.assertRaises(ValueError):evaluate(ds,lambda h,m:[[1]]*len(h),5,ks=(1,))
        with self.assertRaises(ValueError):evaluate(ds,lambda h,m:[[0,float('nan'),0,0,0,0]]*len(h),5,ks=(1,))

if __name__ == '__main__':unittest.main()

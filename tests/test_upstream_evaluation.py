import importlib.util
from pathlib import Path
import unittest
import numpy as np
import torch
from src.evaluation.evaluate import evaluate

spec=importlib.util.spec_from_file_location('reference_metrics',Path(__file__).resolve().parents[1]/'third_party/ADRec/src/metrics.py')
ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)

class ParityTests(unittest.TestCase):
    def compare(self,scores,targets,batch_size):
        ds=[{'input_ids':[i+1],'attention_mask':[True],'target_id':v} for i,v in enumerate(targets)]
        def predict(h,m):return scores[[r[0]-1 for r in h]]
        got=evaluate(ds,predict,scores.shape[1]-1,batch_size=batch_size,ks=(1,3))
        sums={}
        for start in range(0,len(ds),batch_size):
            vals=ref.hrs_and_ndcgs_k(scores[start:start+batch_size],torch.tensor(targets[start:start+batch_size]).reshape(-1,1),[1,3])
            for k,v in vals.items():sums.setdefault(k,[]).append(v)
        expected={k:round(float(np.mean(v))*100,4) for k,v in sums.items()}
        self.assertEqual(got['metrics'],expected)
        self.assertEqual(got['units'],'percent')
        return got

    def test_padding_ties_partial_batch(self):
        self.compare(torch.tensor([[99.,3,2,1,0],[0,0,0,0,0],[0,4,3,2,1]]),[1,2,1],2)
    def test_unequal_batches_not_user_average(self):
        got=self.compare(torch.tensor([[0.,5,2,1],[0.,5,2,1],[0.,5,2,1]]),[1,1,3],2)
        self.assertEqual(got['metrics']['HR@1'],50.)
    def test_random_dtypes(self):
        for dtype in (torch.float32,torch.float64):
            torch.manual_seed(1)
            self.compare(torch.randn(11,30,dtype=dtype),list(range(1,12)),4)
    def test_no_users(self):
        with self.assertRaises(ValueError):evaluate([],lambda h,m:[],30)

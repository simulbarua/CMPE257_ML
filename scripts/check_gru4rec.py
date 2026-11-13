"""Local GRU4Rec compatibility checks; not paper benchmark reproduction.

Reference class body is read unchanged from vendored RecBole source. A minimal
base supplies only RecBole field names, catalog size and gather_indexes; this
checks model math, not the RecBole trainer/data augmentation framework.
"""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import torch
from torch import nn
from torch.nn.init import xavier_uniform_, xavier_normal_
from src.models.gru4rec import GRU4Rec
from src.data import load_dataset, SequenceDataset
from src.data.load import BABY_SHA256
from src.data.torch_adapter import make_loader
from src.evaluation import evaluate

class ReferenceBase(nn.Module):
    def __init__(self, config, dataset):
        super().__init__()
        self.n_items=dataset.n_items
        self.ITEM_SEQ='item_seq';self.ITEM_SEQ_LEN='length';self.POS_ITEM_ID='target'
    def gather_indexes(self, output, index):
        return output.gather(1,index.view(-1,1,1).expand(-1,1,output.size(-1))).squeeze(1)

def reference_class():
    tree=ast.parse(Path('third_party/RecBole/gru4rec.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GRU4Rec')
    namespace=dict(torch=torch,nn=nn,xavier_uniform_=xavier_uniform_,xavier_normal_=xavier_normal_,SequentialRecommender=ReferenceBase)
    exec(compile(ast.Module(body=[cls],type_ignores=[]),'reference_gru4rec','exec'),namespace)
    return namespace['GRU4Rec']

def run():
    torch.set_num_threads(1);torch.manual_seed(42)
    refclass=reference_class();parity=[]
    for layers in (1,2):
        cfg=dict(embedding_size=16,hidden_size=24,loss_type='CE',num_layers=layers,dropout_prob=.3)
        model=GRU4Rec(30,16,24,layers,.3)
        ref=refclass(cfg,SimpleNamespace(n_items=31));ref.load_state_dict(model.state_dict())
        left=torch.tensor([[0,0,2,3,4],[0,0,0,0,5],[1,2,3,4,6]])
        right=torch.tensor([[2,3,4,0,0],[5,0,0,0,0],[1,2,3,4,6]])
        lengths=torch.tensor([3,1,5]);targets=torch.tensor([7,8,9])
        interaction={'item_seq':right,'length':lengths,'target':targets}
        for training in (False,True):
            model.train(training);ref.train(training)
            model.zero_grad();ref.zero_grad()
            torch.manual_seed(123);out=model(left)[1];scores=model.calculate_score(out)
            loss=model.calculate_loss(out,targets[:,None])
            torch.manual_seed(123);expected=ref.full_sort_predict(interaction)
            torch.testing.assert_close(scores,expected,rtol=0,atol=0)
            torch.manual_seed(123);expected_loss=ref.calculate_loss(interaction)
            torch.testing.assert_close(loss,expected_loss,rtol=0,atol=0)
            loss.backward();expected_loss.backward()
            for p,q in zip(model.parameters(),ref.parameters()):
                torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
            a=torch.optim.Adam(model.parameters(),lr=.001);b=torch.optim.Adam(ref.parameters(),lr=.001)
            a.step();b.step()
            for p,q in zip(model.parameters(),ref.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)
            parity.append({'layers':layers,'training':training,'scores_loss_gradients_update':'exact match'})
    full=load_dataset('data/baby/dataset.pkl',BABY_SHA256)
    data={k:v[:32] for k,v in full.items()}
    train=SequenceDataset(data,'train',50,'last')
    batch=next(iter(make_loader(train,32)))
    model=GRU4Rec(4731,32,32,dropout=0)
    opt=torch.optim.Adam(model.parameters(),lr=.02)
    losses=[]
    # Deliberate overfit of a tiny real-data batch detects learning-path failures.
    for step in range(150):
        model.train();opt.zero_grad()
        out=model(batch['input_ids'])[1]
        loss=model.calculate_loss(out,batch['labels'])
        assert torch.isfinite(loss)
        loss.backward();opt.step();losses.append(float(loss.detach()))
    model.eval()
    def predict(h,m):
        with torch.no_grad():return model.calculate_score(model(torch.tensor(h))[1])
    train_scores=predict(batch['input_ids'].tolist(),None)
    accuracy=float((train_scores.argmax(1)==batch['target_id']).float().mean())
    assert losses[-1] < losses[0]*.2 and accuracy >= .9
    eval_results={s:evaluate(SequenceDataset(data,s,50,'last'),predict,4731,batch_size=13) for s in ('val','test')}
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)/'checkpoint.pt';torch.save(model.state_dict(),p)
        clone=GRU4Rec(4731,32,32,dropout=0).eval();clone.load_state_dict(torch.load(p,weights_only=True))
        with torch.no_grad():restored=clone.calculate_score(clone(batch['input_ids'])[1])
        torch.testing.assert_close(train_scores,restored,rtol=0,atol=0)
    results={'reference':'RecBole v1.2.0 class with minimal framework-base harness; not full RecBole pipeline',
             'reference_parity':parity,'device':'cpu','training_users':32,'steps':150,
             'initial_loss':losses[0],'final_loss':losses[-1],'training_top1_accuracy':accuracy,
             'checkpoint_predictions':'exact match','evaluation':eval_results,
             'limitations':['Small intentional overfit, not evidence of generalization','No Kaggle GPU check','ADRec paper exact GRU4Rec setup remains unknown','One final CE target per training sequence; no automatic prefix expansion']}
    Path('results/gru4rec_validation.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))
if __name__=='__main__':run()

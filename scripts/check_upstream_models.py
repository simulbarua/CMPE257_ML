"""CPU integration checks, not trained benchmark results. Run from repository root."""
import json,sys,tempfile,os,copy,importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'third_party/ADRec/src'))
from sasrec import SASRec
from model import Att_Diffuse_model
from pcgrad import PCGrad
from metrics import hrs_and_ndcgs_k
from utils import TrainDataset,ValDataset,TestDataset
from src.data.load import load_dataset,BABY_SHA256
from src.data.datasets import SequenceDataset
from src.data.torch_adapter import make_loader
from src.evaluation.evaluate import evaluate
from src.utils.reproducibility import seed_everything
from src.models.gru4rec import GRU4Rec

def run():
    torch.set_num_threads(1);seed_everything(2025)
    full=load_dataset(ROOT/'data/baby/dataset.pkl',BABY_SHA256)
    data={k:v[:9] for k,v in full.items()}
    args=SimpleNamespace(**yaml.safe_load((ROOT/'third_party/ADRec/src/config.yaml').read_text()))
    args.dataset='baby';args.item_num=4731;args.device='cpu';args.hidden_size=16
    args.max_len=50;args.diffusion_steps=4;args.pretrained=False;args.freeze_emb=False
    datasets={s:SequenceDataset(data,s,args.max_len) for s in data}
    originals={'train':TrainDataset(data['train'],args.max_len,True),
               'val':ValDataset(data['train'],data['val'],args.max_len),
               'test':TestDataset(data['train'],data['val'],data['test'],args.max_len)}
    for s,ds in datasets.items():
        for i in range(len(ds)):
            x,y=originals[s][i]
            assert x.tolist()==ds[i]['input_ids'] and y.tolist()==ds[i]['labels']
    report={'kind':'CPU integration smoke check, not benchmark reproduction','users':9,
            'hidden_size':16,'diffusion_steps':4,'steps_per_stage':2,'dataset_parity':True,'models':{}}
    def train(model):
        model.train();opt=PCGrad(torch.optim.Adam(model.parameters(),lr=.001),args)
        before={k:v.clone() for k,v in model.state_dict().items()}
        losses=[]
        for b in list(make_loader(datasets['train'],4))[:2]:
            opt.zero_grad();out=model(b['input_ids'],b['labels'],train_flag=True)
            ce=model.calculate_loss(out[1] if isinstance(model, GRU4Rec) else out[0],b['labels']);parts=[ce]
            if len(out)>2:parts.append(out[2])
            assert all(torch.isfinite(v).all() for v in parts)
            opt.pc_backward(parts)
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            opt.step();losses.append(float(ce.detach()))
        assert any(not torch.equal(before[k],v) for k,v in model.state_dict().items())
        return losses
    def check(model,name):
        model.eval()
        def predict(histories,masks):
            x=torch.tensor(histories)
            # Shift only the visible history. Dummy final ID supplies mask/shape,
            # not the held-out answer. Upstream ADRec replaces that final embedding with noise.
            dummy=torch.cat([x[:,1:],torch.ones((len(x),1),dtype=torch.long)],1)
            dummy = dummy.masked_fill(x == 0, 0)
            return model.calculate_score(model(x,dummy,train_flag=False)[1])
        result={}
        for split in ('val','test'):
            ds=datasets[split];sc=[];expected={}
            with torch.no_grad():
                for b in make_loader(ds,4):
                    torch.manual_seed(123)
                    actual=model(b['input_ids'],b['labels'],train_flag=False)[1]
                    torch.manual_seed(123)
                    scores=predict(b['input_ids'].tolist(),b['attention_mask'].tolist())
                    assert torch.equal(scores,model.calculate_score(actual)), 'Target leakage or conditioning mismatch'
                    sc.append(scores)
                    vals=hrs_and_ndcgs_k(scores,b['target_id'].reshape(-1,1),[5,10,20])
                    for k,v in vals.items():expected.setdefault(k,[]).append(v)
            scores=torch.cat(sc);offset=0
            def cached(h,m):
                nonlocal offset
                ans=scores[offset:offset+len(h)];offset+=len(h);return ans
            got=evaluate(ds,cached,4731,batch_size=4)
            ref={k:round(float(np.mean(v))*100,4) for k,v in expected.items()}
            assert got['metrics']==ref
            result[split]={'metrics':ref,'parity':True,'label_independence':True}
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'weights.pt';torch.save(model.state_dict(),path)
            clone=copy.deepcopy(model);clone.load_state_dict(torch.load(path,weights_only=True))
            torch.manual_seed(123);a=predict([datasets['test'][0]['input_ids']],None)
            # Check reloaded parameter values exactly.
            assert all(torch.equal(v,clone.state_dict()[k]) for k,v in model.state_dict().items())
        report['models'][name]=result
    args.model='sasrec';sas=SASRec(args);loss=train(sas);check(sas,'SASRec+');report['models']['SASRec+']['losses']=loss
    args.model='pretrain';pre=SASRec(args);preloss=train(pre)
    # Exercise the original checkpoint loading path in a temporary working directory.
    with tempfile.TemporaryDirectory() as t:
        old=os.getcwd();os.chdir(t)
        try:
            Path('saved/pretrain/baby').mkdir(parents=True)
            torch.save(pre.state_dict(), 'saved/pretrain/baby/pretrain.pth')
            args.model='adrec';args.pretrained=True;args.freeze_emb=True
            ad=Att_Diffuse_model(args)
        finally:os.chdir(old)
    weight=ad.item_embedding.weight.detach().clone()
    warm=train(ad);assert torch.equal(weight,ad.item_embedding.weight)
    ad.item_embedding.weight.requires_grad_(True)
    fine=train(ad);assert not torch.equal(weight,ad.item_embedding.weight)
    check(ad,'ADRec');report['models']['ADRec'].update(pretrain_losses=preloss,warmup_losses=warm,finetune_losses=fine,freeze_unfreeze_verified=True)
    args.model='gru4rec'
    gru=GRU4Rec(4731,embedding_size=16,hidden_size=16)
    gruloss=train(gru);check(gru,'GRU4Rec_RecBole_adapter')
    report['models']['GRU4Rec_RecBole_adapter'].update(losses=gruloss, source='RecBole v1.2.0 CE adaptation; not verified as the ADRec paper baseline')
    (ROOT/'results/local_model_smoke.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':run()

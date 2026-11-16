"""Check the fixed-author-checkpoint sweep, paired comparisons, and resume guards."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
import zipfile

import torch
from src.training.kaggle import make_config

NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebooks/adrec_author_embeddings_kaggle.ipynb'
CELLS = {c['id']: ''.join(c['source']) for c in json.loads(NOTEBOOK.read_text())['cells']}


class AuthorEmbeddingTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.pre = self.root / 'authors.pth'
        self.pre.write_bytes(b'one fixed author checkpoint')
        self.ns = {'Path':Path, 'torch':torch, 'WORK_ROOT':self.root, 'DEVICE':'cpu',
                   'DATASET_PATH':self.root/'dataset.pkl', 'INPUTS':self.root/'input',
                   'AUTHOR_CHECKPOINT':self.pre, 'INPUT_MANIFEST':{'code_manifest_sha256':'test'},
                   'SEEDS':list(range(42,47)), 'WARMUPS':[5,0], 'RUN_FULL':True, 'RESUME_ROOT':None,
                   'SMOKE_CHECK':{'passed':True}, 'SOURCE_MIGRATIONS':[]}
        for id in ['input-helpers','experiment-helpers']:
            exec(compile(CELLS[id], id, 'exec'),self.ns)
        baseline=self.ns['INPUTS']/'baseline'
        baseline.mkdir(parents=True)
        (baseline/'original.json').write_text('{"original": true}\n')
        self.ns['AUTHOR_SPEC']={'sha256':self.ns['sha256'](self.pre)}
        refs={}
        for seed in self.ns['SEEDS']:
            cfg=make_config('adrec','full');cfg['seed']=seed
            refs[seed]={'config':cfg,'reports':{w:self.report(dict(cfg,warmup_epochs=w),seed-40+(1 if w==5 else 0)) for w in [5,0]}}
        self.ns['REFERENCES']=refs
        self.mock=Mock(side_effect=self.trainer)
        self.ns['train']=self.mock

    def report(self,cfg,score):
        return {'model':'adrec','mode':cfg['mode'],'users':11761,'best_epoch':10,
                'epochs_completed':31,'early_stopped':True,
                'test':{'users':11761,'protocol':'adrec_upstream','units':'percent',
                        'metrics':{m:score for m in self.ns['METRICS']}}}

    def trainer(self,cfg,dataset,output,**kwargs):
        self.assertEqual(kwargs['pretrained_checkpoint'],self.pre)
        self.assertTrue(kwargs['finalize'])
        report=self.report(cfg,cfg['seed']-39+(2 if cfg['warmup_epochs']==5 else 0))
        for name,obj in [('config.json',cfg),('results.json',report),('history.json',[]),('environment.json',{})]:
            self.ns['write_json'](output/name,obj)
        (output/'requirements-lock.txt').write_text('test\n')
        for name in ['best.pt','latest.pt']:(output/name).write_bytes(b'weights')
        return report

    def test_config_isolation_and_smoke_uses_real_five_epoch_schedule(self):
        base=self.ns['REFERENCES'][42]['config']
        self.assertEqual(self.ns['variant_config'](base,5),base)
        zero=self.ns['variant_config'](base,0)
        self.assertEqual({k for k in base if zero[k]!=base[k]},{'warmup_epochs'})
        for w in [5,0]:
            cfg=self.ns['variant_config'](base,w,smoke=True)
            self.assertEqual(cfg['warmup_epochs'],w)
            self.assertEqual(cfg['epochs'],6)
            self.assertEqual(cfg['hidden_size'],128)
            self.assertEqual(cfg['diffusion_steps'],32)
        with self.assertRaises(ValueError):self.ns['variant_config'](base,3)

    def test_ten_runs_share_checkpoint_and_completed_runs_are_reused(self):
        exec(compile(CELLS['sweep'],'sweep','exec'),self.ns)
        pairs=[(c.args[0]['seed'],c.args[0]['warmup_epochs']) for c in self.mock.call_args_list]
        self.assertEqual(pairs,[(s,w) for s in range(42,47) for w in [5,0]])
        self.assertTrue(all(c.args[0]['model']=='adrec' for c in self.mock.call_args_list))
        self.mock.reset_mock()
        exec(compile(CELLS['sweep'],'sweep','exec'),self.ns)
        self.mock.assert_not_called()
        completion=self.ns['read_json'](self.ns['RESULTS_DIR']/'completion.json')
        self.assertTrue(completion['complete'])
        self.assertEqual(len(completion['completed_runs']),10)

    def test_pairing_statistics_and_no_checkpoint_in_result_archive(self):
        for s,w in [(42,5),(42,0),(44,5),(44,0),(43,5)]:
            cfg=self.ns['variant_config'](self.ns['REFERENCES'][s]['config'],w)
            self.ns['ensure_run'](cfg,self.ns['RUNS_ROOT']/f'authors_warmup{w}_seed{s}')
        rows=self.ns['export_results']()
        self.assertEqual(len(rows),8)
        self.assertEqual({r['seed'] for r in rows},{42,44})
        summary=self.ns['summarize'](rows)
        five=[r for r in summary if r['embedding_source']=='authors' and r['warmup_epochs']==5]
        self.assertEqual(five[0]['HR@20'],6.0)
        self.assertEqual(five[1]['HR@20'],1.4142)
        self.assertEqual(five[0]['seeds'],'42,44')
        with (self.ns['RESULTS_DIR']/'paired_differences.csv').open() as f:
            diffs=list(csv.DictReader(f))
        self.assertTrue(all(float(r['HR@20'])==-2 for r in diffs if r['comparison']=='authors_warmup0_minus_warmup5'))
        completion=self.ns['read_json'](self.ns['RESULTS_DIR']/'completion.json')
        self.assertFalse(completion['complete'])
        self.assertEqual(len(completion['completed_runs']),5)
        self.ns['save_archives']()
        with zipfile.ZipFile(self.ns['ARCHIVE_ROOT']/'adrec_baby_author_embeddings_results.zip') as z:
            self.assertFalse(any(n.endswith(('.pt','.pth')) for n in z.namelist()))
        bad=self.ns['RUNS_ROOT']/'authors_warmup5_seed42/config.json'
        self.ns['write_json'](bad,self.ns['variant_config'](self.ns['REFERENCES'][42]['config'],5,smoke=True))
        with self.assertRaises(ValueError):self.ns['export_results']()

    def test_resume_copy_and_changed_inputs_are_rejected(self):
        cfg=self.ns['variant_config'](self.ns['REFERENCES'][42]['config'],5)
        prior=self.root/'prior'
        self.ns['ensure_run'](cfg,prior)
        output=self.ns['RUNS_ROOT']/'authors_warmup5_seed42'
        self.mock.reset_mock()
        self.ns['ensure_run'](cfg,output,prior)
        self.mock.assert_not_called()
        (output/'results.json').unlink()
        self.ns['ensure_run'](cfg,output)
        self.assertEqual(self.mock.call_args.kwargs['resume'],output/'latest.pt')
        with self.assertRaises(ValueError):self.ns['ensure_run'](dict(cfg,warmup_epochs=0),output)
        self.ns['INPUT_MANIFEST']['code_manifest_sha256']='changed'
        with self.assertRaises(ValueError):self.ns['ensure_run'](cfg,output)
        self.ns['INPUT_MANIFEST']['code_manifest_sha256']='test'
        self.pre.write_bytes(b'new embedding file')
        with self.assertRaises(ValueError):self.ns['ensure_run'](cfg,output)

    def test_checkpoint_hash_shape_and_finiteness_checks(self):
        torch.save({'item_embedding.weight':torch.ones(3,4)},self.pre)
        def spec():
            raw=self.pre.read_bytes()
            return {'size_bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
                    'git_blob_sha1':hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest(),
                    'state_key':'item_embedding.weight','embedding_shape':[3,4],'dtype':'torch.float32'}
        correct=spec()
        self.assertEqual(tuple(self.ns['validate_author_checkpoint'](self.pre,correct).shape),(3,4))
        with self.assertRaises(ValueError):self.ns['validate_author_checkpoint'](self.pre,dict(correct,embedding_shape=[4,4]))
        with self.assertRaises(ValueError):self.ns['validate_author_checkpoint'](self.pre,dict(correct,sha256='wrong'))
        torch.save({'item_embedding.weight':torch.full((3,4),float('nan'))},self.pre)
        with self.assertRaises(ValueError):self.ns['validate_author_checkpoint'](self.pre,spec())
        self.assertEqual(len(self.ns['summarize']([{'embedding_source':'authors','warmup_epochs':5,'seed':42,
                         **{m:1 for m in self.ns['METRICS']}}])),1)


if __name__=='__main__':unittest.main()

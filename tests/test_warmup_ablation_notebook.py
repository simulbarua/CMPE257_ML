"""Verify warm-up isolation, paired exports, and notebook resume behavior."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
import zipfile

from src.training.kaggle import make_config

NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebooks/adrec_warmup_ablation_kaggle.ipynb'
CELLS = {c['id']: ''.join(c['source']) for c in json.loads(NOTEBOOK.read_text())['cells']}


class WarmupNotebookTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.ns = {'Path': Path, 'WORK_ROOT': self.root, 'DEVICE': 'cpu',
                   'DATASET_PATH': self.root / 'dataset.pkl', 'INPUT_MANIFEST': {'code_manifest_sha256': 'test'},
                   'SEEDS': list(range(42, 47)), 'BASELINE': self.root / 'baseline',
                   'RUN_FULL': True, 'RESUME_ROOT': None, 'SMOKE_CHECK': {'passed': True}}
        for id in ['input-helpers', 'experiment-helpers']:
            exec(compile(CELLS[id], id, 'exec'), self.ns)
        self.ns['BASELINE'].mkdir()
        (self.ns['BASELINE'] / 'original.json').write_text('{"unchanged": true}\n')

    def config(self, seed):
        cfg = make_config('adrec', 'full')
        cfg['seed'] = seed
        return cfg

    def report(self, cfg, score):
        return {'model': 'adrec', 'mode': cfg['mode'], 'users': 11761, 'best_epoch': 5,
                'epochs_completed': 26, 'early_stopped': True,
                'test': {'users': 11761, 'protocol': 'adrec_upstream', 'units': 'percent',
                         'metrics': {m: score for m in self.ns['METRICS']}}}

    def references(self):
        refs = {}
        for seed in self.ns['SEEDS']:
            pre = self.root / f'pretrain_seed{seed}.pt'
            pre.write_bytes(f'original pretrained weights {seed}'.encode())
            cfg = self.config(seed)
            refs[seed] = {'config': cfg, 'report': self.report(cfg, seed - 40), 'pretrain': pre}
        self.ns.update(REFERENCES=refs, SOURCE_MIGRATIONS=[])
        return refs

    def trainer(self, cfg, dataset, output, **kwargs):
        seed = cfg['seed']
        self.assertEqual(kwargs['pretrained_checkpoint'], self.ns['REFERENCES'][seed]['pretrain'])
        self.assertTrue(kwargs['finalize'])
        report = self.report(cfg, seed - 41)
        for name, obj in [('config.json', cfg), ('results.json', report),
                          ('history.json', [{'epoch': 0, 'embedding_trainable': True}]),
                          ('environment.json', {'test': True})]:
            self.ns['write_json'](output / name, obj)
        (output / 'requirements-lock.txt').write_text('test\n')
        (output / 'best.pt').write_bytes(b'best')
        (output / 'latest.pt').write_bytes(b'latest')
        return report

    def test_only_warmup_changes_full_config_and_smoke_retains_embedding_shape(self):
        base = self.config(42)
        changed = self.ns['variant_config'](base)
        self.assertEqual({k for k in base if base[k] != changed[k]}, {'warmup_epochs'})
        self.assertEqual(base['warmup_epochs'], 5)
        self.assertEqual(changed['warmup_epochs'], 0)
        smoke = self.ns['variant_config'](base, smoke=True)
        for k in ['seed', 'hidden_size', 'diffusion_steps', 'max_len', 'dataset_sha256']:
            self.assertEqual(smoke[k], base[k])
        self.assertEqual(smoke['user_limit'], 9)

    def test_sweep_uses_matching_pretraining_and_resumes_without_duplicate_runs(self):
        refs = self.references()
        mock = Mock(side_effect=self.trainer)
        self.ns['train'] = mock
        exec(compile(CELLS['sweep'], 'sweep', 'exec'), self.ns)
        self.assertEqual([c.args[0]['seed'] for c in mock.call_args_list], list(range(42, 47)))
        self.assertTrue(all(c.args[0]['model'] == 'adrec' and c.args[0]['warmup_epochs'] == 0 for c in mock.call_args_list))
        mock.reset_mock()
        exec(compile(CELLS['sweep'], 'sweep', 'exec'), self.ns)
        mock.assert_not_called()
        run = self.ns['RUNS_ROOT'] / 'adrec_warmup0_seed42'
        (run / 'results.json').unlink()
        self.ns['ensure_run'](self.ns['variant_config'](refs[42]['config']), refs[42]['pretrain'], run)
        self.assertEqual(mock.call_args.kwargs['resume'], run / 'latest.pt')
        with self.assertRaises(ValueError):
            self.ns['ensure_run'](dict(self.ns['variant_config'](refs[42]['config']), warmup_epochs=5), refs[42]['pretrain'], run)
        refs[42]['pretrain'].write_bytes(b'different embeddings')
        with self.assertRaises(ValueError):
            self.ns['ensure_run'](self.ns['variant_config'](refs[42]['config']), refs[42]['pretrain'], run)

    def test_partial_export_pairs_same_seeds_and_keeps_weights_out(self):
        refs = self.references()
        self.ns['train'] = self.trainer
        for seed in [42, 44]:
            self.ns['ensure_run'](self.ns['variant_config'](refs[seed]['config']), refs[seed]['pretrain'],
                                  self.ns['RUNS_ROOT'] / f'adrec_warmup0_seed{seed}')
        rows = self.ns['export_results'](refs, [])
        self.assertEqual(len(rows), 4)
        self.assertEqual({r['seed'] for r in rows}, {42, 44})
        folder = self.ns['RESULTS_DIR']
        with (folder / 'accuracy_summary.csv').open() as f:
            summary = list(csv.DictReader(f))
        zero = [r for r in summary if r['variant'] == 'warmup0']
        self.assertEqual([r['statistic'] for r in zero], ['mean', 'std (sample, n-1)'])
        self.assertEqual(float(zero[0]['HR@20']), 2.0)
        self.assertEqual(float(zero[1]['HR@20']), 1.4142)
        self.assertEqual(zero[0]['seeds'], '42,44')
        with (folder / 'paired_differences.csv').open() as f:
            self.assertTrue(all(float(r['HR@20']) == -1 for r in csv.DictReader(f)))
        self.assertFalse(self.ns['read_json'](folder / 'completion.json')['complete'])
        self.assertEqual((folder / 'baseline/original.json').read_bytes(), (self.ns['BASELINE'] / 'original.json').read_bytes())
        self.ns['save_archives']()
        with zipfile.ZipFile(self.ns['ARCHIVE_ROOT'] / 'adrec_baby_warmup0_results.zip') as z:
            self.assertFalse(any(n.endswith('.pt') for n in z.namelist()))
        # Smoke data must never be exported into a full comparison.
        p = self.ns['RUNS_ROOT'] / 'adrec_warmup0_seed42/config.json'
        self.ns['write_json'](p, self.ns['variant_config'](refs[42]['config'], smoke=True))
        with self.assertRaises(ValueError):
            self.ns['export_results'](refs, [])

    def test_single_seed_has_no_standard_deviation_and_zip_paths_are_checked(self):
        rows = [{'variant': 'warmup0', 'warmup_epochs': 0, 'seed': 42,
                 **{m: 1.0 for m in self.ns['METRICS']}}]
        self.assertEqual(len(self.ns['summarize_rows'](rows)), 1)
        archive = self.root / 'bad.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr('../escape', 'bad')
        with self.assertRaises(ValueError):
            self.ns['unpack'](archive, self.root / 'extracted')
        self.assertFalse((self.root / 'escape').exists())


if __name__ == '__main__':
    unittest.main()

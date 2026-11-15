"""Contracts for the notebook's checkpoint reuse and timing comparison."""
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock
import zipfile

import torch

NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebooks/adrec_diffusion_steps_kaggle.ipynb'
CELLS = {c['id']: ''.join(c['source']) for c in json.loads(NOTEBOOK.read_text())['cells']}


def helpers():
    namespace = dict(Path=Path, hashlib=hashlib, json=json, shutil=shutil,
                     torch=torch, zipfile=zipfile, BENCHMARK_SEED=2025,
                     DATASET_PATH='unused', DEVICE='cpu')
    exec(compile(CELLS['helpers'], str(NOTEBOOK), 'exec'), namespace)
    return namespace


class DiffusionNotebookTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ns = helpers()

    def test_archive_layouts_and_traversal_rejection(self):
        outer_path = self.root / 'adrec_all_results_and_models.zip'
        def nested(files):
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, 'w') as z:
                for name, value in files.items():
                    z.writestr(name, value)
            return buffer.getvalue()
        with zipfile.ZipFile(outer_path, 'w') as z:
            z.writestr('adrec_results.zip', nested({'metadata.json': '{}'}))
            z.writestr('adrec_full_seed42.zip', nested({'pretrain/best.pt': 'pretrain', 'adrec/best.pt': 'adrec'}))
        refs = self.ns['locate_references'](None, self.root, self.root / 'cache', [42])
        self.assertEqual((refs[42]['run'] / 'pretrain/best.pt').read_text(), 'pretrain')
        self.assertTrue((refs[42]['metadata'] / 'metadata.json').is_file())
        extracted = self.root / 'extracted'
        self.ns['unpack'](outer_path, extracted)
        refs2 = self.ns['locate_references'](extracted, self.root, self.root / 'cache2', [42])
        self.assertEqual((refs2[42]['run'] / 'adrec/best.pt').read_text(), 'adrec')
        folder_layout = self.root / 'folders'
        shutil.copytree(refs[42]['run'], folder_layout / 'adrec_full_seed42')
        shutil.copytree(refs[42]['metadata'], folder_layout / 'adrec_results')
        refs3 = self.ns['locate_references'](folder_layout, self.root, self.root / 'cache3', [42])
        self.assertEqual(refs3[42]['run'], (folder_layout / 'adrec_full_seed42').resolve())
        bad = self.root / 'bad.zip'
        with zipfile.ZipFile(bad, 'w') as z:
            z.writestr('../outside.txt', 'bad')
        with self.assertRaises(ValueError):
            self.ns['unpack'](bad, self.root / 'safe')
        self.assertFalse((self.root / 'outside.txt').exists())

    def test_variants_only_change_steps_and_resume_preserves_embedding_identity(self):
        base = {'model': 'adrec', 'seed': 43, 'diffusion_steps': 32,
                'mode': 'full', 'hidden_size': 128, 'epochs': 500}
        cfg = self.ns['variant_config'](base, 8, 'full')
        self.assertEqual({k for k in base if cfg[k] != base[k]}, {'diffusion_steps'})
        self.assertEqual(self.ns['variant_config'](base, 16, 'smoke')['hidden_size'], 128)
        pre = self.root / 'pretrain.pt'
        pre.write_bytes(b'original embeddings')
        out = self.root / 'run'
        report = {'early_stopped': True, 'epochs_completed': 26, 'test': {'metrics': {'HR@20': 5}}}
        def fake_train(actual_cfg, dataset, output, **kwargs):
            self.assertEqual(actual_cfg, cfg)
            self.assertEqual(kwargs['pretrained_checkpoint'], pre)
            for name in ['latest.pt', 'best.pt']:
                (output / name).write_bytes(b'weights')
            self.ns['write_json'](output / 'config.json', cfg)
            self.ns['write_json'](output / 'results.json', report)
            return report
        trainer = Mock(side_effect=fake_train)
        self.ns['train'] = trainer
        self.ns['ensure_variant'](cfg, pre, out)
        self.ns['ensure_variant'](cfg, pre, out)
        self.assertEqual(trainer.call_count, 1)
        (out / 'results.json').unlink()
        self.ns['ensure_variant'](cfg, pre, out)
        self.assertEqual(trainer.call_args.kwargs['resume'], out / 'latest.pt')
        with self.assertRaises(ValueError):
            self.ns['ensure_variant'](dict(cfg, diffusion_steps=16), pre, out)
        pre.write_bytes(b'different embeddings')
        with self.assertRaises(ValueError):
            self.ns['ensure_variant'](cfg, pre, out)

    def test_summary_csv_and_timing_scope(self):
        summary = self.ns['summarize_rows']([
            {'diffusion_steps': 8, 'seed': 42, 'HR@20': 4},
            {'diffusion_steps': 8, 'seed': 43, 'HR@20': 6},
        ], ['diffusion_steps'], ['HR@20'])[0]
        self.assertEqual(summary['HR@20_mean'], 5)
        self.assertAlmostEqual(summary['HR@20_std'], 2**0.5)
        self.assertEqual(summary['n_runs'], 2)
        one = {'diffusion_steps': 8, 'seed': 42, 'HR@20': 4}
        with self.assertRaises(ValueError):
            self.ns['summarize_rows']([one, one], ['diffusion_steps'], ['HR@20'])
        self.assertIsNone(self.ns['summarize_rows']([one], ['diffusion_steps'], ['HR@20'])[0]['HR@20_std'])
        csv_path = self.root / 'summary.csv'
        self.ns['save_csv'](csv_path, [summary])
        self.assertEqual(len(csv_path.read_text().splitlines()), 2)
        model = torch.nn.Linear(2, 21)
        predictor = Mock(side_effect=lambda model, x: torch.ones((len(x), 21)))
        self.ns['predict_scores'] = predictor
        self.ns['seed_everything'] = Mock()
        batches = [torch.ones((4, 2), dtype=torch.long)]
        timings = self.ns['benchmark'](model, batches, 2, 3, 42)
        self.assertEqual(predictor.call_count, 5)
        self.assertEqual(len(timings['samples_ms']), 3)
        self.assertEqual(timings['batch_size'], 4)
        self.assertAlmostEqual(timings['users_per_second'], 4000 / timings['mean_batch_ms'])
        dataset = [{'input_ids': [0, i + 1]} for i in range(9)]
        a = self.ns['timing_batches'](dataset, 8, 4, 'cpu')
        b = self.ns['timing_batches'](dataset, 8, 4, 'cpu')
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(a, b)))


if __name__ == '__main__':
    unittest.main()

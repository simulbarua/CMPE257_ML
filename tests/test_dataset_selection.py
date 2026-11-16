"""Dataset identity, backward compatibility, and notebook export isolation."""
import hashlib
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.data.catalog import dataset_spec, find_dataset
from src.data.load import BABY_SHA256, load_dataset
from src.training import kaggle
from scripts.package_kaggle import package
import test_adrec_notebook as adrec_notebook
from test_diffusion_steps_notebook import helpers

ROOT = Path(__file__).resolve().parents[1]


class DatasetSelectionTests(unittest.TestCase):
    def test_default_baby_config_matches_existing_checkpoints(self):
        for model, path in [('adrec', 'results/adrec/baby/adrec_seed42_config.json'),
                            ('sasrec', 'results/sasrec/baby/sasrec_seed42_config.json'),
                            ('gru4rec', 'results/gru4rec/baby/gru4rec_seed42_config.json')]:
            self.assertEqual(kaggle.make_config(model, 'full'),
                             json.loads((ROOT / path).read_text()))
            self.assertEqual(kaggle.make_config(model, 'full', dataset='baby'),
                             kaggle.make_config(model, 'full'))
        self.assertEqual(dataset_spec()['sha256'], BABY_SHA256)
        for name in ['beauty', 'ml-100k']:
            cfg = kaggle.make_config('adrec', 'full', dataset=name)
            self.assertEqual(cfg['dataset_sha256'], dataset_spec(name)['sha256'])
            self.assertEqual(kaggle._args(cfg, 100, 'cpu').dataset, name)
        with self.assertRaises(ValueError):
            kaggle.make_config('sasrec', dataset='movies')

    def test_discovery_selects_checksum_and_rejects_wrong_or_ambiguous_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ['baby', 'beauty']:
                p = root / name / 'dataset.pkl'
                p.parent.mkdir()
                p.write_bytes(name.encode())
            digest = hashlib.sha256(b'beauty').hexdigest()
            with patch('src.data.catalog.dataset_spec', return_value={'sha256': digest}):
                self.assertEqual(find_dataset(root, 'beauty'), root / 'beauty/dataset.pkl')
                with self.assertRaises(ValueError):
                    find_dataset(root, 'beauty', root / 'baby/dataset.pkl')
                (root / 'dataset.pkl').write_bytes(b'beauty')
                with self.assertRaises(ValueError):
                    find_dataset(root, 'beauty')
                self.assertEqual(find_dataset(root, 'beauty', root / 'beauty/dataset.pkl'),
                                 root / 'beauty/dataset.pkl')

    def test_pretrained_identity_checks_even_with_equal_vocabulary(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = kaggle.make_config('pretrain', dataset='beauty')
            model = kaggle.build_model(cfg, 23, 'cpu')
            path = Path(tmp) / 'best.pt'
            torch.save({'config': cfg, 'model_state': model.state_dict()}, path)
            adrec = dict(cfg, model='adrec')
            kaggle.build_model(adrec, 23, 'cpu', path)
            for changed in [dict(adrec, dataset='ml-100k'), dict(adrec, dataset_sha256='wrong')]:
                with self.assertRaisesRegex(ValueError, 'different dataset'):
                    kaggle.build_model(changed, 23, 'cpu', path)
            torch.save(model.state_dict(), path)
            with self.assertRaisesRegex(ValueError, 'metadata'):
                kaggle.build_model(adrec, 23, 'cpu', path)

    def test_packager_rejects_wrong_dataset_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = root / 'dataset.pkl'
            p.write_bytes(pickle.dumps({}))
            with self.assertRaisesRegex(ValueError, 'checksum'):
                package(root / 'output', p, dataset_name='beauty')
            self.assertFalse((root / 'output').exists())

    def test_diffusion_migration_is_exact_and_other_changes_fail(self):
        ns = helpers()
        registry = json.loads((ROOT / 'configs/legacy_source_compatibility.json').read_text())
        migration = registry['migrations'][0]
        files = {migration['path']: migration['before']}
        accepted = ns['check_source_compatibility'](files, ROOT, 'baby')
        self.assertEqual(accepted, [migration])
        for changed, dataset in [(files, 'beauty'), ({migration['path']: 'wrong'}, 'baby')]:
            with self.assertRaises(ValueError):
                ns['check_source_compatibility'](changed, ROOT, dataset)
        with self.assertRaisesRegex(ValueError, 'CHECKPOINT_INPUT'):
            ns['locate_references'](None, ROOT, ROOT, [42], dataset='beauty')

    def test_new_dataset_exports_and_cross_dataset_mixing(self):
        fixture = adrec_notebook.ExperimentTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        run = fixture.fixture(42)
        for stage in ['adrec', 'pretrain']:
            p = run / stage / 'config.json'
            cfg = json.loads(p.read_text())
            cfg.update(dataset='beauty', dataset_sha256=dataset_spec('beauty')['sha256'])
            p.write_text(json.dumps(cfg))
        dest = fixture.root / 'beauty_results'
        adrec_notebook.export_run(run, dest)
        self.assertEqual(adrec_notebook.summarize(dest)[0]['dataset'], 'Beauty')
        self.assertTrue((dest / 'adrec_beauty_summary.csv').exists())
        self.assertFalse((dest / 'adrec_baby_summary.csv').exists())
        with self.assertRaises(ValueError):
            adrec_notebook.export_run(fixture.fixture(43), dest)


if __name__ == '__main__':
    unittest.main()

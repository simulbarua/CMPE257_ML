"""Training contract checks on small synthetic data, not benchmark results."""
import hashlib
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.training import kaggle
from src.utils.reproducibility import seed_everything


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        # Full catalog extends past top-20, including a user with a longer history.
        data = {"train": [[1, 2, 3], [4, 5, 6, 7], [8, 9, 10]],
                "val": [[11], [12], [13]], "test": [[21], [22], [23]]}
        self.dataset = self.root / "dataset.pkl"
        self.dataset.write_bytes(pickle.dumps(data))

    def config(self, model):
        cfg = kaggle.make_config(model)
        cfg["dataset_sha256"] = hashlib.sha256(self.dataset.read_bytes()).hexdigest()
        return cfg

    def run_training(self, cfg, name, **kwargs):
        # Avoid collecting a full pip/git report repeatedly in unit tests.
        with patch.object(kaggle, "_environment", return_value={"test": True}), \
             patch.object(kaggle.subprocess, "check_output", return_value="test environment\n"):
            return kaggle.train(cfg, self.dataset, self.root / name, device="cpu", **kwargs)

    def test_resume_matches_uninterrupted_and_reloads_inference(self):
        cfg = self.config("gru4rec")
        full = dict(cfg, epochs=4)
        self.run_training(full, "continuous", finalize=False)
        self.run_training(cfg, "first", finalize=False)
        self.run_training(full, "resumed", resume=self.root / "first/latest.pt", finalize=False)
        a = torch.load(self.root / "continuous/latest.pt", weights_only=True)
        b = torch.load(self.root / "resumed/latest.pt", weights_only=True)
        self.assertEqual(a["history"], b["history"])
        self.assertEqual(a["scheduler"], b["scheduler"])
        for name, tensor in a["model_state"].items():
            self.assertTrue(torch.equal(tensor, b["model_state"][name]), name)
        self.assertIsNone(json.loads((self.root / "resumed/results.json").read_text())["test"])
        result = self.run_training(full, "final", resume=self.root / "resumed/latest.pt")
        self.assertEqual(result["epochs_completed"], 4)
        self.assertEqual(result["test"]["units"], "percent")
        self.assertTrue((self.root / "final/latest.pt").exists())
        model, _ = kaggle.load_for_inference(self.root / "final/best.pt", "cpu")
        scores = kaggle.predict_scores(model, [[0, 1, 2]])
        self.assertEqual(tuple(scores.shape), (1, 24))
        with self.assertRaises(ValueError):
            self.run_training(dict(full, seed=99), "bad", resume=self.root / "first/latest.pt")
        with self.assertRaises(FileExistsError):
            self.run_training(full, "first")

    def test_adrec_stages_freeze_resume_and_history_only_prediction(self):
        self.run_training(self.config("pretrain"), "pretrain", finalize=False)
        pre = self.root / "pretrain/best.pt"
        original = kaggle._weights(pre)["item_embedding.weight"].clone()
        observed = []
        real_save = kaggle._atomic_save

        def observe(value, path):
            if path.name == "latest.pt":
                observed.append(value["model_state"]["item_embedding.weight"].detach().cpu().clone())
            real_save(value, path)

        cfg = self.config("adrec")
        with patch.object(kaggle, "_atomic_save", side_effect=observe):
            self.run_training(cfg, "adrec", pretrained_checkpoint=pre, finalize=False)
        self.assertTrue(torch.equal(original, observed[0]), "Warmup changed frozen embeddings")
        self.assertFalse(torch.equal(original, observed[1]), "Fine-tuning did not update embeddings")
        full = dict(cfg, epochs=3)
        self.run_training(full, "adrec_continuous", pretrained_checkpoint=pre, finalize=False)
        self.run_training(full, "adrec_resume", resume=self.root / "adrec/latest.pt", finalize=False)
        a = torch.load(self.root / "adrec_continuous/latest.pt", weights_only=True)
        b = torch.load(self.root / "adrec_resume/latest.pt", weights_only=True)
        self.assertEqual(a["history"], b["history"])
        for name, tensor in a["model_state"].items():
            self.assertTrue(torch.equal(tensor, b["model_state"][name]), name)
        model, _ = kaggle.load_for_inference(self.root / "adrec/best.pt", "cpu")
        x = torch.tensor([[0, 0, 1, 2, 3]])
        true_labels = torch.tensor([[0, 0, 2, 3, 11]])
        seed_everything(123)
        with torch.no_grad():
            expected = model.calculate_score(model(x, true_labels, train_flag=False)[1])
        seed_everything(123)
        actual = kaggle.predict_scores(model, x)
        self.assertTrue(torch.equal(actual, expected))
        bad_cfg = dict(cfg, hidden_size=32)
        with self.assertRaises(ValueError):
            kaggle.build_model(bad_cfg, 23, "cpu", pre)

    def test_sasrec_uses_positions_and_saves_test_metrics(self):
        cfg = self.config("sasrec")
        result = self.run_training(cfg, "sasrec")
        self.assertEqual(result["test"]["protocol"], "adrec_upstream")
        model, saved_cfg = kaggle.load_for_inference(self.root / "sasrec/best.pt", "cpu")
        self.assertEqual(model.args.model, "sasrec")
        self.assertEqual(saved_cfg, cfg)


if __name__ == "__main__":
    unittest.main()

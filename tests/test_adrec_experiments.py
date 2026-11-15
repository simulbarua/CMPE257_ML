"""Verify result aggregation and completed-stage reuse independently of training."""
import csv
import json
from pathlib import Path
import statistics
import tempfile
import unittest
from unittest.mock import patch

from src.training.adrec_experiments import METRICS, RUN_COLUMNS, SUMMARY_COLUMNS, export_run, run_seed, run_stage, summarize
from src.training.kaggle import make_config


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def fixture(self, seed, mode="full"):
        run = self.root / f"adrec_{mode}_seed{seed}"
        for stage in ("pretrain", "adrec"):
            folder = run / stage
            folder.mkdir(parents=True)
            cfg = make_config(stage, mode)
            cfg["seed"] = seed
            values = {key: float(seed - 40 + i) for i, key in enumerate(METRICS)}
            report = {"model": stage, "mode": mode, "best_epoch": 5, "epochs_completed": 26,
                      "early_stopped": True, "test": None if stage == "pretrain" else {
                          "protocol": "adrec_upstream", "units": "percent", "metrics": values}}
            for name, value in [("config", cfg), ("results", report), ("history", [{"epoch": 0}]),
                                ("environment", {"seed": seed})]:
                (folder / f"{name}.json").write_text(json.dumps(value))
            (folder / "requirements-lock.txt").write_text("torch==test\n")
            # Export hashes bytes without deserializing checkpoints.
            (folder / "best.pt").write_bytes(f"best-{stage}-{seed}".encode())
            (folder / "latest.pt").write_bytes(f"latest-{stage}-{seed}".encode())
        return run

    def test_five_seed_csv_schema_statistics_and_metadata_only(self):
        dest = self.root / "results"
        for seed in range(42, 47):
            export_run(self.fixture(seed), dest)
        with (dest / "adrec_baby_runs.csv").open() as handle:
            reader = csv.DictReader(handle)
            self.assertEqual(reader.fieldnames, RUN_COLUMNS)
            rows = list(reader)
        self.assertEqual([int(row["seed"]) for row in rows], list(range(42, 47)))
        with (dest / "adrec_baby_summary.csv").open() as handle:
            reader = csv.DictReader(handle)
            self.assertEqual(reader.fieldnames, SUMMARY_COLUMNS)
            summary = list(reader)
        self.assertEqual(summary[0]["n_runs"], "5")
        self.assertEqual(summary[0]["seeds"], "42-46")
        self.assertEqual(summary[1]["statistic"], "std (sample, n-1)")
        self.assertEqual(float(summary[0]["HR@5"]), 4.0)
        self.assertEqual(float(summary[1]["HR@5"]), round(statistics.stdev([2, 3, 4, 5, 6]), 4))
        self.assertFalse(list(dest.rglob("*.pt")))
        self.assertTrue((dest / "pretrain/pretrain_seed46_config.json").exists())
        self.assertEqual(json.loads((dest / "environment.json").read_text())["seed"], 42)

    def test_single_run_has_no_fake_std_and_mixed_runs_are_rejected(self):
        dest = self.root / "results"
        run = self.fixture(42)
        export_run(run, dest)
        with (dest / "adrec_baby_summary.csv").open() as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 1)
        with self.assertRaises(ValueError):
            export_run(self.fixture(43, "smoke"), dest)
        result_path = run / "adrec/results.json"
        result = json.loads(result_path.read_text())
        result["test"]["metrics"]["HR@5"] = 99
        result_path.write_text(json.dumps(result))
        with self.assertRaises(ValueError):
            export_run(run, dest)
        self.assertEqual(len(summarize(dest)), 1)

    def test_completed_stage_reused_and_missing_result_resumes(self):
        run = self.fixture(43)
        source = run / "adrec"
        cfg = json.loads((source / "config.json").read_text())
        with patch("src.training.adrec_experiments.train") as train:
            out = self.root / "resumed/adrec"
            report = run_stage(cfg, "unused", out, resume_from=source)
            self.assertTrue(report["early_stopped"])
            train.assert_not_called()
            self.assertEqual((source / "best.pt").read_bytes(), (out / "best.pt").read_bytes())
            (out / "results.json").unlink()
            run_stage(cfg, "unused", out)
            self.assertEqual(train.call_args.kwargs["resume"], out / "latest.pt")
            self.assertTrue(train.call_args.kwargs["finalize"])
        with self.assertRaises(ValueError):
            run_stage(dict(cfg, seed=44), "unused", self.root / "bad", resume_from=source)

    def test_both_stages_receive_seed_and_own_embedding_checkpoint(self):
        for seed in range(42, 47):
            run = self.fixture(seed, "smoke")
            with patch("src.training.adrec_experiments.run_stage") as stage, \
                 patch("src.training.adrec_experiments.export_run", return_value=[]):
                run_seed(seed, "smoke", "unused", self.root, self.root / "out")
                pretrain_call, adrec_call = stage.call_args_list
                self.assertEqual(pretrain_call.args[0]["seed"], seed)
                self.assertEqual(adrec_call.args[0]["seed"], seed)
                self.assertEqual(pretrain_call.args[0]["model"], "pretrain")
                self.assertEqual(adrec_call.kwargs["pretrained_checkpoint"], run / "pretrain/best.pt")


if __name__ == "__main__":
    unittest.main()

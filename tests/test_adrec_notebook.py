"""Verify result aggregation and completed-stage reuse independently of training."""
import csv
import json
from pathlib import Path
import statistics
import tempfile
import shutil
import unittest
from unittest.mock import Mock

NOTEBOOK = json.loads((Path(__file__).resolve().parents[1] / "notebooks/adrec_kaggle.ipynb").read_text())
EXPORT_SOURCE = "".join(next(cell["source"] for cell in NOTEBOOK["cells"] if cell["id"] == "export-helpers"))
SWEEP_SOURCE = "".join(next(cell["source"] for cell in NOTEBOOK["cells"] if cell["id"] == "seed-sweep"))
EXPORTS = {}
exec(compile(EXPORT_SOURCE, "adrec_kaggle:export-helpers", "exec"), EXPORTS)
METRICS, RUN_COLUMNS, SUMMARY_COLUMNS = (EXPORTS[key] for key in ("METRICS", "RUN_COLUMNS", "SUMMARY_COLUMNS"))
export_run, summarize = EXPORTS["export_run"], EXPORTS["summarize"]
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

    def sweep_context(self, seeds, train, resume_root=None):
        work = self.root / "working"
        work.mkdir(exist_ok=True)
        return {**EXPORTS, "make_config": make_config, "train": train,
                "RUN_MODE": "full", "SEEDS": seeds, "RESUME_ROOT": resume_root,
                "WORK_ROOT": work, "RUNS_ROOT": work / "runs",
                "ARCHIVE_ROOT": work, "DATASET": "baby",
                "RESULTS_DIR": work / "results/adrec", "DATASET_PATH": "unused"}

    def test_direct_loop_sets_both_seeds_and_reuses_completed_stages(self):
        fixtures = {seed: self.fixture(seed) for seed in range(42, 47)}

        def fake_train(cfg, dataset, output, **kwargs):
            seed, stage = cfg["seed"], cfg["model"]
            self.assertEqual(kwargs["finalize"], stage == "adrec")
            if stage == "adrec":
                self.assertEqual(kwargs["pretrained_checkpoint"], output.parent / "pretrain/best.pt")
                self.assertTrue(kwargs["pretrained_checkpoint"].exists())
            shutil.copytree(fixtures[seed] / stage, output, dirs_exist_ok=True)
            return json.loads((output / "results.json").read_text())

        train = Mock(side_effect=fake_train)
        context = self.sweep_context(list(range(42, 47)), train)
        exec(compile(SWEEP_SOURCE, "adrec_kaggle:seed-sweep", "exec"), context)
        calls = [(call.args[0]["seed"], call.args[0]["model"]) for call in train.call_args_list]
        self.assertEqual(calls, [(seed, stage) for seed in range(42, 47) for stage in ("pretrain", "adrec")])
        self.assertEqual(sorted(context["completed_runs"]), list(range(42, 47)))
        self.assertTrue((context["WORK_ROOT"] / "adrec_results.zip").exists())
        train.reset_mock()
        exec(compile(SWEEP_SOURCE, "adrec_kaggle:seed-sweep", "exec"), context)
        train.assert_not_called()
        # A saved epoch without final results resumes only that unfinished stage.
        stage_dir = context["RUNS_ROOT"] / "adrec_full_seed42/adrec"
        (stage_dir / "results.json").unlink()
        exec(compile(SWEEP_SOURCE, "adrec_kaggle:seed-sweep", "exec"), context)
        train.assert_called_once()
        self.assertEqual(train.call_args.kwargs["resume"], stage_dir / "latest.pt")

    def test_attached_completed_runs_are_copied_and_config_mismatch_fails(self):
        source = self.fixture(43)
        train = Mock(side_effect=AssertionError("Completed stages must not train again"))
        context = self.sweep_context([43], train, resume_root=self.root)
        exec(compile(SWEEP_SOURCE, "adrec_kaggle:seed-sweep", "exec"), context)
        train.assert_not_called()
        destination = context["RUNS_ROOT"] / "adrec_full_seed43"
        self.assertEqual((source / "adrec/best.pt").read_bytes(), (destination / "adrec/best.pt").read_bytes())
        config_path = destination / "adrec/config.json"
        cfg = json.loads(config_path.read_text())
        cfg["seed"] = 44
        config_path.write_text(json.dumps(cfg))
        with self.assertRaises(ValueError):
            exec(compile(SWEEP_SOURCE, "adrec_kaggle:seed-sweep", "exec"), context)


if __name__ == "__main__":
    unittest.main()

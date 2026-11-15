"""Run independent ADRec seeds and export the baseline-compatible result format."""
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import statistics

from src.training.kaggle import make_config, train

METRICS = ["HR@5", "HR@10", "HR@20", "NDCG@5", "NDCG@10", "NDCG@20"]
RUN_COLUMNS = ["run_id", "model", "dataset", "seed", "best_epoch", "epochs_completed", *METRICS]
SUMMARY_COLUMNS = ["model", "dataset", "n_runs", "seeds", "statistic", *METRICS]
METADATA = ["config.json", "results.json", "history.json", "environment.json", "requirements-lock.txt"]


def _json(path):
    return json.loads(Path(path).read_text())


def _write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate(cfg, result):
    if cfg["model"] != "adrec" or result["model"] != "adrec":
        raise ValueError("Expected ADRec results")
    if result["mode"] != cfg["mode"] or type(cfg["seed"]) is not int:
        raise ValueError("Invalid result mode or seed")
    test = result.get("test")
    if not test or test["protocol"] != "adrec_upstream" or test["units"] != "percent":
        raise ValueError("Export requires completed test evaluation using adrec_upstream")
    for key in METRICS:
        if not math.isfinite(test["metrics"][key]) or not 0 <= test["metrics"][key] <= 100:
            raise ValueError("Invalid test metric")
    return {k: v for k, v in cfg.items() if k != "seed"}


def _records(folder):
    records = []
    reference = None
    for path in Path(folder).glob("adrec_seed*_results.json"):
        prefix = path.name.removesuffix("results.json")
        cfg = _json(path.with_name(prefix + "config.json"))
        result = _json(path)
        if prefix != f"adrec_seed{cfg['seed']}_":
            raise ValueError("Seed does not match the filename")
        identity = _validate(cfg, result)
        if reference is not None and identity != reference:
            raise ValueError("Do not combine different configurations or smoke/full runs")
        reference = identity
        records.append((cfg, result))
    return sorted(records, key=lambda pair: pair[0]["seed"])


def summarize(folder):
    """Same CSV columns and sample-standard-deviation convention as the baselines."""
    folder = Path(folder)
    records = _records(folder)
    if not records:
        return []
    rows = [{"run_id": f"adrec_{cfg['mode']}_seed{cfg['seed']}", "model": "ADRec",
             "dataset": "Baby", "seed": cfg["seed"], "best_epoch": result["best_epoch"],
             "epochs_completed": result["epochs_completed"], **result["test"]["metrics"]}
            for cfg, result in records]
    seeds = [r["seed"] for r in rows]
    seed_label = (f"{seeds[0]}-{seeds[-1]}" if len(seeds) > 1 and seeds == list(range(seeds[0], seeds[-1] + 1))
                  else ",".join(map(str, seeds)))
    common = {"model": "ADRec", "dataset": "Baby", "n_runs": len(rows), "seeds": seed_label}
    summary = [{**common, "statistic": "mean", **{k: round(statistics.mean(r[k] for r in rows), 4) for k in METRICS}}]
    if len(rows) > 1:
        summary.append({**common, "statistic": "std (sample, n-1)",
                        **{k: round(statistics.stdev(r[k] for r in rows), 4) for k in METRICS}})
    for filename, columns, values in [("adrec_baby_runs.csv", RUN_COLUMNS, rows),
                                      ("adrec_baby_summary.csv", SUMMARY_COLUMNS, summary)]:
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(values)
        _write(folder / filename, stream.getvalue())
    cfg = records[0][0]
    coverage = "Complete five-seed set." if seeds == [42, 43, 44, 45, 46] else "Partial seed set; summary includes only the listed completed runs."
    _write(folder / "EXPERIMENT.md", f"""# Experiment record: ADRec on Amazon Baby

- Model: ADRec, {cfg['mode']} mode.
- Completed seeds: {seed_label}. {coverage}
- Dataset SHA-256: `{cfg['dataset_sha256']}`. Released train/validation/test splits retained.
- Training: separate embedding pretraining and ADRec training for each seed; the same seed is set in both stages.
- Training notebook: `notebooks/adrec_kaggle.ipynb`.
- Configuration: `adrec_seed<N>_config.json`; pretraining metadata is under `pretrain/`.
- Selection: improvement in any tracked validation metric; evaluation interval {cfg['eval_interval']}, patience {cfg['patience']}.
- Evaluation: `adrec_upstream`, batch size {cfg['evaluation_batch_size']}, HR/NDCG percentages, all catalog columns including padding; no seen-item filtering.
- Aggregation: arithmetic mean and sample standard deviation (n-1) of the per-seed test percentages. A single run has no reported standard deviation.
- Source/runtime: per-seed environment files contain source hashes, Python, PyTorch, CUDA, and hardware. Per-seed requirements files record installed packages.
- Checkpoints: hashes and storage references are in `adrec_seed<N>_checkpoints.json`. Weights are stored separately from these result files.
- Timing: training and inference benchmarks were not recorded by this export.
- Deviations: project seed settings, resumable training, enforced embedding freeze during alignment, and evaluation-mode checkpoint inference are documented in `docs/KAGGLE_TRAINING.md`.

`environment.json` and `requirements-lock.txt` are copies from the lowest listed seed for compatibility with the baseline result layout. Per-seed files retain each run's own environment.
""")
    first = seeds[0]
    for name in ("environment.json", "requirements-lock.txt"):
        source = folder / f"adrec_seed{first}_{name}"
        if source.exists():
            shutil.copyfile(source, folder / name)
    return rows


def export_run(run_dir, results_dir, checkpoint_location=None):
    """Copy metadata only; never include .pt weights in the shareable folder."""
    run_dir, results_dir = Path(run_dir), Path(results_dir)
    adrec, pretrain = run_dir / "adrec", run_dir / "pretrain"
    cfg, result = _json(adrec / "config.json"), _json(adrec / "results.json")
    identity = _validate(cfg, result)
    seed = cfg["seed"]
    pre_cfg = _json(pretrain / "config.json")
    expected_pre = dict(cfg, model="pretrain")
    if pre_cfg != expected_pre:
        raise ValueError("Pretraining and ADRec must use matching settings and seeds")
    for existing_cfg, existing_result in _records(results_dir):
        if _validate(existing_cfg, existing_result) != identity:
            raise ValueError("Existing export has incompatible configuration")
        if existing_cfg["seed"] == seed and existing_result != result:
            raise ValueError("A different result already exists for this seed")
    for stage in (pretrain, adrec):
        for name in [*METADATA, "best.pt", "latest.pt"]:
            if not (stage / name).is_file():
                raise FileNotFoundError(stage / name)
    for stage, source, target in [("pretrain", pretrain, results_dir / "pretrain"),
                                   ("adrec", adrec, results_dir)]:
        target.mkdir(parents=True, exist_ok=True)
        for name in METADATA:
            shutil.copyfile(source / name, target / f"{stage}_seed{seed}_{name}")
    checkpoints = {"seed": seed, "location": checkpoint_location or f"runs/{run_dir.name}",
                   "files": {f"{stage}/{name}": _hash(run_dir / stage / name)
                             for stage in ("pretrain", "adrec") for name in ("best.pt", "latest.pt")}}
    _write(results_dir / f"adrec_seed{seed}_checkpoints.json", json.dumps(checkpoints, indent=2) + "\n")
    return summarize(results_dir)


def run_stage(cfg, dataset_path, output, *, pretrained_checkpoint=None, resume_from=None):
    """Reuse completed stages; otherwise resume the latest completed epoch."""
    output = Path(output)
    source = output if (output / "latest.pt").exists() else Path(resume_from) if resume_from else None
    if source and source.is_dir() and (source / "latest.pt").exists():
        saved_cfg = _json(source / "config.json")
        if saved_cfg != cfg:
            raise ValueError(f"Stage configuration mismatch: {source}")
        report_path = source / "results.json"
        report = _json(report_path) if report_path.exists() else None
        complete = report and (report["early_stopped"] or report["epochs_completed"] >= cfg["epochs"])
        evaluated = cfg["model"] == "pretrain" or report and report.get("test") is not None
        if complete and evaluated and (source / "best.pt").exists():
            if source.resolve() != output.resolve():
                shutil.copytree(source, output, dirs_exist_ok=True)
            print(f"Reusing completed {cfg['model']} seed {cfg['seed']}", flush=True)
            return report
        resume = source / "latest.pt"
    else:
        if resume_from is not None:
            raise FileNotFoundError(Path(resume_from) / "latest.pt")
        resume = None
    return train(cfg, dataset_path, output, pretrained_checkpoint=pretrained_checkpoint,
                 resume=resume, finalize=cfg["model"] != "pretrain")


def run_seed(seed, mode, dataset_path, runs_root, results_dir, resume_root=None):
    if type(seed) is not int:
        raise ValueError("Seed must be an integer")
    run_id = f"adrec_{mode}_seed{seed}"
    run_dir = Path(runs_root) / run_id
    prior = Path(resume_root) / run_id if resume_root else None
    pre_cfg, cfg = make_config("pretrain", mode), make_config("adrec", mode)
    pre_cfg["seed"] = cfg["seed"] = seed
    for stage, stage_cfg in [("pretrain", pre_cfg), ("adrec", cfg)]:
        remote_stage = prior / stage if prior and (prior / stage / "latest.pt").exists() else None
        print(f"Starting {stage}, seed {seed}, mode {mode}", flush=True)
        run_stage(stage_cfg, dataset_path, run_dir / stage,
                  pretrained_checkpoint=run_dir / "pretrain/best.pt" if stage == "adrec" else None,
                  resume_from=remote_stage)
    rows = export_run(run_dir, results_dir)
    return {"run_dir": str(run_dir), "result": _json(run_dir / "adrec/results.json"), "rows": rows}

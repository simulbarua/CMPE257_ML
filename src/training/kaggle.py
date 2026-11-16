"""Team training orchestration; vendored ADRec model code remains unchanged.

The shared evaluator retains the upstream metric protocol. See
docs/KAGGLE_TRAINING.md for training choices and reproduction limitations.
"""
import copy
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from src.data import SequenceDataset, load_dataset
from src.data.load import dataset_summary
from src.data.catalog import dataset_spec
from src.data.torch_adapter import make_loader
from src.evaluation import evaluate
from src.models.gru4rec import GRU4Rec
from src.utils.reproducibility import seed_everything

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "third_party/ADRec/src"
if str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))


def make_config(model, mode="smoke", dataset="baby"):
    """All notebooks start from the same versioned settings."""
    if model not in {"adrec", "pretrain", "sasrec", "gru4rec"}:
        raise ValueError("Unknown model")
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be smoke or full")
    cfg = json.loads((ROOT / "configs/shared.json").read_text())
    cfg.update(json.loads((ROOT / "configs/kaggle.json").read_text()))
    spec = dataset_spec(dataset)
    cfg["dataset_sha256"] = spec["sha256"]
    # Keep legacy Baby configs byte-for-byte equivalent for resume/notebook users.
    if dataset != "baby":
        cfg["dataset"] = dataset
    cfg.update(model=model, mode=mode, user_limit=None, diffusion_steps=32)
    if mode == "smoke":
        cfg.update(user_limit=9, hidden_size=16, diffusion_steps=4,
                   train_batch_size=4, epochs=2, eval_interval=1,
                   warmup_epochs=1, evaluation_batch_size=4)
    return cfg


def _args(cfg, num_items, device):
    settings = yaml.safe_load((UPSTREAM / "config.yaml").read_text())
    settings.update(dataset=cfg.get("dataset", "baby"), item_num=num_items, device=str(device),
                    model=cfg["model"], max_len=cfg["max_len"],
                    hidden_size=cfg["hidden_size"],
                    diffusion_steps=cfg["diffusion_steps"],
                    pretrained=False, freeze_emb=False)
    return SimpleNamespace(**settings)


def _weights(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    return saved["model_state"] if "model_state" in saved else saved


def build_model(cfg, num_items, device, pretrained_checkpoint=None):
    args = _args(cfg, num_items, device)
    if cfg["model"] == "gru4rec":
        model = GRU4Rec(num_items, cfg["hidden_size"], cfg["hidden_size"],
                        cfg["gru_num_layers"], cfg["gru_dropout"])
    elif cfg["model"] in {"sasrec", "pretrain"}:
        from sasrec import SASRec
        model = SASRec(args)
    else:
        from model import Att_Diffuse_model
        if pretrained_checkpoint is None:
            # Used only when restoring a complete ADRec checkpoint below.
            model = Att_Diffuse_model(args)
        else:
            saved = torch.load(pretrained_checkpoint, map_location="cpu", weights_only=True)
            if "config" in saved:
                previous = saved["config"]
                if (previous.get("dataset", "baby") != cfg.get("dataset", "baby")
                        or previous.get("dataset_sha256") != cfg["dataset_sha256"]):
                    raise ValueError("Pretrained checkpoint belongs to a different dataset")
                if previous.get("model") != "pretrain":
                    raise ValueError("Expected an embedding-pretraining checkpoint")
            elif cfg.get("dataset", "baby") != "baby":
                raise ValueError("New datasets require pretrained checkpoint dataset metadata")
            weight = (saved["model_state"] if "model_state" in saved else saved)["item_embedding.weight"]
            if tuple(weight.shape) != (num_items + 1, cfg["hidden_size"]):
                raise ValueError("Pretrained embedding vocabulary/dimension mismatch")

            class ADRecWithEmbeddings(Att_Diffuse_model):
                def load_pretrained_emb_weight(self):
                    return weight.clone()

            # Override only the fixed upstream checkpoint path. All model math
            # and the from_pretrained initialization remain upstream code.
            args.pretrained = True
            args.freeze_emb = True
            model = ADRecWithEmbeddings(args)
    return model.to(device)


def predict_scores(model, histories, masks=None):
    """Full catalog scores using visible history only; no held-out labels."""
    device = next(model.parameters()).device
    x = torch.as_tensor(histories, dtype=torch.long, device=device)
    if x.ndim != 2 or (x.gt(0).sum(1) == 0).any():
        raise ValueError("Each history must contain at least one item")
    dummy = torch.cat([x[:, 1:], torch.ones_like(x[:, :1])], dim=1)
    dummy = dummy.masked_fill(x == 0, 0)
    model.eval()
    with torch.no_grad():
        return model.calculate_score(model(x, dummy, train_flag=False)[1])


def _report(model, dataset, cfg, num_items):
    return evaluate(dataset, lambda h, m: predict_scores(model, h, m), num_items,
                    batch_size=cfg["evaluation_batch_size"], ks=cfg["metric_ks"])


def _atomic_save(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _rng_state(loader):
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loader": loader.generator.get_state()}


def _restore_rng(state, loader):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    ns = state["numpy"]
    np.random.set_state((ns[0], np.array(ns[1], dtype=np.uint32), *ns[2:]))
    if state["cuda"] and torch.cuda.is_available():
        if len(state["cuda"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(state["cuda"])
    loader.generator.set_state(state["loader"])


def _environment(device):
    manifest = ROOT / "bundle_manifest.json"
    if manifest.exists():
        source = json.loads(manifest.read_text())
    else:
        source = {"base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "working_tree": subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True)}
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "device": str(device), "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "source": source}


def train(cfg, dataset_path, output_dir, *, pretrained_checkpoint=None,
          resume=None, device=None, finalize=True):
    """Train one stage; resume an epoch-boundary latest.pt, including optimizer/RNG.

    Set finalize=False to save progress without looking at test results when
    splitting a long run across sessions. Increase only epochs on resume.
    """
    cfg = copy.deepcopy(cfg)
    if cfg["epochs"] < 1 or cfg["eval_interval"] < 1 or cfg["patience"] < 1:
        raise ValueError("epochs, eval_interval and patience must be positive")
    if cfg["model"] == "adrec" and cfg["epochs"] <= cfg["warmup_epochs"]:
        raise ValueError("ADRec needs epochs beyond its frozen-embedding warmup")
    device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if cfg["mode"] == "full" and device.type != "cuda":
        raise ValueError("Enable a GPU for full training. Use smoke mode for CPU checks.")
    seed_everything(cfg["seed"])
    torch.set_float32_matmul_precision("high")  # matches upstream trainer
    full = load_dataset(dataset_path, cfg["dataset_sha256"])
    summary = dataset_summary(full)
    num_items = summary["max_item_id"]  # full vocabulary even in smoke mode
    limit = cfg["user_limit"]
    data = {s: rows[:limit] for s, rows in full.items()} if limit else full
    datasets = {s: SequenceDataset(data, s, cfg["max_len"],
                target_mode="last" if cfg["model"] == "gru4rec" else "per_token") for s in data}
    loader = make_loader(datasets["train"], cfg["train_batch_size"], shuffle=True, seed=cfg["seed"])
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "latest.pt").exists() and resume is None:
        raise FileExistsError("Run already exists: set resume=latest.pt or choose a new output folder")
    checkpoint = torch.load(resume, map_location="cpu", weights_only=True) if resume else None
    if checkpoint:
        before = {k: v for k, v in checkpoint["config"].items() if k != "epochs"}
        after = {k: v for k, v in cfg.items() if k != "epochs"}
        if before != after or cfg["epochs"] < checkpoint["next_epoch"]:
            raise ValueError("Resume requires the same configuration; only epochs may increase")
    if cfg["model"] == "adrec" and not (checkpoint or pretrained_checkpoint):
        raise ValueError("ADRec requires pretrained embeddings or a full resume checkpoint")
    model = build_model(cfg, num_items, device, None if checkpoint else pretrained_checkpoint)
    args = _args(cfg, num_items, device)
    from pcgrad import PCGrad
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"],
                                 weight_decay=cfg["weight_decay"])
    wrapped = PCGrad(optimizer, args)  # preserves upstream gradient combination
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["scheduler_t_max"])
    start, stale, best_epoch, history = 0, 0, None, []
    best_metrics, best_state = {}, None
    stopped = False
    if checkpoint:
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start, stale = checkpoint["next_epoch"], checkpoint["stale"]
        best_epoch, best_metrics = checkpoint["best_epoch"], checkpoint["best_metrics"]
        best_state, history = checkpoint["best_state"], checkpoint["history"]
        stopped = checkpoint["early_stopped"]
        _restore_rng(checkpoint["rng"], loader)
    env = _environment(device)
    if pretrained_checkpoint and not checkpoint:
        env["pretrained_checkpoint"] = {
            "filename": Path(pretrained_checkpoint).name,
            "sha256": hashlib.sha256(Path(pretrained_checkpoint).read_bytes()).hexdigest(),
        }
    elif checkpoint:
        env["resume_checkpoint_sha256"] = hashlib.sha256(Path(resume).read_bytes()).hexdigest()
        # Preserve a portable restart file even if the saved run is complete.
        _atomic_save(checkpoint, output / "latest.pt")
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (output / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (output / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
    packages = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (output / "requirements-lock.txt").write_text(packages)
    for epoch in range(start, cfg["epochs"]):
        if stopped:
            break
        model.train()
        if cfg["model"] == "adrec":
            model.item_embedding.weight.requires_grad_(epoch >= cfg["warmup_epochs"])
        ce_sum = dif_sum = 0.0
        for batch in loader:
            x, y = batch["input_ids"].to(device), batch["labels"].to(device)
            wrapped.zero_grad()
            result = model(x, y, train_flag=True)
            ce = model.calculate_loss(result[1] if cfg["model"] == "gru4rec" else result[0], y)
            losses = [ce]
            dif = result[2] if cfg["model"] == "adrec" else torch.zeros((), device=device)
            if cfg["model"] == "adrec" and args.loss == "mse":
                losses.append(args.loss_scale * dif)
            if not all(torch.isfinite(loss).all() for loss in losses):
                raise FloatingPointError("Nonfinite training loss")
            wrapped.pc_backward(losses)
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("Nonfinite gradients")
            # Upstream PCGrad assigns zero gradients to frozen parameters;
            # Adam weight decay would still move them. Enforce the intended freeze.
            if cfg["model"] == "adrec" and not model.item_embedding.weight.requires_grad:
                model.item_embedding.weight.grad = None
            wrapped.step()
            ce_sum += ce.item()
            dif_sum += dif.item()
        scheduler.step()
        row = {"epoch": epoch, "ce_loss": ce_sum / len(loader),
               "diffusion_loss": dif_sum / len(loader),
               "embedding_trainable": model.item_embedding.weight.requires_grad}
        # Match upstream zero-based evaluation epochs; ensure short runs get
        # a checkpoint even if no scheduled validation was reached.
        should_eval = epoch > 0 and epoch % cfg["eval_interval"] == 0
        if best_state is None and epoch == cfg["epochs"] - 1:
            should_eval = True
        if should_eval:
            val = _report(model, datasets["val"], cfg, num_items)
            metrics = val["metrics"]
            row["validation"] = val
            improved = best_state is None or any(v > best_metrics[k] for k, v in metrics.items())
            if improved:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_epoch, stale = epoch, 0
                best_metrics = {k: max(v, best_metrics.get(k, float("-inf"))) for k, v in metrics.items()}
            else:
                stale += 1
            # Always reach ADRec fine-tuning, even with zero smoke-run metrics.
            stopped = stale >= cfg["patience"] and (cfg["model"] != "adrec" or epoch >= cfg["warmup_epochs"])
        history.append(row)
        snapshot = {"config": cfg, "num_items": num_items, "model_state": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "next_epoch": epoch + 1, "stale": stale, "early_stopped": stopped,
                    "best_epoch": best_epoch, "best_metrics": best_metrics,
                    "best_state": best_state, "history": history, "rng": _rng_state(loader)}
        _atomic_save(snapshot, output / "latest.pt")
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps(row), flush=True)
    if best_state is None:
        raise RuntimeError("No validation checkpoint available")
    _atomic_save({"config": cfg, "num_items": num_items, "model_state": best_state,
                  "epoch": best_epoch}, output / "best.pt")
    report = {"model": cfg["model"], "mode": cfg["mode"], "users": len(datasets["train"]),
              "dataset": summary, "best_epoch": best_epoch, "early_stopped": stopped,
              "epochs_completed": len(history), "test": None}
    if finalize and cfg["model"] != "pretrain":
        model.load_state_dict(best_state)
        # Set a recorded seed for final diffusion sampling. Training validation
        # consumes the normal RNG stream as in upstream.
        seed_everything(cfg["seed"])
        report["test"] = _report(model, datasets["test"], cfg, num_items)
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def load_for_inference(checkpoint_path, device=None):
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model = build_model(saved["config"], saved["num_items"], device)
    model.load_state_dict(saved["model_state"])
    model.eval()
    return model, saved["config"]

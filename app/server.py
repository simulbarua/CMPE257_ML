"""Web demo: recommend next items from a shopper's history with one or more models.

Run from the repository root:
    python -m app.server --model adrec=checkpoints/baby/best.pt \
        --model sasrec=checkpoints/baby/sasrec_best.pt \
        --model gru4rec=checkpoints/baby/gru4rec_best.pt

Then open http://localhost:8000 and choose a model, or "Compare all".
`--fake` serves random scores for all three models so the page can be tried
without checkpoints. Uses only the Python standard library for the web server;
the models themselves need the shared requirements (PyTorch).
"""
import argparse
import json
import random
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from src.data.catalog import dataset_spec
from src.data.load import load_dataset

PAGE = Path(__file__).with_name("static") / "index.html"
MAX_K = 50
LABELS = {"adrec": "ADRec", "sasrec": "SASRec+", "gru4rec": "GRU4Rec"}


class Recommender:
    """Loads one model once at startup and ranks the full item catalog."""

    def __init__(self, name, checkpoint, num_items, dataset, fake=False):
        self.name, self.num_items, self.fake = name, num_items, fake
        self.label = LABELS.get(name, name)
        if fake:
            self.label += " (fake scores)"
            self.max_len, self.seed = 50, None
            return
        import torch
        from src.training.kaggle import load_for_inference
        self.torch = torch
        self.model, cfg = load_for_inference(checkpoint)
        if cfg["model"] != name:
            raise ValueError(f"{checkpoint} holds a {cfg['model']} model, not {name}")
        if cfg.get("dataset", "baby") != dataset:
            raise ValueError(f"{checkpoint} was trained on {cfg.get('dataset')}, not {dataset}")
        self.max_len, self.seed = cfg["max_len"], cfg["seed"]
        self.label = f"{self.label} (seed {self.seed})"

    def recommend(self, history_item_ids, k=10):
        """Ranked item IDs and model scores (not probabilities)."""
        history = list(history_item_ids)[-self.max_len:]
        if self.fake:
            rng = random.Random(self.name + ",".join(map(str, history)))
            ids = rng.sample(range(1, self.num_items + 1), k)
            return [{"item_id": i, "score": round(10 - r * 0.3 + rng.random() * 0.1, 4)}
                    for r, i in enumerate(ids)]
        from src.training.kaggle import predict_scores
        from src.utils.reproducibility import seed_everything
        seed_everything(self.seed)  # ADRec samples noise at inference; keep it repeatable
        padded = [0] * (self.max_len - len(history)) + history
        scores = predict_scores(self.model, [padded]).cpu()
        scores[:, 0] = -self.torch.inf  # never recommend padding
        top = self.torch.topk(scores[0], k=k)
        return [{"item_id": int(i), "score": round(float(s), 4)}
                for s, i in zip(top.values, top.indices)]


def parse_request(body, num_items, model_names):
    """Validate input; raise ValueError with a message the page can show."""
    try:
        payload = json.loads(body or b"{}")
    except json.JSONDecodeError:
        raise ValueError("Request is not valid JSON")
    history, k = payload.get("history"), payload.get("k", 10)
    model = payload.get("model", model_names[0])
    if not isinstance(history, list) or not history:
        raise ValueError("History must contain at least one item ID")
    if any(type(i) is not int or not 1 <= i <= num_items for i in history):
        raise ValueError(f"Item IDs must be whole numbers from 1 to {num_items}")
    if type(k) is not int or not 1 <= k <= MAX_K:
        raise ValueError(f"k must be a whole number from 1 to {MAX_K}")
    if model != "all" and model not in model_names:
        raise ValueError(f"Unknown model: {model}")
    return history, k, model_names if model == "all" else [model]


def pick_examples(data, count=5, min_len=5):
    """Placeholder examples: visible test-time history only (train + val).
    The held-out test target is never included. Replace with the team's
    validated example list when it is available."""
    examples = []
    for user, train in enumerate(data["train"]):
        history = train + data["val"][user]
        if len(history) >= min_len:
            examples.append({"label": f"Shopper #{user}", "history": history[-20:]})
        if len(examples) == count:
            break
    return examples


class Handler(BaseHTTPRequestHandler):
    def _send(self, status, body, content_type="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/info":
            self._send(200, {"dataset": DATASET_NAME, "num_items": NUM_ITEMS, "examples": EXAMPLES,
                             "models": [{"name": r.name, "label": r.label}
                                        for r in RECOMMENDERS.values()]})
        else:
            self._send(404, {"error": "Not found"})

    def do_POST(self):
        if self.path != "/api/recommend":
            return self._send(404, {"error": "Not found"})
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        try:
            history, k, names = parse_request(body, NUM_ITEMS, list(RECOMMENDERS))
        except ValueError as error:
            return self._send(400, {"error": str(error)})
        results = []
        for name in names:
            recommender = RECOMMENDERS[name]
            start = time.perf_counter()
            items = recommender.recommend(history, k)
            latency = (time.perf_counter() - start) * 1000
            results.append({"model": name, "label": recommender.label, "items": items,
                            "latency_ms": round(latency, 1),
                            "history_used": len(history[-recommender.max_len:])})
        self._send(200, {"results": results})


def parse_models(values):
    models = {}
    for value in values:
        name, sep, path = value.partition("=")
        if not sep or name not in LABELS or not path:
            raise SystemExit(f"--model must look like adrec=PATH, sasrec=PATH or gru4rec=PATH (got {value})")
        models[name] = path
    return models


def main():
    global RECOMMENDERS, EXAMPLES, DATASET_NAME, NUM_ITEMS
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", default=[], metavar="NAME=PATH",
                        help="a model to serve; repeat for several (default adrec=checkpoints/baby/best.pt)")
    parser.add_argument("--checkpoint", help="shortcut for --model adrec=PATH")
    parser.add_argument("--dataset-name", default="baby", choices=["baby", "beauty", "ml-100k"])
    parser.add_argument("--data", default="data/baby/dataset.pkl")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--fake", action="store_true", help="random scores for all three models")
    args = parser.parse_args()
    DATASET_NAME = args.dataset_name
    spec = dataset_spec(DATASET_NAME)
    data = load_dataset(args.data, spec["sha256"])
    EXAMPLES = pick_examples(data)
    NUM_ITEMS = spec["summary"]["max_item_id"]
    if args.fake:
        models = {name: None for name in LABELS}
    else:
        models = parse_models(args.model)
        if args.checkpoint:
            models.setdefault("adrec", args.checkpoint)
        if not models:
            models = {"adrec": "checkpoints/baby/best.pt"}
    RECOMMENDERS = {name: Recommender(name, path, NUM_ITEMS, DATASET_NAME, args.fake)
                    for name, path in models.items()}
    print(f"Serving {', '.join(r.label for r in RECOMMENDERS.values())} on {DATASET_NAME} "
          f"at http://localhost:{args.port}")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()

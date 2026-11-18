"""Web demo: recommend next items from a shopper's history.

Run from the repository root:
    python -m app.server --checkpoint checkpoints/baby/best.pt --data data/baby/dataset.pkl

Then open http://localhost:8000. Uses only the Python standard library for the
web server; the model itself needs the shared requirements (PyTorch).
`--fake` serves random scores so the page can be tried without a model.
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


class Recommender:
    """Loads the model once at startup and ranks the full item catalog."""

    def __init__(self, checkpoint, num_items, fake=False):
        self.num_items, self.fake = num_items, fake
        if fake:
            self.model_name, self.max_len, self.seed = "fake (random scores)", 50, None
            return
        import torch
        from src.training.kaggle import load_for_inference
        self.torch = torch
        self.model, cfg = load_for_inference(checkpoint)
        self.model_name, self.max_len, self.seed = cfg["model"], cfg["max_len"], cfg["seed"]
        if cfg.get("dataset", "baby") != DATASET_NAME:
            raise ValueError(f"Checkpoint was trained on {cfg.get('dataset')}, not {DATASET_NAME}")

    def recommend(self, history_item_ids, k=10):
        """Ranked item IDs and model scores (not probabilities)."""
        history = list(history_item_ids)[-self.max_len:]
        if self.fake:
            rng = random.Random(",".join(map(str, history)))
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


def parse_request(body, num_items):
    """Validate input; raise ValueError with a message the page can show."""
    try:
        payload = json.loads(body or b"{}")
    except json.JSONDecodeError:
        raise ValueError("Request is not valid JSON")
    history, k = payload.get("history"), payload.get("k", 10)
    if not isinstance(history, list) or not history:
        raise ValueError("History must contain at least one item ID")
    if any(type(i) is not int or not 1 <= i <= num_items for i in history):
        raise ValueError(f"Item IDs must be whole numbers from 1 to {num_items}")
    if type(k) is not int or not 1 <= k <= MAX_K:
        raise ValueError(f"k must be a whole number from 1 to {MAX_K}")
    return history, k


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
            self._send(200, {"model": RECOMMENDER.model_name, "dataset": DATASET_NAME,
                             "num_items": RECOMMENDER.num_items, "examples": EXAMPLES})
        else:
            self._send(404, {"error": "Not found"})

    def do_POST(self):
        if self.path != "/api/recommend":
            return self._send(404, {"error": "Not found"})
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        try:
            history, k = parse_request(body, RECOMMENDER.num_items)
        except ValueError as error:
            return self._send(400, {"error": str(error)})
        start = time.perf_counter()
        items = RECOMMENDER.recommend(history, k)
        latency = (time.perf_counter() - start) * 1000
        self._send(200, {"items": items, "latency_ms": round(latency, 1),
                         "history_used": history[-RECOMMENDER.max_len:]})


def main():
    global RECOMMENDER, EXAMPLES, DATASET_NAME
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/baby/best.pt")
    parser.add_argument("--dataset-name", default="baby", choices=["baby", "beauty", "ml-100k"])
    parser.add_argument("--data", default="data/baby/dataset.pkl")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--fake", action="store_true", help="random scores, no model needed")
    args = parser.parse_args()
    DATASET_NAME = args.dataset_name
    spec = dataset_spec(DATASET_NAME)
    data = load_dataset(args.data, spec["sha256"])
    EXAMPLES = pick_examples(data)
    RECOMMENDER = Recommender(args.checkpoint, spec["summary"]["max_item_id"], args.fake)
    print(f"Serving {RECOMMENDER.model_name} on {DATASET_NAME} at http://localhost:{args.port}")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()

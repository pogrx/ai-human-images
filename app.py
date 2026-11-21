"""Local web app for trying the AI-image detector.

Starts a small web server on this computer only (127.0.0.1) and opens it in
your browser. Drop in images to score them, or test the model on held-out
images it never trained on and see whether it gets them right.

    python app.py
    python app.py --checkpoint checkpoints/detector.pt --port 7860 --no-browser

Uses only the Python standard library on top of the project's requirements.
"""

import argparse
import base64
import io
import json
import random
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from PIL import Image, ImageOps

from data import dataset_split, load_rgb
from predict import Detector

ROOT = Path(__file__).resolve().parent
PAGES = {"/": ROOT / "ui" / "index.html", "/insights": ROOT / "ui" / "insights.html"}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
PREVIEW_MAX_SIDE = 900


def held_out_by_dataset(info, data_roots=None):
    """Images the checkpoint was validated on but never trained on, per dataset:
    {dataset name: [(path, label)]}. Defaults to the datasets the model was trained on."""
    split = info.get("split") or {}
    seed, fraction = split.get("seed", 0), split.get("val_fraction", 0.15)
    roots = data_roots or split.get("datasets") or ["data"]
    out = {}
    for root in roots:
        root = Path(root)
        root = root if root.is_absolute() else ROOT / root
        if root.is_dir():
            out[root.name] = dataset_split(root, fraction, seed)[1]
    return out


def held_out_samples(info, data_roots=None):
    """All held-out images across datasets: [(path, label)]."""
    return [s for samples in held_out_by_dataset(info, data_roots).values() for s in samples]


def preview_data_url(img):
    """Downscaled JPEG of exactly the pixels that were analysed (so overlays line up)."""
    thumb = img.copy()
    thumb.thumbnail((PREVIEW_MAX_SIDE, PREVIEW_MAX_SIDE))
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


class App:
    def __init__(self, checkpoint, data_roots=None):
        self.checkpoint = Path(checkpoint)
        self.detector = Detector(checkpoint)
        self.lock = threading.Lock()  # one inference at a time
        by_dataset = held_out_by_dataset(self.detector.info, data_roots)
        # (path, label, dataset name)
        self.samples = [(p, l, name) for name, samples in by_dataset.items() for p, l in samples]
        print(f"loaded {checkpoint}; held-out test images: "
              + ", ".join(f"{name} {len(s)}" for name, s in by_dataset.items()))

    def analyze(self, img):
        img = ImageOps.exif_transpose(img)  # match how phones/browsers orient photos
        start = time.perf_counter()
        with self.lock:
            result = self.detector.analyze(img)
        result.update(width=img.width, height=img.height, preview=preview_data_url(img),
                      ms=round((time.perf_counter() - start) * 1000))
        return result

    def info(self):
        i = self.detector.info
        m = i.get("val_metrics", {})
        names = list(dict.fromkeys(name for _, _, name in self.samples))
        per = m.get("per_dataset", {})
        return {
            "arch": i.get("arch"), "epoch": i.get("epoch"), "crop_size": i.get("crop_size"),
            "val_acc": m.get("acc"), "val_auc": m.get("auc"),
            "held_out": {"ai": sum(1 for _, l, _ in self.samples if l == 1),
                         "real": sum(1 for _, l, _ in self.samples if l == 0)},
            "datasets": [{"name": n,
                          "ai": sum(1 for _, l, d in self.samples if d == n and l == 1),
                          "real": sum(1 for _, l, d in self.samples if d == n and l == 0),
                          "val_acc": per.get(n, {}).get("acc"), "val_auc": per.get(n, {}).get("auc")}
                         for n in names],
        }

    def insights(self):
        """Results written by analyze.py, or None if it hasn't been run for this model."""
        path = self.checkpoint.with_name("insights.json")
        if not path.is_file():
            return None
        result = json.loads(path.read_text())
        result["stale"] = abs(result["model"]["checkpoint_mtime"] - self.checkpoint.stat().st_mtime) > 1
        return result

    def pick_samples(self, kind, n, dataset="any"):
        """Random held-out images. With dataset="any", each pick first chooses a dataset
        at random, so a huge dataset doesn't crowd out a small one."""
        pools = {}
        for i, (_, label, name) in enumerate(self.samples):
            if (dataset in ("any", name)) and (kind == "mixed" or label == (1 if kind == "ai" else 0)):
                pools.setdefault(name, []).append(i)
        picks = []
        while len(picks) < n and pools:
            name = random.choice(list(pools))
            pool = pools[name]
            picks.append(pool.pop(random.randrange(len(pool))))
            if not pool:
                del pools[name]
        return [{"id": i, "label": "ai" if self.samples[i][1] else "real", "dataset": self.samples[i][2],
                 "source": self.samples[i][0].name.split("_")[0], "name": self.samples[i][0].name}
                for i in picks]


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # keep the console quiet
            pass

        def send_json(self, obj, status=HTTPStatus.OK):
            body = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            query = parse_qs(url.query)
            if url.path in PAGES:
                body = PAGES[url.path].read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif url.path == "/api/info":
                self.send_json(app.info())
            elif url.path == "/api/insights":
                result = app.insights()
                if result is None:
                    return self.send_json({"error": "No insights for this model yet."}, HTTPStatus.NOT_FOUND)
                self.send_json(result)
            elif url.path == "/api/samples":
                kind = query.get("kind", ["mixed"])[0]
                if kind not in ("mixed", "ai", "real"):
                    return self.send_json({"error": "kind must be mixed, ai or real"}, HTTPStatus.BAD_REQUEST)
                n = max(1, min(24, int(query.get("n", ["6"])[0])))
                self.send_json(app.pick_samples(kind, n, query.get("dataset", ["any"])[0]))
            elif url.path == "/api/analyze-sample":
                try:
                    path, _, _ = app.samples[int(query["id"][0])]
                except (KeyError, ValueError, IndexError):
                    return self.send_json({"error": "unknown sample"}, HTTPStatus.NOT_FOUND)
                self.send_json(app.analyze(load_rgb(path)))
            else:
                self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self):
            if urlparse(self.path).path != "/api/analyze":
                return self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            length = int(self.headers.get("Content-Length") or 0)
            if not 0 < length <= MAX_UPLOAD_BYTES:
                return self.send_json({"error": "Image must be between 1 byte and 50 MB."},
                                      HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            data = self.rfile.read(length)
            try:
                img = load_rgb(io.BytesIO(data))
            except Exception:
                return self.send_json({"error": "That file couldn't be read as an image."},
                                      HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
            self.send_json(app.analyze(img))

    return Handler


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "detector.pt"))
    p.add_argument("--data", nargs="+", default=None,
                   help="dataset folders for held-out test images (default: the ones the model was trained on)")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--no-browser", action="store_true", help="don't open a browser window")
    args = p.parse_args()

    try:
        app = App(args.checkpoint, args.data)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(app))
    url = f"http://127.0.0.1:{args.port}"
    print(f"AI image detector running at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, [url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopped")


if __name__ == "__main__":
    main()

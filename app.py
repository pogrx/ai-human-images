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

from data import collect_samples, load_rgb, split_holdout
from predict import Detector

ROOT = Path(__file__).resolve().parent
PAGES = {"/": ROOT / "ui" / "index.html", "/insights": ROOT / "ui" / "insights.html"}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
PREVIEW_MAX_SIDE = 900


def held_out_samples(info, data_root):
    """Images the checkpoint was validated on but never trained on: [(path, label)]."""
    split = info.get("split") or {"seed": 0, "val_fraction": 0.15, "val_dir": None}
    if split.get("val_dir") and Path(split["val_dir"]).is_dir():
        return collect_samples(split["val_dir"])
    data_root = Path(data_root)
    train_dir = data_root / "train" if (data_root / "train").is_dir() else data_root
    if not train_dir.is_dir():
        return []
    _, val = split_holdout(collect_samples(train_dir), split["val_fraction"], split["seed"])
    return val


def preview_data_url(img):
    """Downscaled JPEG of exactly the pixels that were analysed (so overlays line up)."""
    thumb = img.copy()
    thumb.thumbnail((PREVIEW_MAX_SIDE, PREVIEW_MAX_SIDE))
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


class App:
    def __init__(self, checkpoint, data_root):
        self.checkpoint = Path(checkpoint)
        self.detector = Detector(checkpoint)
        self.lock = threading.Lock()  # one inference at a time
        self.samples = held_out_samples(self.detector.info, data_root)
        print(f"loaded {checkpoint}; {len(self.samples)} held-out test images available")

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
        return {
            "arch": i.get("arch"), "epoch": i.get("epoch"), "crop_size": i.get("crop_size"),
            "val_acc": m.get("acc"), "val_auc": m.get("auc"),
            "held_out": {"ai": sum(1 for _, l in self.samples if l == 1),
                         "real": sum(1 for _, l in self.samples if l == 0)},
        }

    def insights(self):
        """Results written by analyze.py, or None if it hasn't been run for this model."""
        path = self.checkpoint.with_name("insights.json")
        if not path.is_file():
            return None
        result = json.loads(path.read_text())
        result["stale"] = abs(result["model"]["checkpoint_mtime"] - self.checkpoint.stat().st_mtime) > 1
        return result

    def pick_samples(self, kind, n):
        pool = [i for i, (_, label) in enumerate(self.samples)
                if kind == "mixed" or label == (1 if kind == "ai" else 0)]
        picks = random.sample(pool, min(n, len(pool)))
        return [{"id": i, "label": "ai" if self.samples[i][1] else "real",
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
                self.send_json(app.pick_samples(kind, n))
            elif url.path == "/api/analyze-sample":
                try:
                    path, _ = app.samples[int(query["id"][0])]
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
    p.add_argument("--data", default=str(ROOT / "data"), help="dataset folder, used to find held-out test images")
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

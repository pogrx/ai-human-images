"""Measure what the detector relies on, for the Insights page of app.py.

Runs four experiments and writes the results to checkpoints/insights.json:

1. Damage tests. Held-out images are blurred, shrunk and re-enlarged,
   re-compressed as JPEG, or given added noise, at increasing strength. If the
   scores collapse when fine detail is destroyed, the model is reading fine
   detail rather than what the picture shows.
2. Detail swap. Each image is split into its coarse content (a blurred copy)
   and its fine detail (what the blur removed). Real content is combined with
   AI detail and vice versa. The score follows whichever part the model uses.
3. Frequency fingerprints. The average frequency spectrum of each generator's
   fine detail, next to the real photos'. Upsampling grids show up as bright
   dots or lines; JPEG's 8x8 blocks add a regular grid of dots to every group.
4. Per-generator scores on the held-out images.

    python analyze.py
    python analyze.py --checkpoint checkpoints/detector.pt --data data
"""

import argparse
import base64
import io
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

from app import held_out_samples
from data import collect_samples, load_rgb, pad_to, to_tensor
from predict import Detector
from train import roc_auc

ROOT = Path(__file__).resolve().parent

# Sequential blue ramp (light -> dark) used to colour the spectrum images.
RAMP = ["#f7fbff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b", "#06203f"]


def center_crop(img, size):
    img = pad_to(img, size)
    left, top = (img.width - size) // 2, (img.height - size) // 2
    return img.crop((left, top, left + size, top + size))


def source_of(path):
    return Path(path).name.split("_")[0]


# ---------- perturbations (PIL RGB image -> PIL RGB image) ----------

def blur(sigma):
    return lambda im: im.filter(ImageFilter.GaussianBlur(sigma))


def rescale(factor):
    def f(im):
        small = im.resize((max(1, round(im.width * factor)), max(1, round(im.height * factor))), Image.BICUBIC)
        return small.resize(im.size, Image.BICUBIC)
    return f


def jpeg(quality):
    def f(im):
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    return f


def noise(sigma, seed=0):
    def f(im):
        rng = np.random.default_rng(seed + im.width * 7 + int(np.asarray(im)[0, 0, 0]))
        arr = np.asarray(im, dtype=np.float32) + rng.normal(0, sigma, (im.height, im.width, 3))
        return Image.fromarray(np.clip(arr, 0, 255).round().astype(np.uint8))
    return f


DAMAGE_TESTS = {
    "blur": {"label": "Blur", "unit": "blur radius (px)", "levels": [0, 0.5, 1, 1.5, 2, 3], "make": blur},
    "rescale": {"label": "Shrink and re-enlarge", "unit": "shrunk to (% of size)",
                "levels": [100, 90, 75, 60, 50], "make": lambda pct: rescale(pct / 100)},
    "jpeg": {"label": "JPEG re-compression", "unit": "JPEG quality", "levels": [100, 95, 85, 75, 60, 40],
             "make": jpeg},
    "noise": {"label": "Added noise", "unit": "noise strength (of 255)", "levels": [0, 1, 2, 4, 8],
              "make": noise},
}
NO_CHANGE = {"blur": 0, "rescale": 100, "jpeg": 100, "noise": 0}


@torch.no_grad()
def score_crops(model, crops, batch=64):
    """List of PIL crops -> numpy array of AI confidences."""
    out = []
    for i in range(0, len(crops), batch):
        x = torch.stack([to_tensor(c) for c in crops[i:i + batch]])
        out.append(torch.sigmoid(model(x)).numpy())
    return np.concatenate(out)


def summarize(scores, labels):
    ai, real = scores[labels == 1], scores[labels == 0]
    return {
        "mean_ai": float(ai.mean()), "mean_real": float(real.mean()),
        "flag_ai": float((ai >= 0.5).mean()), "flag_real": float((real >= 0.5).mean()),
        "acc": float(((scores >= 0.5) == (labels == 1)).mean()), "auc": float(roc_auc(labels, scores)),
    }


def split_detail(im, sigma=2.0):
    """Image -> (coarse content, fine detail) as float arrays; content + detail == image."""
    arr = np.asarray(im, dtype=np.float32)
    coarse = np.asarray(im.filter(ImageFilter.GaussianBlur(sigma)), dtype=np.float32)
    return coarse, arr - coarse


def to_image(arr):
    return Image.fromarray(np.clip(arr, 0, 255).round().astype(np.uint8))


# ---------- frequency fingerprints ----------

def residual_spectrum(im):
    """Power spectrum of an image's fine detail (greyscale minus a 3x3 local mean)."""
    g = np.asarray(im.convert("L"), dtype=np.float32)
    local_mean = np.asarray(im.convert("L").filter(ImageFilter.BoxBlur(1)), dtype=np.float32)
    r = g - local_mean
    r -= r.mean()
    return np.abs(np.fft.fftshift(np.fft.fft2(r))) ** 2


def radial_profile(power):
    """Average power at each distance from the centre of a shifted spectrum (0 = coarse, 1 = finest)."""
    n = power.shape[0]
    y, x = np.indices(power.shape)
    r = np.hypot(x - n // 2, y - n // 2).astype(int)
    sums = np.bincount(r.ravel(), power.ravel())
    counts = np.bincount(r.ravel())
    prof = sums[: n // 2] / counts[: n // 2]
    return prof


def spectrum_png(power):
    """Log-scaled spectrum -> base64 PNG coloured with the blue ramp."""
    logp = np.log10(power + 1e-6)
    lo, hi = np.percentile(logp, [2, 99.8])
    t = np.clip((logp - lo) / (hi - lo + 1e-9), 0, 1)
    ramp = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in RAMP], dtype=np.float32)
    pos = t * (len(ramp) - 1)
    i0 = np.floor(pos).astype(int).clip(0, len(ramp) - 2)
    frac = (pos - i0)[..., None]
    rgb = ramp[i0] * (1 - frac) + ramp[i0 + 1] * frac
    buf = io.BytesIO()
    Image.fromarray(rgb.round().astype(np.uint8)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "detector.pt"))
    p.add_argument("--data", default=str(ROOT / "data"))
    p.add_argument("--spectrum-images", type=int, default=250, help="images per group for the fingerprints")
    p.add_argument("--out", default=None, help="default: insights.json next to the checkpoint")
    args = p.parse_args()

    start = time.time()
    det = Detector(args.checkpoint)
    model, size = det.model, det.crop_size
    samples = held_out_samples(det.info, args.data)
    if not samples:
        raise SystemExit("No held-out images found; check --data.")
    print(f"{len(samples)} held-out images")
    crops = [center_crop(load_rgb(path), size) for path, _ in samples]
    labels = np.array([label for _, label in samples])
    sources = [source_of(path) for path, _ in samples]

    # 1. damage tests
    damage = {}
    for key, test in DAMAGE_TESTS.items():
        rows = []
        for level in test["levels"]:
            changed = crops if level == NO_CHANGE[key] else [test["make"](level)(c) for c in crops]
            rows.append({"level": level, **summarize(score_crops(model, changed), labels)})
        damage[key] = {"label": test["label"], "unit": test["unit"], "rows": rows}
        print(f"damage test '{key}': " + ", ".join(f"{r['level']}->{r['acc']:.2f}" for r in rows))

    # 2. detail swap: pair each held-out AI image with a real one
    rng = np.random.default_rng(0)
    ai_idx = np.flatnonzero(labels == 1)
    real_idx = rng.permutation(np.flatnonzero(labels == 0))[: len(ai_idx)]
    ai_idx = ai_idx[: len(real_idx)]
    real_content_ai_detail, ai_content_real_detail = [], []
    for a, r in zip(ai_idx, real_idx):
        a_coarse, a_detail = split_detail(crops[a])
        r_coarse, r_detail = split_detail(crops[r])
        real_content_ai_detail.append(to_image(r_coarse + a_detail))
        ai_content_real_detail.append(to_image(a_coarse + r_detail))
    swap = []
    for name, content, detail, imgs in [
        ("Real photo, unchanged", "real", "real", [crops[i] for i in real_idx]),
        ("Real content + AI detail", "real", "ai", real_content_ai_detail),
        ("AI content + real detail", "ai", "real", ai_content_real_detail),
        ("AI image, unchanged", "ai", "ai", [crops[i] for i in ai_idx]),
    ]:
        s = score_crops(model, imgs)
        swap.append({"name": name, "content": content, "detail": detail,
                     "mean": float(s.mean()), "flag_ai": float((s >= 0.5).mean()), "n": len(s)})
    print("detail swap: " + ", ".join(f"{r['name']}={r['mean']:.2f}" for r in swap))

    # 3. per-generator scores on unchanged held-out images
    base = score_crops(model, crops)
    by_source = defaultdict(list)
    for s, src in zip(base, sources):
        by_source[src].append(float(s))
    bins = np.linspace(0, 1, 21)
    generators = []
    for src, s in sorted(by_source.items(), key=lambda kv: (kv[0] == "Real", -np.mean(kv[1]))):
        s = np.array(s)
        generators.append({"source": src, "is_ai": src != "Real", "n": len(s), "mean": float(s.mean()),
                           "median": float(np.median(s)), "flag_ai": float((s >= 0.5).mean()),
                           "hist": np.histogram(s, bins)[0].tolist()})

    # 4. frequency fingerprints over the whole dataset (a property of the images, not the model)
    data_root = Path(args.data)
    train_dir = data_root / "train" if (data_root / "train").is_dir() else data_root
    groups = defaultdict(list)
    for path, label in collect_samples(train_dir):
        groups[source_of(path)].append(path)
    fingerprints, profiles = [], {"ai": [], "real": []}
    for src in sorted(groups, key=lambda g: (g != "Real", g)):
        paths = groups[src][: args.spectrum_images]
        total = np.zeros((size, size))
        for path in paths:
            total += residual_spectrum(center_crop(load_rgb(path), size))
        mean_power = total / len(paths)
        prof = radial_profile(mean_power)
        profiles["real" if src == "Real" else "ai"].append(prof)
        fingerprints.append({"source": src, "n": len(paths), "png": spectrum_png(mean_power)})
        print(f"fingerprint {src}: {len(paths)} images")
    real_prof = np.mean(profiles["real"], axis=0)
    ai_prof = np.mean(profiles["ai"], axis=0)
    freqs = np.arange(len(real_prof)) / (len(real_prof) - 1)
    keep = slice(1, None)  # drop the constant (0-frequency) term
    energy = {"freq": freqs[keep].round(4).tolist(),
              "ai_vs_real": (ai_prof[keep] / real_prof[keep]).round(4).tolist()}

    m = det.info.get("val_metrics", {})
    result = {
        "created": time.strftime("%Y-%m-%d %H:%M"),
        "seconds": round(time.time() - start),
        "model": {"arch": det.info.get("arch"), "epoch": det.info.get("epoch"),
                  "val_acc": m.get("acc"), "val_auc": m.get("auc"),
                  "checkpoint_mtime": Path(args.checkpoint).stat().st_mtime},
        "held_out": {"ai": int((labels == 1).sum()), "real": int((labels == 0).sum())},
        "damage": damage, "swap": swap, "generators": generators, "hist_bins": bins.round(2).tolist(),
        "fingerprints": fingerprints, "energy": energy,
    }
    out = Path(args.out) if args.out else Path(args.checkpoint).with_name("insights.json")
    out.write_text(json.dumps(result))
    print(f"wrote {out} in {result['seconds']} s")


if __name__ == "__main__":
    main()

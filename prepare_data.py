"""Download Hugging Face parquet shards and unpack them into ai/ and real/ image folders.

Hugging Face datasets often ship as .parquet files: tables where each row holds the
raw image bytes plus a label. train.py needs plain image files in class folders, so
this script writes each row out as an image file:

    <out>/ai/GLIDE_v0_000012.jpg
    <out>/real/Real_v0_000013.jpg

Two fixes stop the network from learning shortcuts instead of "AI-ness":

* JPEG matching (on by default). In many datasets the real photos are JPEGs and the
  AI images are lossless PNGs, so "has no JPEG artifacts" alone would identify AI.
  Each non-JPEG AI image is re-saved as JPEG with a quality and chroma subsampling
  drawn from those measured on the real JPEGs, so both classes carry the same
  compression traces.
  Real images are never re-encoded. Turn off with --keep-format.
* Minimum size (default 256, train.py's default crop). Smaller images get
  reflect-padded during training, and if they mostly belong to one class (e.g.
  128 px BigGAN images) the padding itself becomes a giveaway. Images whose
  shorter side is below --min-size are skipped, in both classes. Use 0 to keep all.

Needs pyarrow (not in requirements.txt, since only this script uses it):
    .\\.venv\\Scripts\\python.exe -m pip install pyarrow

Examples:
    # Download 2 of the 4 Tiny-GenImage validation shards (~870 MB) into data/train
    python prepare_data.py --shards 2

    # Unpack parquet files you already downloaded
    python prepare_data.py --files path\\to\\a.parquet path\\to\\b.parquet --out data/train
"""

import argparse
import io
import json
import random
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

from PIL import Image

DEFAULT_REPO = "TheKernel01/Tiny-GenImage"
DEFAULT_SPLIT = "validation"
DEFAULT_NUM_FILES = 4  # Tiny-GenImage: validation-0000X-of-00004.parquet
AI_NAMES = {"fake", "ai", "generated", "synthetic", "1"}


def shard_url(repo, split, index, total):
    return (f"https://huggingface.co/datasets/{repo}/resolve/main/data/"
            f"{split}-{index:05d}-of-{total:05d}.parquet")


def download(url, dest):
    if dest.exists():
        print(f"already downloaded: {dest.name}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    print(f"downloading {url}")
    with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while chunk := resp.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.replace(dest)
    return dest


def image_extension(data):
    """Pick a file extension from the image's magic bytes."""
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    if data[:2] == b"BM":
        return ".bmp"
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return ".tiff"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return ".gif"
    return None


class JpegQualityEstimator:
    """Estimate a JPEG's quality setting by matching its luminance quantization table
    against the tables Pillow itself writes for qualities 1-100."""

    def __init__(self):
        self.reference = {}
        tiny = Image.new("RGB", (8, 8))
        for q in range(1, 101):
            buf = io.BytesIO()
            tiny.save(buf, format="JPEG", quality=q)
            self.reference[q] = list(Image.open(buf).quantization[0])

    def __call__(self, img):
        tables = getattr(img, "quantization", None)
        if not tables or 0 not in tables:
            return None
        table = list(tables[0])
        return min(self.reference, key=lambda q: sum(abs(a - b) for a, b in zip(table, self.reference[q])))


def class_names(schema, column):
    """Names for an integer label column, read from the Hugging Face metadata if present."""
    raw = (schema.metadata or {}).get(b"huggingface")
    if not raw:
        return None
    return json.loads(raw).get("info", {}).get("features", {}).get(column, {}).get("names")


def iter_rows(parquet_path):
    """Yield (image_bytes, is_ai, generator_name) for every row of a parquet file."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet_path)
    label_names = class_names(pf.schema_arrow, "label")
    gen_names = class_names(pf.schema_arrow, "generator")
    has_gen = "generator" in pf.schema_arrow.names
    columns = ["image", "label"] + (["generator"] if has_gen else [])
    for batch in pf.iter_batches(batch_size=64, columns=columns):
        for rec in batch.to_pylist():
            img = rec["image"]
            data = img.get("bytes") if isinstance(img, dict) else img
            label = rec["label"]
            name = label_names[label] if label_names and isinstance(label, int) else str(label)
            gen = rec.get("generator")
            gen = gen_names[gen] if gen_names and isinstance(gen, int) else (gen or name)
            yield data, name.lower() in AI_NAMES, gen


def real_jpeg_settings(files, estimator):
    """(quality, chroma subsampling) of every real colour JPEG, used as the distribution
    to re-save AI images at. Subsampling: 0 = 4:4:4, 1 = 4:2:2, 2 = 4:2:0."""
    from PIL import JpegImagePlugin

    settings = []
    for f in files:
        for data, is_ai, _ in iter_rows(f):
            if not is_ai and data and image_extension(data) == ".jpg":
                img = Image.open(io.BytesIO(data))
                q, sub = estimator(img), JpegImagePlugin.get_sampling(img)
                if q and sub in (0, 1, 2):  # -1: greyscale or unusual layout
                    settings.append((q, sub))
    return settings


def unpack(parquet_path, out_dir, tag, min_size, jpeg_settings, rng):
    counts = Counter()
    for row, (data, is_ai, gen) in enumerate(iter_rows(parquet_path)):
        ext = image_extension(data) if data else None
        if ext is None:
            counts["unreadable"] += 1
            continue
        img = Image.open(io.BytesIO(data))
        if min(img.size) < min_size:
            counts["too_small"] += 1
            continue
        cls = "ai" if is_ai else "real"
        if is_ai and jpeg_settings and ext != ".jpg":
            quality, subsampling = rng.choice(jpeg_settings)
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=quality, subsampling=subsampling)
            data, ext = buf.getvalue(), ".jpg"
        safe_gen = re.sub(r"[^A-Za-z0-9.-]+", "-", str(gen)).strip("-.") or "unknown"
        dest = out_dir / cls / f"{safe_gen}_{tag}_{row:06d}{ext}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        counts[cls] += 1
    return counts


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--files", nargs="*", type=Path, help="local parquet files to unpack (skips downloading)")
    p.add_argument("--repo", default=DEFAULT_REPO, help="Hugging Face dataset id")
    p.add_argument("--split", default=DEFAULT_SPLIT)
    p.add_argument("--num-files", type=int, default=DEFAULT_NUM_FILES,
                   help="how many shards the split has on Hugging Face")
    p.add_argument("--shards", type=int, default=2, help="how many shards to download (each ~430 MB)")
    p.add_argument("--cache", type=Path, default=Path("downloads"),
                   help="where downloaded parquet files are kept")
    p.add_argument("--out", type=Path, default=Path("data/train"))
    p.add_argument("--min-size", type=int, default=256,
                   help="skip images whose shorter side is smaller (0 keeps everything)")
    p.add_argument("--keep-format", action="store_true",
                   help="write AI images in their original format instead of matching the real JPEGs")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    try:
        import pyarrow  # noqa: F401
    except ImportError:
        sys.exit("pyarrow is missing. Install it with:  .\\.venv\\Scripts\\python.exe -m pip install pyarrow")

    if args.files:
        files = args.files
    else:
        n = min(args.shards, args.num_files)
        files = []
        for i in range(n):
            url = shard_url(args.repo, args.split, i, args.num_files)
            files.append(download(url, args.cache / Path(url).name))

    jpeg_settings = []
    if not args.keep_format:
        print("measuring JPEG settings of the real images ...")
        jpeg_settings = real_jpeg_settings(files, JpegQualityEstimator())
        if jpeg_settings:
            qs = sorted(q for q, _ in jpeg_settings)
            subs = Counter({0: "4:4:4", 1: "4:2:2", 2: "4:2:0"}[s] for _, s in jpeg_settings)
            print(f"  {len(qs)} real JPEGs, quality min {qs[0]}, median {qs[len(qs) // 2]}, max {qs[-1]}; "
                  f"chroma {dict(subs)}")
        else:
            print("  no real JPEGs found; AI images keep their original format")

    rng = random.Random(args.seed)
    total = Counter()
    for i, f in enumerate(files):
        tag = f"{args.split[0]}{i}" if not args.files else f.stem[:12]
        print(f"unpacking {f.name} ...")
        counts = unpack(f, args.out, tag, args.min_size, jpeg_settings, rng)
        print("  " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        total.update(counts)
    print(f"done: {total['ai']} ai, {total['real']} real images in {args.out}")
    if total["too_small"]:
        print(f"  skipped {total['too_small']} images under {args.min_size} px")
    if total["unreadable"]:
        print(f"  skipped {total['unreadable']} rows that are not a recognized image")


if __name__ == "__main__":
    main()

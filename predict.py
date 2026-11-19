"""Score images with a trained detector.

Prints one line per image: a confidence from 0 to 1 that the image is
AI-generated (1.0 = 100% AI, 0.0 = 0% AI), followed by the path.

Examples:
    python predict.py photo.jpg
    python predict.py my_folder --csv results.csv
    python predict.py a.png b.webp --checkpoint checkpoints/detector.pt
"""

import argparse
import csv
import sys
from pathlib import Path

import torch

from data import IMAGE_EXTENSIONS, list_images, load_rgb, multi_crops
from model import build_model


class Detector:
    """Load once, then call ``detector(path)`` to get a 0-1 AI confidence."""

    def __init__(self, checkpoint="checkpoints/detector.pt", max_crops=9, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(f"No trained model at {checkpoint}. Train one first: python train.py --data data")
        # weights_only=True: never execute code embedded in a downloaded checkpoint.
        ckpt = torch.load(checkpoint, map_location=self.device, weights_only=True)
        if ckpt.get("class_to_label", {}).get("ai") != 1:
            raise ValueError("Checkpoint label mapping is not ai=1; refusing to report inverted scores.")
        self.info = {k: v for k, v in ckpt.items() if k != "state_dict"}  # arch, epoch, val_metrics, split...
        self.crop_size = ckpt["crop_size"]
        self.max_crops = max_crops
        self.model = build_model(ckpt["arch"], pretrained=False).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    @torch.no_grad()
    def score_image(self, img):
        """PIL RGB image -> confidence in [0, 1] that it is AI-generated.

        Scores several native-resolution crops and averages them, so the whole
        image is considered without resizing away fine pixel detail.
        """
        return self.analyze(img)["confidence"]

    @torch.no_grad()
    def analyze(self, img):
        """Like ``score_image`` but also returns each crop's box and score:
        {"confidence": float, "regions": [{"box": (l, t, r, b), "score": float}, ...]}."""
        crops, boxes = multi_crops(img, self.crop_size, self.max_crops, return_boxes=True)
        scores = torch.sigmoid(self.model(crops.to(self.device))).tolist()
        return {
            "confidence": sum(scores) / len(scores),
            "regions": [{"box": box, "score": s} for box, s in zip(boxes, scores)],
        }

    def __call__(self, path):
        return self.score_image(load_rgb(path))


def expand_paths(inputs):
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            yield from list_images(p)
        elif p.is_file():
            yield p
        else:
            print(f"warning: {item} not found", file=sys.stderr)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", help="image files and/or folders")
    p.add_argument("--checkpoint", default="checkpoints/detector.pt")
    p.add_argument("--max-crops", type=int, default=9, help="crops averaged per image (more = slower, steadier)")
    p.add_argument("--csv", help="also write results to this CSV file")
    args = p.parse_args()

    try:
        detector = Detector(args.checkpoint, max_crops=args.max_crops)
    except FileNotFoundError as exc:
        sys.exit(str(exc))
    rows = []
    for path in expand_paths(args.inputs):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            print(f"warning: skipping non-image {path}", file=sys.stderr)
            continue
        try:
            conf = detector(path)
        except Exception as exc:
            print(f"warning: could not score {path}: {exc}", file=sys.stderr)
            continue
        rows.append((str(path), conf))
        print(f"{conf:.4f}  {path}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "ai_confidence"])
            writer.writerows((path, f"{conf:.6f}") for path, conf in rows)
        print(f"wrote {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()

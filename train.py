"""Train the AI-vs-real image detector.

Expected data layout (any image formats, any sizes, subfolders allowed):

    data/
      train/ai/...     AI-generated images
      train/real/...   authentic photos / artwork
      val/ai/...       (optional; if missing, a slice of train is held out)
      val/real/...

Example:
    python train.py --data data --epochs 20
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import (CLASS_TO_LABEL, EvalTransform, ImageDataset, TrainTransform, check_readable,
                  collect_samples)
from model import ARCHITECTURES, build_model


def roc_auc(labels, scores):
    """Area under the ROC curve (probability a random AI image outscores a random real one)."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    # average ranks for ties
    all_scores = np.concatenate([pos, neg])
    for value in np.unique(all_scores):
        tie = all_scores == value
        if tie.sum() > 1:
            ranks[tie] = ranks[tie].mean()
    return (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, labels, scores = 0.0, [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += criterion(logits, y).item() * len(y)
            labels += y.tolist()
            scores += torch.sigmoid(logits).tolist()
    labels, scores = np.array(labels), np.array(scores)
    return {
        "loss": float(total_loss / len(labels)),
        "acc": float(((scores >= 0.5) == (labels == 1)).mean()),
        "auc": float(roc_auc(labels, scores)),
        "mean_score_ai": float(scores[labels == 1].mean()) if (labels == 1).any() else float("nan"),
        "mean_score_real": float(scores[labels == 0].mean()) if (labels == 0).any() else float("nan"),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data",
                   help="folder containing train/ (and optionally val/ or test/ for validation)")
    p.add_argument("--arch", default="forensic", choices=ARCHITECTURES)
    p.add_argument("--no-pretrained", action="store_true", help="resnet18 only: start from random weights")
    p.add_argument("--crop-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=None, help="default: 1e-3 forensic, 3e-4 resnet18")
    p.add_argument("--val-fraction", type=float, default=0.15, help="used only when data/val is missing")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", default="checkpoints/detector.pt")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = Path(args.data)
    train_dir = data_root / "train" if (data_root / "train").is_dir() else data_root
    train_samples = collect_samples(train_dir)
    val_dir = next((data_root / n for n in ("val", "valid", "validation", "test")
                    if (data_root / n).is_dir()), None)
    if val_dir is not None:
        val_samples = collect_samples(val_dir)
        print(f"validating on {val_dir}")
    else:
        random.shuffle(train_samples)
        n_val = max(2, int(len(train_samples) * args.val_fraction))
        val_samples, train_samples = train_samples[:n_val], train_samples[n_val:]
        print(f"No {data_root / 'val'} folder; holding out {n_val} training images for validation.")

    print(f"checking {len(train_samples) + len(val_samples)} image files...")
    train_samples, bad_train = check_readable(train_samples)
    val_samples, bad_val = check_readable(val_samples)
    bad = bad_train + bad_val
    if bad:
        n_lfs = sum(reason == "git-lfs pointer" for _, reason in bad)
        for path, reason in bad[:5]:
            print(f"  {reason}: {path}")
        total = len(train_samples) + len(val_samples) + len(bad)
        print(f"skipping {len(bad)} of {total} files that could not be read as images")
        if n_lfs:
            print(f"{n_lfs} of them are Git LFS placeholders. Run `git lfs install` then `git lfs pull` "
                  f"in the dataset repository to download the real images.")
        if len(bad) > 0.05 * total:
            raise SystemExit("More than 5% of the dataset is unreadable; fix the files before training.")

    n_ai = sum(label for _, label in train_samples)
    n_real = len(train_samples) - n_ai
    if n_ai == 0 or n_real == 0:
        raise SystemExit(f"Need images in both train/ai and train/real (found ai={n_ai}, real={n_real}).")
    print(f"train: {n_ai} ai, {n_real} real | val: {sum(l for _, l in val_samples)} ai, "
          f"{len(val_samples) - sum(l for _, l in val_samples)} real | device: {device}")

    train_loader = DataLoader(ImageDataset(train_samples, TrainTransform(args.crop_size)),
                              batch_size=args.batch_size, shuffle=True, drop_last=len(train_samples) > args.batch_size,
                              num_workers=args.workers, persistent_workers=args.workers > 0)
    val_loader = DataLoader(ImageDataset(val_samples, EvalTransform(args.crop_size)),
                            batch_size=args.batch_size, num_workers=args.workers)

    model = build_model(args.arch, pretrained=not args.no_pretrained).to(device)
    lr = args.lr or (1e-3 if args.arch == "forensic" else 3e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=args.epochs,
                                                    steps_per_epoch=len(train_loader))
    # Weight the AI class so an imbalanced dataset doesn't bias the confidence.
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_real / n_ai, device=device))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        start, running, seen = time.time(), 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += loss.item() * len(y)
            seen += len(y)
        m = evaluate(model, val_loader, criterion, device)
        print(f"epoch {epoch:3d}/{args.epochs} | train loss {running / seen:.4f} | val loss {m['loss']:.4f} "
              f"acc {m['acc']:.3f} auc {m['auc']:.3f} | mean conf ai {m['mean_score_ai']:.2f} "
              f"real {m['mean_score_real']:.2f} | {time.time() - start:.0f}s")

        key = (m["auc"] if not np.isnan(m["auc"]) else m["acc"], -m["loss"])
        if best is None or key > best:
            best = key
            torch.save({
                "arch": args.arch,
                "crop_size": args.crop_size,
                "state_dict": model.state_dict(),
                "class_to_label": CLASS_TO_LABEL,
                "epoch": epoch,
                "val_metrics": m,
            }, out_path)
            print(f"  saved best model -> {out_path}")

    print(f"done. best val auc {best[0]:.3f}")


if __name__ == "__main__":
    main()

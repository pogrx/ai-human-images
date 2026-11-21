"""Train the AI-vs-real image detector on one or more datasets.

Each dataset is its own folder (any image formats, any sizes, subfolders allowed):

    data/                     data-lowqual/
      train/ai/...              ai/...
      train/real/...            real/...
      val/ai/...   (optional)
      val/real/...

A dataset may put its images in train/ or directly in the folder. If it has a
val/ (or valid/, validation/, test/) folder, that is used for validation;
otherwise 15% of its images are held out.

One model learns from all datasets together, with the datasets kept apart:
  * Every batch comes from a single dataset, so all its images share a crop
    size. Large images are cropped to --crop-size (256); datasets of smaller
    images are trained at their own size (e.g. 32x32) rather than padded up.
  * Each dataset gets the same number of batches per epoch, so a huge dataset
    of tiny images can't drown out a small dataset of large ones. Batches of
    small images hold more images, since each one is cheaper.
  * Every batch is half AI, half real.
  * Validation is reported for each dataset separately.

Examples:
    python train.py                              # uses data/ and data-lowqual/
    python train.py --data data --epochs 20      # one dataset only
"""

import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Sampler

from data import (CLASS_TO_LABEL, MIN_CROP, EvalTransform, ImageDataset, TrainTransform, check_readable,
                  dataset_split, pick_crop_size)
from model import ARCHITECTURES, build_model, recalibrate_bn

DEFAULT_DATASETS = ["data", "data-lowqual"]


def roc_auc(labels, scores):
    """Area under the ROC curve (probability a random AI image outscores a random real one)."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    n_pos, n_neg = int((labels == 1).sum()), int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # Rank all scores (ties share their average rank), then use the Mann-Whitney U statistic.
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    ranks = (ends - (counts - 1) / 2)[inverse]
    return (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


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


class DatasetBatchSampler(Sampler):
    """Batches of indices into a ConcatDataset: each batch from one dataset, half AI and half real.

    ``groups`` holds, per dataset, (index offset, AI indices, real indices, batch size).
    Each epoch every dataset contributes ``steps_per_dataset`` batches, in shuffled order.
    """

    def __init__(self, groups, steps_per_dataset, seed=0):
        self.groups, self.steps, self.seed, self.epoch = groups, steps_per_dataset, seed, 0
        self.pools = [[[], []] for _ in groups]  # per dataset: remaining AI, remaining real

    def __len__(self):
        return self.steps * len(self.groups)

    def _take(self, rng, d, cls, n):
        pool, source = self.pools[d][cls], self.groups[d][1 + cls]
        out = []
        while len(out) < n:
            if not pool:  # reshuffle when a class runs out, so every image is used before repeats
                pool.extend(rng.sample(source, len(source)))
            out.append(pool.pop())
        return out

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        order = [d for d in range(len(self.groups)) for _ in range(self.steps)]
        rng.shuffle(order)
        for d in order:
            offset, _, _, batch = self.groups[d]
            idx = self._take(rng, d, 0, batch // 2) + self._take(rng, d, 1, batch - batch // 2)
            rng.shuffle(idx)
            yield [offset + i for i in idx]


def check_files(name, samples):
    good, bad = check_readable(samples, spot_check=2000)
    if bad:
        n_lfs = sum(reason == "git-lfs pointer" for _, reason in bad)
        for path, reason in bad[:5]:
            print(f"  {reason}: {path}")
        total = len(samples)
        print(f"{name}: skipping {len(bad)} of {total} files that could not be read as images")
        if n_lfs:
            print(f"{n_lfs} of them are Git LFS placeholders. Run `git lfs install` then `git lfs pull` "
                  f"in the dataset repository to download the real images.")
        if len(bad) > 0.05 * total:
            raise SystemExit(f"More than 5% of {name} is unreadable; fix the files before training.")
    return good


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", nargs="+", default=None,
                   help=f"one or more dataset folders (default: {' '.join(DEFAULT_DATASETS)}, whichever exist)")
    p.add_argument("--arch", default="forensic", choices=ARCHITECTURES)
    p.add_argument("--no-pretrained", action="store_true", help="resnet18 only: start from random weights")
    p.add_argument("--crop-size", type=int, default=256, help="largest training crop; smaller images use their own size")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16, help="batch size at the largest crop size")
    p.add_argument("--max-batch", type=int, default=256, help="cap on the batch size for small images")
    p.add_argument("--lr", type=float, default=None, help="default: 1e-3 forensic, 3e-4 resnet18")
    p.add_argument("--val-fraction", type=float, default=0.15, help="used for datasets without a val folder")
    p.add_argument("--max-val", type=int, default=2000, help="validation images scored per dataset each epoch")
    p.add_argument("--bn-images", type=int, default=1024,
                   help="training images per dataset used to measure its own BatchNorm stats each epoch")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default="checkpoints/detector.pt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true",
                   help="continue an interrupted run (same --out, --data and --epochs) from its last finished epoch")
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    roots = args.data or [d for d in DEFAULT_DATASETS if Path(d).is_dir()]
    if not roots:
        raise SystemExit("No dataset folders found; pass them with --data.")

    datasets = []
    for root in roots:
        name = Path(root).name
        train, val = dataset_split(root, args.val_fraction, args.seed)
        train, val = check_files(name, train), check_files(name, val)
        n_ai = sum(label for _, label in train)
        if n_ai == 0 or n_ai == len(train):
            raise SystemExit(f"{name}: need both AI and real training images (found {n_ai} AI of {len(train)}).")
        crop = pick_crop_size(train, args.crop_size, seed=args.seed)
        batch = min(args.max_batch, max(2, int(args.batch_size * (args.crop_size / crop) ** 2)))
        batch -= batch % 2
        val_eval = random.Random(args.seed).sample(val, min(args.max_val, len(val)))
        datasets.append({"name": name, "root": str(root), "train": train, "val": val, "val_eval": val_eval,
                         "crop": crop, "batch": batch, "n_ai": n_ai})
        print(f"{name}: train {n_ai} ai / {len(train) - n_ai} real | held out {len(val)} "
              f"(scoring {len(val_eval)} per epoch) | crop {crop}px, batch {batch}")

    # One ConcatDataset; the sampler keeps each batch inside one dataset.
    parts, groups, offset = [], [], 0
    for d in datasets:
        parts.append(ImageDataset(d["train"], TrainTransform(d["crop"])))
        ai = [i for i, (_, label) in enumerate(d["train"]) if label == 1]
        real = [i for i, (_, label) in enumerate(d["train"]) if label == 0]
        groups.append((offset, ai, real, d["batch"]))
        offset += len(d["train"])
    # Steps per dataset per epoch: enough for one pass over the smallest dataset.
    steps = max(1, min(math.ceil(len(d["train"]) / d["batch"]) for d in datasets))
    sampler = DatasetBatchSampler(groups, steps, args.seed)
    train_loader = DataLoader(ConcatDataset(parts), batch_sampler=sampler,
                              num_workers=args.workers, persistent_workers=args.workers > 0)
    val_loaders = {d["name"]: DataLoader(ImageDataset(d["val_eval"], EvalTransform(d["crop"])),
                                         batch_size=d["batch"], num_workers=args.workers)
                   for d in datasets}
    # Training images used to measure each dataset's own BatchNorm statistics (see model.recalibrate_bn).
    bn_loaders = {d["name"]: DataLoader(
        ImageDataset(random.Random(args.seed + 1).sample(d["train"], min(args.bn_images, len(d["train"]))),
                     EvalTransform(d["crop"])), batch_size=d["batch"], num_workers=args.workers)
        for d in datasets}
    print(f"{len(sampler)} batches per epoch ({steps} per dataset) | device: {device}")

    model = build_model(args.arch, pretrained=not args.no_pretrained).to(device)
    lr = args.lr or (1e-3 if args.arch == "forensic" else 3e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=args.epochs,
                                                    steps_per_epoch=len(sampler))
    criterion = nn.BCEWithLogitsLoss()  # batches are already class-balanced

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Full training state, saved after every epoch so an interrupted run can continue with --resume.
    resume_path = out_path.with_name(out_path.stem + ".resume.pt")
    best, first_epoch = None, 1
    if args.resume:
        if not resume_path.is_file():
            raise SystemExit(f"Nothing to resume: {resume_path} doesn't exist.")
        state = torch.load(resume_path, map_location=device, weights_only=True)
        if state["epochs"] != args.epochs:
            raise SystemExit(f"That run was started with --epochs {state['epochs']}; pass the same value to resume.")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        sampler.epoch = state["sampler_epoch"]
        best = tuple(state["best"]) if state["best"] is not None else None
        first_epoch = state["epoch"] + 1
        print(f"resuming after epoch {state['epoch']}")
    for epoch in range(first_epoch, args.epochs + 1):
        model.train()
        start, running, seen = time.time(), 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += loss.item()
            seen += 1
        # Score each dataset with BatchNorm stats measured on that dataset alone.
        bn_stats, per = {}, {}
        for d in datasets:
            name = d["name"]
            stats = recalibrate_bn(model, (x.to(device) for x, _ in bn_loaders[name]))
            bn_stats[name] = {"crop": d["crop"], "state": {k: v.cpu() for k, v in stats.items()}}
            per[name] = evaluate(model, val_loaders[name], criterion, device)
        # Overall = average over datasets, so each dataset counts equally.
        overall = {k: float(np.mean([m[k] for m in per.values()])) for k in ("loss", "acc", "auc")}
        print(f"epoch {epoch:3d}/{args.epochs} | train loss {running / seen:.4f} | "
              + " | ".join(f"{n}: acc {m['acc']:.3f} auc {m['auc']:.3f}" for n, m in per.items())
              + f" | {time.time() - start:.0f}s")

        key = (overall["auc"], -overall["loss"])
        if best is None or key > best:
            best = key
            torch.save({
                "arch": args.arch,
                "crop_size": args.crop_size,
                # Images smaller than crop_size are scored at their own size, down to min_crop.
                "min_crop": MIN_CROP,
                "state_dict": model.state_dict(),
                "class_to_label": CLASS_TO_LABEL,
                "epoch": epoch,
                # How validation images were chosen, so tools (e.g. app.py) can find held-out images.
                "split": {"seed": args.seed, "val_fraction": args.val_fraction,
                          "datasets": [d["root"] for d in datasets]},
                "datasets": [{"name": d["name"], "root": d["root"], "crop": d["crop"],
                              "n_train": len(d["train"]), "n_val": len(d["val"])} for d in datasets],
                "val_metrics": {**overall, "per_dataset": per},
                # Per-dataset BatchNorm stats; predict.Detector picks the set whose crop size fits the image.
                "bn_stats": bn_stats,
            }, out_path)
            print(f"  saved best model -> {out_path}")

        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "sampler_epoch": sampler.epoch,
                    "epoch": epoch, "epochs": args.epochs, "best": list(best)}, resume_path)

    resume_path.unlink(missing_ok=True)
    print(f"done. best average val auc {best[0]:.3f}")


if __name__ == "__main__":
    main()

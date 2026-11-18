"""Image loading, augmentation and datasets.

Only pixel values are used: images are decoded, converted to RGB, and all
metadata (EXIF, file names, file sizes, formats) is ignored.

Label convention used everywhere in this project:  ai = 1,  real = 0.
"""

import io
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile, ImageFilter
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
CLASS_TO_LABEL = {"ai": 1, "real": 0}


def load_rgb(path):
    """Decode an image file to an RGB PIL image (handles RGBA, palette, grayscale, CMYK)."""
    with Image.open(path) as img:
        img.seek(0)  # first frame of animated GIF/WebP
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            img = img.convert("RGBA")
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.getchannel("A"))
            return background
        return img.convert("RGB")


def to_tensor(img):
    """PIL RGB image -> float tensor (3, H, W) in [0, 1]."""
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr.copy()).permute(2, 0, 1)


def pad_to(img, size):
    """Reflect-pad an image that is smaller than ``size`` in either dimension.

    Small images are padded (never upscaled), because interpolation would
    itself add exactly the kind of artifacts the detector looks for.
    """
    w, h = img.size
    if w >= size and h >= size:
        return img
    arr = np.asarray(img)
    pad_h, pad_w = max(0, size - h), max(0, size - w)
    pads = ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0))
    mode = "reflect" if (h > 1 and w > 1 and pad_h < h and pad_w < w) else "edge"
    return Image.fromarray(np.pad(arr, pads, mode=mode))


def random_jpeg(img, quality_range=(60, 100)):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=random.randint(*quality_range))
    buf.seek(0)
    with Image.open(buf) as out:
        return out.convert("RGB")


class TrainTransform:
    """Augmentations applied identically to both classes.

    Random JPEG re-compression, mild rescaling and blur stop the network from
    learning shortcuts like "PNG = AI, JPEG = real" or "big = real"; it has to
    find generator artifacts that survive these everyday operations.
    """

    def __init__(self, crop_size=256, p_jpeg=0.5, p_resize=0.25, p_blur=0.1):
        self.crop_size = crop_size
        self.p_jpeg, self.p_resize, self.p_blur = p_jpeg, p_resize, p_blur

    def __call__(self, img):
        if random.random() < self.p_resize:
            scale = random.uniform(0.75, 1.25)
            w, h = img.size
            img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                             random.choice([Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]))
        if random.random() < self.p_blur:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 1.0)))
        if random.random() < self.p_jpeg:
            img = random_jpeg(img)
        img = pad_to(img, self.crop_size)
        w, h = img.size
        left = random.randint(0, w - self.crop_size)
        top = random.randint(0, h - self.crop_size)
        img = img.crop((left, top, left + self.crop_size, top + self.crop_size))
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return to_tensor(img)


class EvalTransform:
    """Deterministic centre crop at native resolution (no resizing)."""

    def __init__(self, crop_size=256):
        self.crop_size = crop_size

    def __call__(self, img):
        img = pad_to(img, self.crop_size)
        w, h = img.size
        left, top = (w - self.crop_size) // 2, (h - self.crop_size) // 2
        return to_tensor(img.crop((left, top, left + self.crop_size, top + self.crop_size)))


def multi_crops(img, crop_size=256, max_crops=9):
    """Up to ``max_crops`` native-resolution crops on an even grid covering the image."""
    img = pad_to(img, crop_size)
    w, h = img.size
    n = max(1, int(max_crops ** 0.5))
    nx = min(n, max(1, w // crop_size + (w % crop_size > 0)))
    ny = min(n, max(1, h // crop_size + (h % crop_size > 0)))
    xs = np.linspace(0, w - crop_size, nx).round().astype(int) if nx > 1 else [(w - crop_size) // 2]
    ys = np.linspace(0, h - crop_size, ny).round().astype(int) if ny > 1 else [(h - crop_size) // 2]
    crops = [to_tensor(img.crop((x, y, x + crop_size, y + crop_size))) for y in ys for x in xs]
    return torch.stack(crops)


def list_images(folder):
    folder = Path(folder)
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


# Folder names recognised as each class (case-insensitive). Public datasets use
# many conventions, e.g. ai/real, FAKE/REAL, 1_fake/0_real, generated/human.
AI_FOLDER_NAMES = {"ai", "fake", "fakes", "1_fake", "generated", "ai_generated", "aigenerated",
                   "ai-generated", "synthetic", "synth"}
REAL_FOLDER_NAMES = {"real", "reals", "0_real", "authentic", "human", "natural", "nature",
                     "original", "genuine", "photo", "photos"}


def folder_label(name):
    name = name.lower()
    if name in AI_FOLDER_NAMES:
        return 1
    if name in REAL_FOLDER_NAMES:
        return 0
    return None


def collect_samples(root):
    """Return [(path, label)] for every image under a class folder inside ``root``.

    Class folders may sit directly in ``root`` (root/ai, root/real) or be
    nested, e.g. root/<category>/1_fake and root/<category>/0_real.
    AI folders get label 1, real folders label 0.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"{root} does not exist")
    samples, class_dirs = [], []
    for d in sorted([root] + [p for p in root.rglob("*") if p.is_dir()]):
        label = folder_label(d.name) if d != root else None
        # Skip folders nested inside a class folder we've already taken.
        if label is not None and not any(parent in class_dirs for parent in d.parents):
            class_dirs.append(d)
            samples += [(p, label) for p in list_images(d)]
    if not class_dirs:
        raise FileNotFoundError(
            f"No class folders found in {root}. Expected subfolders named like "
            f"'ai'/'fake'/'1_fake' (AI) and 'real'/'0_real'/'authentic' (real).")
    return samples


def check_readable(samples):
    """Split samples into (readable, unreadable) by attempting to open each file.

    Catches Git LFS pointer files (small text stubs left when a repository is
    cloned without ``git lfs pull``) as well as corrupt images.
    """
    good, bad = [], []
    for path, label in samples:
        try:
            with Image.open(path) as img:
                img.verify()
            good.append((path, label))
        except Exception:
            try:
                is_lfs = Path(path).read_bytes()[:40].startswith(b"version https://git-lfs")
            except OSError:
                is_lfs = False
            bad.append((path, "git-lfs pointer" if is_lfs else "unreadable"))
    return good, bad


class ImageDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        for _ in range(10):
            path, label = self.samples[idx]
            try:
                img = load_rgb(path)
                return self.transform(img), torch.tensor(label, dtype=torch.float32)
            except Exception as exc:  # unreadable file: substitute a random other sample
                print(f"warning: could not read {path} ({exc}); substituting another image")
                idx = random.randrange(len(self))
        raise RuntimeError("Too many unreadable images in a row; check the dataset files.")

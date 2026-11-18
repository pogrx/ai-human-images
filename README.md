# ai-human-images
Neural network to distinguish ai and human images

This network looks at an image's pixels and returns a confidence from **0 to 1** that the image is AI-generated:

- **1.0**: 100% AI
- **0.0**: 0% AI (authentic)

It uses only decoded pixel values. EXIF data, file names, file sizes and formats are ignored.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If you have an NVIDIA GPU, install the CUDA build of PyTorch first (see pytorch.org). The scripts use the GPU automatically.

## 1. Add training data

```
data/
  train/
    ai/      AI-generated images
    real/    authentic images
  val/       optional; if it's missing, 15% of train is held out
    ai/
    real/
```

- **Formats:** any common format works (jpg, png, webp, bmp, tiff, gif), at any size.
- **Folder names:** the usual dataset naming conventions are recognized, case-insensitively.
  - AI: `ai`, `fake`, `1_fake`, `generated`, `synthetic`
  - Real: `real`, `0_real`, `authentic`, `human`
- **Nesting:** class folders can sit inside category folders, e.g. `train/cats/1_fake`.
- **Validation split:** a `val/`, `valid/`, `validation/` or `test/` folder is used for validation if present.
- **Other locations:** point `--data` at any folder with this layout. For example, if the dataset is a cloned GitHub repository:

```powershell
.\.venv\Scripts\python.exe train.py --data ..\my-dataset-repo
```

**Datasets on GitHub usually store images with Git LFS.** If you clone without it, you get tiny text placeholders instead of images. Training detects these and tells you. To fix it, run this in the dataset repository:

```powershell
git lfs install
git lfs pull
```

Tips for a detector that actually generalizes:
- **Match the two classes on everything except "AI-ness".** Use similar subjects, similar resolutions, and the same mix of file formats. If every AI image is a PNG and every real one is a JPEG, the network may learn to spot the format instead of the AI. Training applies random JPEG re-compression and resizing to both classes to reduce this.
- **Mix several generators.** Use several different image models, not just one. A detector trained on a single generator often misses the others.
- **Use more images.** A few thousand per class is a reasonable start.
- **Avoid tiny datasets.** Very low-resolution datasets (such as 32×32 images) don't transfer to full-size photos.

## 2. Train

```powershell
.\.venv\Scripts\python.exe train.py --data data --epochs 20
```

| Option | Default | Notes |
|---|---|---|
| `--arch` | `forensic` | `forensic` is a compact CNN with fixed high-pass filters and is fast on a CPU. `resnet18` starts from ImageNet weights (downloaded on first use, about 45 MB) and usually generalizes better given enough data. |
| `--crop-size` | 256 | Images are cropped at native resolution, never shrunk, so fine generator artifacts survive. |
| `--epochs` | 20 | |
| `--batch-size` | 16 | |
| `--workers` | 2 | Data-loading processes. Use 0 if you hit multiprocessing errors. |
| `--out` | `checkpoints/detector.pt` | |

Each epoch prints validation loss, accuracy, AUC, and the mean confidence for each class. The checkpoint with the best validation AUC is kept.

## 3. Predict

```powershell
.\.venv\Scripts\python.exe predict.py some_image.jpg
.\.venv\Scripts\python.exe predict.py folder_of_images --csv results.csv
```

Output is one line per image:

```
0.9821  images/generated_cat.png
0.0412  images/holiday_photo.jpg
```

For large images, several native-resolution crops (up to 9, set with `--max-crops`) are scored and averaged.

You can also call it from Python:

```python
from predict import Detector
detector = Detector("checkpoints/detector.pt")
print(detector("photo.jpg"))   # e.g. 0.07
```

## Files

- `model.py`: network architectures (`ForensicCNN`, `ResNetDetector`)
- `data.py`: image loading, augmentation, cropping. Labels are `ai = 1`, `real = 0`.
- `train.py`: training loop
- `predict.py`: command-line and Python inference

## Limitations

A score is a statistical estimate, not proof. Accuracy depends entirely on the training data. Expect weaker results on:
- images from generators the model never saw,
- heavily compressed, resized or screenshotted images,
- AI-edited real photos.

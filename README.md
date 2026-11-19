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

The training images are included in this repository under `data/train` (about 415 MB): 1,500 AI images from 6 generators and 1,707 real photos. They come from [Tiny-GenImage](https://huggingface.co/datasets/TheKernel01/Tiny-GenImage), 2 of its 4 validation shards. The real photos are from ImageNet, whose terms don't allow public redistribution, so keep this repository private or remove `data/` before making it public.

To rebuild or extend the set, download it from Hugging Face. Use `--shards 4` for the full validation split.

```powershell
.\.venv\Scripts\python.exe -m pip install pyarrow
.\.venv\Scripts\python.exe prepare_data.py --shards 2
```

`prepare_data.py` re-saves the AI images (originally PNGs) as JPEGs with the same quality settings as the real photos, so the network can't tell them apart by file format alone. Run it with `--help` for other datasets and options.

Or use your own images, laid out like this:

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

## 4. Try it in your browser

```powershell
.\.venv\Scripts\python.exe app.py
```

This opens a local page at http://127.0.0.1:7860. It runs only on your computer; nothing is uploaded anywhere. There you can:
- **Score your own images:** drag and drop, pick files, or paste with Ctrl+V. Each image gets a 0–100% AI score and a verdict.
- **See the heatmap:** it shows which parts of the image looked AI (orange) or real (green).
- **Quiz the model:** score random held-out images it never trained on, see whether each answer was right, and keep a running tally.

### Insights: what the model relies on

```powershell
.\.venv\Scripts\python.exe analyze.py
```

This runs experiments on the held-out images (about 5 minutes on a CPU) and fills the app's **Insights** tab with charts:
- **Damage tests:** how scores change when images are blurred, shrunk, re-compressed or given noise.
- **Detail swap:** what the model does when one image's content is combined with another image's fine detail.
- **Detail energy:** how much fine detail AI images have compared with real photos, at each scale.
- **Fingerprints:** each generator's frequency fingerprint.
- **Results by source:** the share of each generator's images flagged as AI.

Run it again after retraining; the page warns you when its results are out of date.

The main finding for the current model: it judges images almost entirely by their finest pixel-level detail, not by what they show. A 1-pixel blur drops its accuracy to a coin flip, and real photos that have been blurred or shrunk tend to be flagged as AI.

Press Ctrl+C in the terminal to stop the app. Use `--port` to pick another port, or `--checkpoint` to load a different trained model.

You can also call the detector from Python:

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
- `app.py`, `ui/index.html`, `ui/insights.html`: local web app for testing the model and viewing insights
- `analyze.py`: experiments behind the Insights tab
- `prepare_data.py`: downloads and unpacks the dataset

## Limitations

A score is a statistical estimate, not proof. Accuracy depends entirely on the training data. Expect weaker results on:
- images from generators the model never saw,
- heavily compressed, resized or screenshotted images,
- AI-edited real photos.

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

The repository holds two datasets, each in its own folder. Training uses both by default:

| Folder | Images | Size | Source |
|---|---|---|---|
| `data/train/{ai,real}` | 1,500 AI (6 generators), 1,707 real | full-size (256–1024 px) | Tiny-GenImage subset |
| `data-lowqual/{ai,real}` | 60,000 AI, 60,000 real | 32×32 | low-quality set added to the repo; file names start with the split it came from (`LowQual_train_…`, `Real_test_…`) |

The full-size set takes about 415 MB: 1,500 AI images from 6 generators and 1,707 real photos. They come from [Tiny-GenImage](https://huggingface.co/datasets/TheKernel01/Tiny-GenImage), 2 of its 4 validation shards. The real photos are from ImageNet, whose terms don't allow public redistribution, so keep this repository private or remove `data/` before making it public.

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
- **Tiny images are learned separately from large ones.** A model trained only on 32×32 images doesn't transfer to full-size photos, and the reverse is also true. Training on both datasets together, as below, handles both kinds.

## 2. Train

```powershell
.\.venv\Scripts\python.exe train.py
```

With no `--data`, this trains one model on both `data` and `data-lowqual` (whichever exist). To use particular datasets, list them: `--data data`, or `--data data data-lowqual ..\another-dataset`.

How the datasets are combined, while staying separate:
- **One dataset per batch.** Each batch of images comes from just one dataset, so every image in it has the same crop size. Full-size images are cropped to 256 px. The 32×32 images are trained at their own size instead of being padded up, so padding can't become a shortcut.
- **Equal say.** Each dataset gets the same number of training steps per epoch, so the 120,000 small images don't drown out the 3,200 large ones. Batches of small images hold more images (256 vs 16), because each one is much cheaper to process.
- **Balanced batches.** Every batch is half AI, half real.
- **Separate normalization statistics.** The network's BatchNorm layers keep running averages of what they see. One shared average fitted neither the 32 px nor the 256 px images well; in a test it cost 3–5 points of accuracy. So after each epoch these averages are measured separately for each dataset and both sets are saved. When predicting, the set whose crop size is closest to the image's size is used.
- **Separate held-out sets.** Each dataset holds out 15% of its images, or uses its own `val/` folder if it has one. Accuracy is reported for each dataset every epoch.

| Option | Default | Notes |
|---|---|---|
| `--data` | `data data-lowqual` | One or more dataset folders. |
| `--arch` | `forensic` | `forensic` is a compact CNN with fixed high-pass filters and is fast on a CPU. `resnet18` starts from ImageNet weights (downloaded on first use, about 45 MB) and usually generalizes better given enough data. |
| `--crop-size` | 256 | Largest crop. Images are cropped at native resolution, never shrunk, so fine generator artifacts survive. |
| `--epochs` | 20 | Each epoch takes about 6 minutes on a CPU with both datasets. |
| `--batch-size` | 16 | Batch size at 256 px. Smaller images get proportionally bigger batches, up to `--max-batch` (256). |
| `--max-val` | 2000 | Held-out images scored per dataset each epoch, to keep epochs quick. |
| `--workers` | 4 | Data-loading processes. Use 0 if you hit multiprocessing errors. |
| `--out` | `checkpoints/detector.pt` | |

A run on a CPU takes about 4 minutes per epoch. It saves its full progress after every epoch. If the run is interrupted (the computer sleeps, the window closes), run the same command again with `--resume` to continue from the last finished epoch.

Current model: 20 epochs on both datasets. Held-out results, scored through `predict.py`:

| Dataset | Accuracy | AUC | AI images caught | Real images wrongly flagged |
|---|---|---|---|---|
| `data` (481 images) | 91.3% | 0.978 | 97% | 14% |
| `data-lowqual` (4,000 sampled) | 93.7% | 0.984 | 95% | 8% |

For comparison, a model trained on `data` alone scored 91.7% on `data` but only 49% on `data-lowqual`, no better than guessing.

The checkpoint with the best average AUC across datasets is kept. When predicting, images smaller than 256 px are scored at their own size, matching how they were trained.

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

This opens a local page at http://127.0.0.1:7860, after about 25 seconds spent listing the held-out images of both datasets. It runs only on your computer; nothing is uploaded anywhere. There you can:
- **Score your own images:** drag and drop, pick files, or paste with Ctrl+V. Each image gets a 0–100% AI score and a verdict.
- **See the heatmap:** it shows which parts of the image looked AI (orange) or real (green).
- **Quiz the model:** score random held-out images it never trained on, see whether each answer was right, and keep a running tally.

### Insights: what the model relies on

```powershell
.\.venv\Scripts\python.exe analyze.py
```

This runs experiments on one dataset's held-out images (`--data data` by default; use `--data data-lowqual` for the small images). It takes about 5 minutes on a CPU and fills the app's **Insights** tab with charts:
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

# 🎬 Image-to-Video Generation on Kaggle

A Kaggle notebook demonstrating local video generation using **Stable Video Diffusion (SVD)** by Stability AI.
The notebook takes a single input image and generates multiple videos with different random seeds,
stamps each video with its seed as a watermark, and merges them all into one final video.

---

## 📋 Project Overview

| Property | Value |
|---|---|
| Model | `stabilityai/stable-video-diffusion-img2vid-xt` |
| Task | Image-to-Video generation |
| Platform | Kaggle (free T4 GPU) |
| Output | Merged `.mp4` with watermarked segments per seed |

---

## 🗂️ File Structure

```
├── llma-sp.ipynb       # Main Kaggle notebook
└── README.md            # This file
```

---

## ⚙️ Requirements

All dependencies are installed directly inside the notebook. No local setup needed.

| Library | Purpose |
|---|---|
| `diffusers` | Loads and runs the SVD pipeline |
| `transformers` | Required by diffusers internally |
| `accelerate` | Enables CPU offloading to save GPU memory |
| `imageio[ffmpeg]` | Exports frames to `.mp4` |
| `Pillow` | Image loading, resizing, and watermark drawing |
| `torch` | PyTorch — runs the neural network on GPU |

---

## 🚀 How to Run

### 1. Open on Kaggle

- Go to [kaggle.com](https://www.kaggle.com) and create a new notebook
- Copy the code from `notebook.ipynb` into the cells

### 2. Enable GPU

In the Kaggle notebook sidebar:
```
Settings → Accelerator → GPU T4 x1 → Save
```

### 3. Run the cells in order

```
Cell 1 — Install libraries
Cell 2 — Load model
Cell 3 — Load input image
Cell 4 — Generate videos in a loop
Cell 5 — Display merged video
```

The full run takes approximately **15–30 minutes** on a Kaggle T4 GPU depending on the number of seeds.

---

## 🔧 Configuration

You can adjust these variables at the top of the generation loop:

```python
SEEDS = [42, 123, 999]   # one video generated per seed — add or remove as needed
```

And these parameters inside `pipe(...)`:

```python
num_frames=25            # number of frames per video segment
num_inference_steps=25   # more steps = better quality, slower
motion_bucket_id=127     # 1–255: controls how much motion is generated
noise_aug_strength=0.02  # how closely output follows the input image
```

---

## 📦 How It Works

```
Input Image (1024×576)
        ↓
  VAE Encoder — compresses image into latent space
        ↓
  UNet Denoising — 25 steps of noise removal, guided by the image
        ↓
  VAE Decoder — converts latents back to pixel frames
        ↓
  Watermark — seed number stamped on each frame
        ↓
  Repeat for each seed in SEEDS
        ↓
  All frames merged → output_merged.mp4
```

SVD is a **latent diffusion model** — generation happens in a compressed latent space
(~8× smaller than pixel space), which makes it feasible to run on a 15 GB GPU.

---

## ⚠️ Known Limitations

- Requires exactly **1024×576** input image resolution
- Each generation takes ~5–10 minutes on Kaggle T4
- Audio is not generated — SVD produces silent video only
- The model sometimes struggles with complex hand or face motion

---

## 📚 References

- [Stable Video Diffusion paper](https://arxiv.org/abs/2311.15127)
- [Hugging Face model card](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt)
- [Diffusers documentation](https://huggingface.co/docs/diffusers)

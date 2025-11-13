✅ Two separate nodes

* **Zoom Sequence (Single Batch)**
* **Zoom Sequence (Batched, with persistent zoom state)**

✅ Updated installation & file list
✅ Clear explanation of what each node does
✅ Matches your two-file setup (`comfy_zoom_sequence.py` + `batch_comfy_zoom_sequence.py`)

---

# 🌀 ComfyUI Zoom Sequence Nodes

A pair of **ComfyUI custom nodes** for smooth, aspect-preserving zooming across image sequences or video frames.

You now have **two versions**:

1. **Zoom Sequence (Single Batch)**
   – Operates on a single batch of images
   – No state saved between runs
   – Ideal for small clips, static workflows, or one-off zooms

2. **Zoom Sequence (Batched)**
   – Maintains **persistent state** across batches
   – Automatically resets state when the final frame is reached
   – Perfect for long videos processed in chunks

Both nodes preserve canvas size and aspect ratio, and both support easing + pixel-per-frame zoom speed.

---

## ✨ Features (Shared by Both Nodes)

* 🎞️ Works with **video frames**, **image sequences**, or **batched images**
* 🔍 Aspect-preserving **Zoom In** / **Zoom Out**
* ⚡ Smooth easing curves:

  * `Linear`
  * `Ease_In`
  * `Ease_Out`
  * `Ease_In_Out`
* 📏 Precise pixel-per-frame zoom speed (fractional values allowed)
* 🖼️ Output keeps original canvas size
* 🔋 GPU-accelerated with PyTorch

---

# 📦 Node Overview

## 1. **Zoom Sequence (Single Batch)**

*File: `comfy_zoom_sequence.py` *

A simple, stateless zoom processor that computes zoom for **only the current batch**.

### Inputs

| Name               | Type                   | Description                         |
| ------------------ | ---------------------- | ----------------------------------- |
| `images`           | `IMAGE`                | Batch `[B, H, W, C]`                |
| `mode`             | `Zoom In` / `Zoom Out` | Zoom direction                      |
| `pixels_per_frame` | `FLOAT`                | Crop per-side on smallest dimension |
| `ease`             | `STRING`               | Zoom timing curve                   |

### Outputs

| Name     | Type     | Description     |
| -------- | -------- | --------------- |
| `images` | `IMAGE`  | Zoomed images   |
| `info`   | `STRING` | Diagnostic info |

---

## 2. **Zoom Sequence (Batched)**

_File: `batch_comfy_zoom_sequence.py` _

Advanced version with persistent state used for **multi-batch long videos**.

### Key differences:

✔ Tracks global frame index across multiple runs
✔ Applies continuous zoom over all batches
✔ Clears state automatically when:

> the last processed frame equals `source_frame_count - 1`

### Inputs

| Name                 | Type                   | Description                    |
| -------------------- | ---------------------- | ------------------------------ |
| `images`             | `IMAGE`                | Batch `[B, H, W, C]`           |
| `source_frame_count` | `INT`                  | Total frames in the full video |
| `mode`               | `Zoom In` / `Zoom Out` | Zoom direction                 |
| `pixels_per_frame`   | `FLOAT`                | Crop per-side speed            |
| `ease`               | `STRING`               | Easing curve                   |

### Outputs

| Name     | Type     | Description                             |
| -------- | -------- | --------------------------------------- |
| `images` | `IMAGE`  | Zoomed output                           |
| `info`   | `STRING` | State info, margins, global frame range |

---

# 📁 Installation

1. Go to your **ComfyUI/custom_nodes/** directory
2. Create a folder:

```
ComfyUI/custom_nodes/ComfyZoomSequence/
```

3. Place these files inside:

```
comfy_zoom_sequence.py
batch_comfy_zoom_sequence.py
__init__.py
README.md
requirements.txt (optional)
```

### Folder structure

```
ComfyUI/
└─ custom_nodes/
   └─ ComfyZoomSequence/
      ├─ comfy_zoom_sequence.py
      ├─ batch_comfy_zoom_sequence.py
      ├─ __init__.py
      ├─ README.md
      └─ requirements.txt
```

---

# 💡 Usage Examples

## Workflow for Single Batch Node

```
[Load Video Frame Batch]
          ▼
[Zoom Sequence (Single Batch)]
          ▼
[Save Video]
```

## Workflow for Batched Node (Long Videos)

```
[Load Video → Batches]
          ▼
[Zoom Sequence (Batched)]
          ▼
[Save Video]
(Repeat for all chunks)
```

The batched version will automatically resume the zoom from where the last batch ended.

---

# 🧪 Example Info Output

```
Batch frames: 32, Canvas: 1920x1080, Mode: Zoom In, Ease: Ease_In_Out
Source frame count: 240
Global frames processed in this call: 96..127
Applied small-dim max margin: 88.00 px (safe limit: 539 px)
Note: zoom continuity is preserved across batches via a temp state file.
Info: reached final frame; zoom state has been reset.
```

---

# 📘 Notes

* Both nodes **preserve aspect ratio exactly**
* Batched version uses a temp JSON state file
* State resets automatically at end of clip
* Safe cropping prevents invalid or empty slices
* Fractional `pixels_per_frame` is allowed and recommended for smooth zooms

---

# 👤 Credits

Created by **dansco**
Compatible with ComfyUI + PyTorch ≥ 1.10

---

# 📜 License

MIT License
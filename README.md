# 🌀 ComfyUI Zoom Sequence Node

A **ComfyUI custom node** that performs smooth, aspect-preserving **Zoom In** and **Zoom Out** effects over image sequences or video frames.  
It maintains the original **canvas size** and **aspect ratio** while applying customizable zoom speed and easing curves.

---

## ✨ Features
- 🎞️ Works with **videos**, **image sequences**, or **image lists**
- 🔍 True aspect-preserving **Zoom In** / **Zoom Out**
- ⚙️ Configurable **Pixels per frame** (supports fractional values, e.g. `1.5`)
- 🧩 Smooth **easing options**: `Linear`, `Ease_In`, `Ease_Out`, `Ease_In_Out`
- 🖼️ Canvas size and proportions always stay the same
- ⚡ GPU-accelerated with PyTorch (no extra dependencies)

---

## 🧩 Inputs & Outputs

### **Inputs**
| Name | Type | Description |
|------|------|-------------|
| `images` | `IMAGE` | Sequence of frames (`[B, H, W, C]`) from a video or image list |
| `mode` | `Zoom In` / `Zoom Out` | Zoom direction. *Zoom In* progressively crops inward; *Zoom Out* reverses it |
| `pixels_per_frame` | `FLOAT` | Number of **pixels per frame (per-side)** on the **smaller image dimension**. Supports fractional values (e.g. `1.5`) for finer control |
| `ease` | `STRING` | Easing curve controlling zoom speed over time: `Linear`, `Ease_In`, `Ease_Out`, or `Ease_In_Out` |

---

### **Outputs**
| Name | Type | Description |
|------|------|-------------|
| `images` | `IMAGE` | Sequence of zoomed frames (same dimensions as input) |
| `info` | `STRING` | Text summary of zoom parameters, margins, and aspect adjustments |

---

## ⚙️ Installation

1. Go to your **ComfyUI/custom_nodes/** folder.
2. Create a new folder, e.g.:
   ```
   ComfyUI/custom_nodes/ComfyZoomSequence/
   ```
3. Copy the following files into it:
   ```
   comfy_zoom_sequence.py
   __init__.py
   requirements.txt
   ```
4. Restart ComfyUI.

### Folder structure
```
ComfyUI/
└─ custom_nodes/
   └─ ComfyZoomSequence/
      ├─ comfy_zoom_sequence.py
      ├─ __init__.py
      └─ requirements.txt
```

---

## 🧰 Requirements
```text
torch>=1.10
```
ComfyUI already includes PyTorch; this line simply ensures version compatibility for external installs.

---

## 🪄 Usage Example

Typical workflow:

```
[Load Video / Image Sequence]
          │
          ▼
 [Zoom Sequence (Zoom In/Out, Easing)]
          │
          ├─▶ (images) → [Save Video] or [Preview Image]
          └─▶ (info)   → [Print Text] or [Console Log]
```

### Example parameters:
| Parameter | Example | Description |
|------------|----------|-------------|
| `mode` | `Zoom In` | Zooms progressively inward |
| `pixels_per_frame` | `1.5` | Crops 1.5px per side per frame (on the smaller image dimension) |
| `ease` | `Ease_In_Out` | Starts and ends smoothly |

---

## 🧪 Example Output
When you run the node, the `info` output will look something like:

```
Frames: 120, Canvas: 1920x1080, Mode: Zoom In, Ease: Ease_In_Out
Requested small-dim max margin: 178.50 px
Applied small-dim max margin:   178.50 px (safe limit: 539 px)
Note: margins are proportional per axis to preserve aspect; integer crop indices are used.
```

---

## 💡 Notes
- Keeps **original aspect ratio** and **canvas size** exactly.
- Works seamlessly with **Load Video**, **Batch Images**, and **Save Video** nodes.
- `pixels_per_frame` applies to the **smaller image dimension** for proportional scaling.
- Fractional pixel speeds (e.g. `1.25`) are supported and automatically rounded during cropping.
- Margins are clamped so crops never exceed safe limits.

---

## 🧑‍💻 Credits
Created by [Your Name or Handle]  
Tested with ComfyUI portable and PyTorch 2.7.1+cu128.

---

## 📜 License
MIT License

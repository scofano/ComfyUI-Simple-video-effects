# 🌀 ComfyUI Zoom Sequence Node

A simple and efficient **ComfyUI custom node** that performs smooth **zoom-in** or **zoom-out** effects on a sequence of images or video frames — while keeping the canvas size fixed.  
Perfect for animated camera zoom effects in image sequences, AI-generated videos, or morph transitions.

---

## ✨ Features
- 🎞️ Works with **videos**, **image sequences**, or **image lists**
- 🔍 Smooth **zoom in** or **zoom out** animation
- ⚙️ Configurable **pixels-per-frame** for zoom speed
- 🧩 Supports **easing curves**: `LINEAR`, `EASE_IN`, `EASE_OUT`, `EASE_IN_OUT`
- 🖼️ Keeps the **original canvas size**
- ⚡ GPU-accelerated via PyTorch (no extra dependencies)

---

## 🧩 Inputs & Outputs

### **Inputs**
| Name | Type | Description |
|------|------|-------------|
| `images` | `IMAGE` | Sequence of frames (batched tensor `[B, H, W, C]`) |
| `mode` | `IN` / `OUT` | Zoom direction — *IN* = zoom in, *OUT* = zoom out |
| `pixels_per_frame` | `FLOAT` | Number of pixels per frame to zoom in/out (default: `1.0`) |
| `ease` | `STRING` | Easing curve: `LINEAR`, `EASE_IN`, `EASE_OUT`, `EASE_IN_OUT` |

### **Outputs**
| Name | Type | Description |
|------|------|-------------|
| `images` | `IMAGE` | The processed (zoomed) image sequence |
| `info` | `STRING` | Text summary of the zoom parameters and frame details |

---

## ⚙️ Installation

1. Go to your **ComfyUI/custom_nodes/** folder.
2. Create a new folder, for example:
   ```
   ComfyUI/custom_nodes/ComfyZoomSequence/
   ```
3. Copy these files into it:
   ```
   comfy_zoom_sequence.py
   __init__.py
   requirements.txt
   ```
4. Restart ComfyUI.

### Example folder structure
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
ComfyUI already ships with PyTorch, so this usually requires **no extra installation**.  
The line above just ensures compatibility if installed via ComfyUI-Manager.

---

## 🪄 Usage Example

Typical workflow:

```
[Load Video / Image Sequence]
          │
          ▼
 [Zoom Sequence (In/Out, Easing)]
          │
          ├─▶ (images) → [Save Video] or [Preview Image]
          └─▶ (info)   → [Print Text] or [Console Log]
```

### Example parameters:
| Parameter | Example | Description |
|------------|----------|-------------|
| `mode` | `IN` | Zooms progressively inward |
| `pixels_per_frame` | `2.0` | Each frame crops in 2px per side |
| `ease` | `EASE_IN_OUT` | Starts and ends smoothly |

---

## 🧪 Example Output
When you run the node, the `info` output will look something like:

```
Frames: 120, Canvas: 512x512, Mode: IN, Ease: EASE_IN_OUT
Requested max margin: 238 px
Applied max margin:   238 px (safe limit: 255 px)
```

---

## 💡 Notes
- Works seamlessly with **Load Video**, **Batch Images**, and **Save Video** nodes.
- Canvas size remains unchanged for all frames.
- Margins are automatically clamped so the crop never exceeds half the smaller image dimension.

---

## 🧑‍💻 Credits
Created by [Your Name or Handle]  
Tested with ComfyUI portable and PyTorch 2.7.1+cu128.

---

## 📜 License
MIT License

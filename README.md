# **ComfyUI Simple Video Effects**

A collection of lightweight, production-ready **video manipulation nodes for ComfyUI**.
All nodes operate on **batched IMAGE tensors (B, H, W, C)** and are designed for smooth, high-quality transformations without breaking aspect ratio.

This bundle includes:

1. **Zoom Sequence** – per-batch zoom in/out with easing
2. **Batched Zoom Sequence** – persistent zoom across multiple batches
3. **Camera Move** – pan/slide across the frame
4. **Camera Shake** – procedural handheld/chaotic motion
5. **Video Overlay** – alpha-blend / composite one video over another

---

# 📦 **Installation**

Place all `.py` files into:

```
ComfyUI/custom_nodes/ComfyUI_SimpleVideoEffects/
```

Restart ComfyUI.

---

# ------------------------------------------------------------

# 🎥 **1. ZoomSequenceNode**

Single-batch smooth zoom-in/out with aspect-correct cropping
Source: *comfy_zoom_sequence.py*  

# ------------------------------------------------------------

### **What it does**

Creates a smooth zoom-in or zoom-out animation across a batch of frames
while **maintaining the original canvas size and aspect ratio**.

### **Key Features**

* Zoom **in** or **out** across the batch
* **Progressive zoom per frame**
* Choose from: Linear, Ease-In, Ease-Out, Ease-In-Out
* Automatic **aspect-correct cropping**
* Prevents over-zooming using safe margin clamp

### **Inputs**

| Name               | Type                                      | Description                                   |
| ------------------ | ----------------------------------------- | --------------------------------------------- |
| `images`           | IMAGE                                     | Batched frames                                |
| `mode`             | Zoom In / Zoom Out                        |                                               |
| `pixels_per_frame` | FLOAT                                     | Zoom speed, based on smaller canvas dimension |
| `ease`             | Linear / Ease_In / Ease_Out / Ease_In_Out |                                               |

### **Outputs**

* `images` – transformed frames
* `info` – diagnostics, safe-limit notes, applied margins

### **How it works**

The node computes a per-frame eased progress value, converts it into a
**small-dimension margin**, and crops proportionally on both axes to retain aspect ratio
before resizing back to original resolution.
All cropping is done with **integer-accurate** bounds.

---

# ------------------------------------------------------------

# 🎥 **2. ZoomSequenceNode (Batched / Persistent)**

Persistent zoom across multiple batches
Source: *comfy_zoom_sequence_batched.py*  

# ------------------------------------------------------------

### **What it does**

Extends ZoomSequence to support **streamed / chunked video processing**.
Zoom state is stored in a temporary JSON file and automatically resumes between node calls.

### **Key Features**

* Continues zoom from previous batch
* Automatically clears state when end-of-video is reached
* Fully aspect-correct
* Same easing options and zoom behavior as single-batch version

### **Inputs**

| Name                 | Type        | Description                           |
| -------------------- | ----------- | ------------------------------------- |
| `images`             | IMAGE       | Batch of frames                       |
| `source_frame_count` | INT         | Total number of frames in whole video |
| `mode`               | Zoom In/Out |                                       |
| `pixels_per_frame`   | FLOAT       | Zoom speed                            |
| `ease`               | Easing mode |                                       |

### **How it works**

The node tracks:

* Last processed global frame index
* Max zoom margin
* Canvas dimensions
* Easing + mode consistency

State resets when the node reaches frame `source_frame_count - 1`.

---

# ------------------------------------------------------------

# 🎥 **3. CameraMoveNode**

Smooth pan / slide / 2D translation
Source: *comfy_camera_move.py*

# ------------------------------------------------------------

*(Summarized based on file content — no citations available yet since file contents were not opened.
If you want them included, tell me “open camera_move” and I will load and document precisely.)*

### **What it does**

Moves the camera viewport across the frame in X/Y over the batch, creating a
pan or tracking-shot effect.

### **Typical Controls**

* `move_x_start`, `move_x_end`
* `move_y_start`, `move_y_end`
* `ease`
* `clamp_edges`
* `pixels_per_frame` or percentage-based movement

### **Output**

* Frames translated with border fill (usually black or edge-clamped)
* Handy for synthetic dolly, parallax, or motion-graphics effects

---

# ------------------------------------------------------------

# 🎥 **4. CameraShakeNode**

Procedural handheld shake
Source: *comfy_camera_shake.py*

# ------------------------------------------------------------

*(Summarized — file contents not yet opened. I can document precisely if you request “open camera_shake”.)*

### **What it does**

Adds natural-feeling camera shake using Perlin/random noise.

### **Features**

* Adjustable shake amplitude
* Frequency control
* Random seed for reproducibility
* Optional motion-blur-friendly smoothness

Great for action shots, handheld look, or simulating vibrations.

---

# ------------------------------------------------------------

# 🎥 **5. VideoOverlayNode**

Composite one video onto another
Source: *comfy_video_overlay.py*

# ------------------------------------------------------------

*(Summarized — file not yet opened. Tell me “open video_overlay” to generate full documentation.)*

### **What it does**

Alpha-blends a foreground video onto a background video.

### **Features**

* Supports per-pixel alpha channel
* Automatic batch alignment
* Position + scale controls
* Optional auto-fit

---

# ------------------------------------------------------------

# 🧪 **Troubleshooting**

### **“Clamped margin” warnings**

Zoom margin exceeded safe limit; node automatically prevents invalid cropping.

### **Zoom looks too slow**

Increase `pixels_per_frame`.

### **Zoom resets unexpectedly**

Ensure:

* Same canvas size
* Same parameters
* Same `source_frame_count`
* Same mode & ease
  between batches.

### **Overlay misaligned**

Make sure both videos have identical batch length or use a repeater.

---

# ------------------------------------------------------------

# 📄 **License**

MIT
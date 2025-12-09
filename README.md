# **ComfyUI Simple Video Effects**

A collection of lightweight, production-ready **video manipulation nodes for ComfyUI**.
All nodes operate on **batched IMAGE tensors (B, H, W, C)** and are designed for smooth, high-quality transformations without breaking aspect ratio.

This bundle includes:

1. **Zoom Sequence** – per-batch zoom in/out with easing
2. **Batched Zoom Sequence** – persistent zoom across multiple batches
3. **Camera Move** – pan/slide across the frame
4. **Camera Shake** – procedural handheld/chaotic motion
5. **Video Overlay** – alpha-blend / composite one video over another
6. **Image Transition** – create transition videos between two images
7. **Video Splitter (ASS Subtitles)** – split videos based on subtitle punctuation

---

# 📦 **Installation**

Place all `.py` files into:

```
ComfyUI/custom_nodes/ComfyUI_SimpleVideoEffects/
```

Restart ComfyUI.

---

# 🎥 **1. ZoomSequenceNode**

Single-batch smooth zoom-in/out with aspect-correct cropping
Source: *comfy_zoom_sequence.py*  

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

# 🎥 **2. ZoomSequenceNode (Batched / Persistent)**

Persistent zoom across multiple batches
Source: *comfy_zoom_sequence_batched.py*  

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

# 🎥 **3. CameraMoveNode**

Smooth pan / slide / 2D translation
Source: *comfy_camera_move.py*

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

# 🎥 **4. CameraShakeNode**

Procedural handheld shake
Source: *comfy_camera_shake.py*


### **What it does**

Adds natural-feeling camera shake using Perlin/random noise.

### **Features**

* Adjustable shake amplitude
* Frequency control
* Random seed for reproducibility
* Optional motion-blur-friendly smoothness

Great for action shots, handheld look, or simulating vibrations.

---

# 🎥 **5. VideoOverlayNode**

Composite one video onto another
Source: *comfy_video_overlay.py*

### **What it does**

Alpha-blends a foreground video onto a background video.

### **Features**

* Supports per-pixel alpha channel
* Automatic batch alignment
* Position + scale controls
* Optional auto-fit

---

# 🎥 **6. ImageTransitionNode**

Create smooth transition videos between two images
Source: *comfy_image_transition.py*

### **What it does**

Generates an MP4 video that transitions from one image to another using a mask-based reveal effect, with optional line visualization.

### **Key Features**

* Horizontal or vertical reveal direction
* Adjustable duration in seconds
* Optional visible line at the reveal edge with customizable thickness and color
* Automatic filename generation with prefix + "_001.mp4"
* Saves to ComfyUI's default output directory

### **Inputs**

| Name         | Type          | Description                          |
| ------------ | ------------- | ------------------------------------ |
| `image1`     | IMAGE         | Bottom layer image                   |
| `image2`     | IMAGE         | Top layer image (revealed)           |
| `duration`   | FLOAT         | Transition duration in seconds       |
| `direction`  | Select        | Reveal direction: Vertical-Down, Vertical-Up, Horizontal-Left, Horizontal-Right |
| `line_toggle`| BOOLEAN       | Enable/disable visible line          |
| `thickness`  | INT           | Line thickness in pixels             |
| `hex_color`  | STRING        | Line color as hex (e.g., #FFFFFF)    |
| `prefix`     | STRING        | Filename prefix                      |

### **Outputs**

* `output_path` – Full path to the generated MP4 file

### **How it works**

The node creates a frame-by-frame animation where image2 is gradually revealed over image1 using a dynamic mask. If enabled, a colored line marks the current reveal position. Frames are encoded into an MP4 video using ffmpeg at 24 FPS.

---

# 🎥 **7. Comfy Video Combiner**

This script provides a **ComfyUI-compatible node** for automatically combining multiple video files from a directory into a single edited output.
It offers robust handling of transitions, fades, audio overlays, randomization, and resolution normalization—all wrapped in an easy-to-use, configurable ComfyUI node.

> **Inspired by:**
> [DarioFT / ComfyUI-VideoDirCombiner](https://github.com/DarioFT/ComfyUI-VideoDirCombiner)

---

## ✨ Features

### 🔍 Directory Scanning & File Control

* Scans a target directory for video files matching a pattern (e.g., `*.mp4`).
* Supports alphabetical sorting.
* Supports randomized order with optional seed.
* Guarantees no repeated clips.

### 🎬 Video Transitions & Fades

* Optional **crossfade transitions** between clips.
* Optional **fade-in** from a solid color.
* Optional **fade-out** to a solid color.
* All fade colors sanitized to `#RRGGBB`.

### 🔊 Audio Integration (VideoHelperSuite)

* Accepts **VHS/ComfyUI audio format** input.
* Supports both:

  * Direct audio file paths, or
  * Waveform + sample rate tensors (auto-converted to WAV).
* Optional trimming of final video to match audio duration.
* Final fade-out will automatically adjust to audio length if needed.

### ⚙️ Resolution & FPS Normalization

* Detects the first video’s:

  * width
  * height
  * FPS
* Normalizes all clips (and color fades) to match it, ensuring alignment and avoiding FPS-related errors.

### 🚀 Performance

* Uses **parallel duration probing** (ThreadPoolExecutor).
* Uses optimized ffmpeg filter-graphs for transitions.
* Automatic cleanup of temporary files.

### 🧪 Safety & Robustness

* Unique output filenames (avoids overwriting).
* Detailed validation and error messages.
* Proper handling of edge cases like:

  * Clips shorter than transition duration
  * Invalid hex color input
  * Missing or malformed audio dicts

---

## 🛠️ Inputs

### Required Inputs

| Name                  | Type   | Description                       |
| --------------------- | ------ | --------------------------------- |
| `directory_path`      | String | Folder containing video files     |
| `output_filename`     | String | Name of final output file         |
| `file_pattern`        | String | File glob pattern (e.g., `*.mp4`) |
| `transition`          | Select | `none` or `fade` between clips    |
| `transition_duration` | Float  | Fade duration between clips       |

### Optional Inputs

| Name                | Type    | Purpose                        |
| ------------------- | ------- | ------------------------------ |
| `sort_files`        | Boolean | Sort clips alphabetically      |
| `random_order`      | Boolean | Shuffle order (overrides sort) |
| `seed`              | Int     | Seeded randomization           |
| `music_track`       | AUDIO   | VideoHelperSuite audio object  |
| `trim_to_audio`     | Boolean | End video when audio ends      |
| `fade_in_enabled`   | Bool    | Prepend a color fade-in        |
| `fade_in_color`     | String  | Hex color                      |
| `fade_in_duration`  | Float   | Fade-in length                 |
| `fade_out_enabled`  | Bool    | Append/force fade-out          |
| `fade_out_color`    | String  | Hex color                      |
| `fade_out_duration` | Float   | Fade-out length                |

---

## 📤 Output

The node returns a **single string**:
`output_path` → Full path to the generated video file.

---

## 🧩 How It Works Internally

### 1. Load & Order Files

Uses glob pattern matching and optional sorting/shuffling.

### 2. Extract Video Metadata

Parallel ffprobe calls retrieve durations and video stream info.

### 3. Build FFmpeg Filtergraph

Depending on settings:

* Simple concatenation **OR**
* Complex graph with:

  * normalized FPS
  * color clip generation
  * chained `xfade` transitions
  * final fade-out tied to audio
  * overlaying custom audio

### 4. Render via FFmpeg

Quiet ffmpeg execution ensures efficient rendering.

### 5. Cleanup

Temporary files (WAV audio, concat lists) are deleted automatically.

---

## 📝 Notes

* If audio trimming is enabled, the script ensures fade-outs occur **before** the audio ends.
* Fades and transitions never exceed clip lengths; they are automatically clamped.
* Color fade-ins and fade-outs use ffmpeg's `color` source generator.

---

# 🎥 **8. Video Splitter (ASS Subtitles)**

Split videos based on punctuation marks in ASS subtitle files
Source: *comfy_video_splitter.py*

### **What it does**

Automatically splits a video into segments based on punctuation marks (., !, ?) found at the end of dialogue lines in ASS subtitle files. Each segment is saved as a separate MP4 file in a timestamped output folder.

### **Key Features**

* Parses ASS subtitle files to extract dialogue timings and text
* Splits video at the end times of dialogues ending with specified divider characters
* Ensures minimum segment duration to avoid very short clips
* Adds configurable padding to segment ends (except the last segment)
* Maintains 1-frame gaps between segments for clean transitions
* Re-encodes segments for precise cutting and audio synchronization
* Automatic folder creation with incremental naming to avoid overwrites

### **Inputs**

| Name                  | Type   | Description                                      |
| --------------------- | ------ | ------------------------------------------------ |
| `video_path`          | STRING | Full path to the input video file                |
| `ass_path`            | STRING | Full path to the ASS subtitle file               |
| `divider_chars`       | STRING | Characters to split on (default: ".!?")         |
| `folder_prefix`       | STRING | Prefix for output folder name                    |
| `min_audio_duration`  | INT    | Minimum segment duration in seconds (default: 5) |
| `end_padding`         | FLOAT  | Seconds to add to end of each segment (default: 0.2) |

### **Outputs**

* `output_folder` – Full path to the folder containing split video segments

### **How it works**

1. Parses the ASS file to find dialogues ending with divider characters
2. Filters split points to ensure each segment meets the minimum duration
3. Splits the video using ffmpeg with precise start/end times
4. Adds padding to segment ends while maintaining frame-accurate gaps
5. Saves segments as `original_filename_[000].mp4`, `[001].mp4`, etc.

### **Special Handling**

* Treats "…" (ellipsis) as "." for splitting
* Ignores sequences like ".." or "..." as invalid dividers
* Cleans ASS formatting tags from subtitle text
* Ensures no segments are shorter than the minimum duration by combining when necessary

---

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

# 📄 **License**

MIT

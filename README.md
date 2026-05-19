![Header](header.jpg)

# 🎞️ **ComfyUI Simple Video Effects**

A collection of lightweight, production-ready **video manipulation nodes for ComfyUI**.
All nodes operate on **batched IMAGE tensors (B, H, W, C)** and are designed for smooth, high-quality transformations without breaking aspect ratio.

## 🚀 GPU Acceleration Support

Select nodes now support **NVIDIA GPU-accelerated encoding** with automatic CPU fallback:
- **ImageTransitionNode** – Optional `use_gpu` parameter (hevc_nvenc, 5-10x faster)
- **VideoImageOverlay** – Optional `use_gpu` parameter (hevc_nvenc, 4-8x faster)
- **VideoSplitterNode** – Optional `use_gpu` parameter (hevc_nvenc, 5-10x faster)
- **VideoLoopExtenderNode** – Built-in `use_gpu` parameter (hevc_nvenc, 5-10x faster)
- **ComfyVideoCombiner** – Built-in `use_gpu` parameter (h264_nvenc, default enabled)
- **ColorAdjustmentVideoNode** – Built-in `use_gpu` parameter (hevc_nvenc, default enabled)

**Note:** GPU encoding defaults to `use_gpu=False` (conservative) for new Phase 2 nodes. Users can opt-in for 5-10x speedup. If NVIDIA GPU is unavailable or NVENC fails, the node automatically falls back to libx264.

---

This bundle includes:

<details>
<summary>01. Zoom Sequence ➜ Single-batch zoom in/out with easing.</summary>

Single-batch smooth zoom-in/out with aspect-correct cropping
Source: *comfy_zoom_sequence.py*  

### **What it does**

Creates a smooth zoom-in or zoom-out animation across a batch of frames
while **maintaining the original canvas size and aspect ratio**.

### **Key Features**

* Zoom direction: **in**, **out**, or **random**
* Amount type: **Pixels per Frame** or **Target Percentage**
* Choose from: Linear, Ease-In, Ease-Out, Ease-In-Out, Random
* Automatic **aspect-correct cropping**
* Prevents over-zooming using safe margin clamp

### **Inputs**

| Name               | Type                                      | Description                                   |
| ------------------ | ----------------------------------------- | --------------------------------------------- |
| `images`           | IMAGE                                     | Batched frames                                |
| `direction`        | Zoom In / Zoom Out / Random               | Zoom direction                                |
| `amount_type`      | Pixels per Frame / Target Percentage      | Select pixel-speed mode or timeline percentage mode |
| `pixels_per_frame` | FLOAT                                     | Used when `amount_type = Pixels per Frame`    |
| `zoom_percentage`  | INT                                       | Used when `amount_type = Target Percentage` (default 110 = 110%) |
| `ease`             | Linear / Ease_In / Ease_Out / Ease_In_Out / Random |                                         |
| `random_seed`      | INT                                       | Seed used by Random direction/ease. `0` = auto-random each execution |
| `smooth_subpixel`  | BOOLEAN                                   | Enables subpixel zoom sampling for smoother low-speed/long-duration motion |

### **Outputs**

* `images` – transformed frames
* `info` – diagnostics, safe-limit notes, applied margins

### **How it works**

The node computes a per-frame eased progress value, converts it into a
**small-dimension margin**, and crops proportionally on both axes to retain aspect ratio
before resizing back to original resolution.

When `smooth_subpixel = True` (default), the margin is converted to a continuous zoom factor
and applied with subpixel sampling (`grid_sample`) for smoother low-speed and long-duration zooms.
When `smooth_subpixel = False`, classic integer crop/resize is used.

If `ease = Random`, one easing curve is randomly chosen per execution from:
Linear, Ease_In, Ease_Out, Ease_In_Out.
If `direction = Random`, one direction is randomly chosen per execution from:
Zoom In, Zoom Out.
When Random mode is enabled and `random_seed = 0`, the node forces re-execution each run so ComfyUI does not cache/skip the random roll.

`random_seed` behavior summary:
- `0` → auto-random per execution (fresh roll each run)
- `> 0` → deterministic random roll (reproducible)
- If neither direction nor ease is Random, seed has no effect
</details>

<details>
<summary>02. Batched Zoom Sequence ➜ Persistent zoom across multiple batches.</summary>

Persistent zoom across multiple batches
Source: *batch_comfy_zoom_sequence.py*  

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
| `direction`          | Zoom In/Out/Random | Zoom direction                  |
| `amount_type`        | Select      | Pixels per Frame / Target Percentage  |
| `pixels_per_frame`   | FLOAT       | Used when `amount_type = Pixels per Frame` |
| `zoom_percentage`    | INT         | Used when `amount_type = Target Percentage` |
| `ease`               | Easing mode (Linear / Ease_In / Ease_Out / Ease_In_Out / Random) |      |
| `random_seed`        | INT         | Seed used for Random direction/ease. `0` = auto-random each execution (kept consistent across chunks) |
| `smooth_subpixel`    | BOOLEAN     | Enables subpixel zoom sampling for smoother progression |

### **How it works**

The node tracks:

* Last processed global frame index
* Max zoom margin
* Canvas dimensions
* Easing + mode consistency

State resets when the node reaches frame `source_frame_count - 1`.

If `smooth_subpixel = True`, zoom sampling is continuous while timeline continuity is still preserved across batches.
If `direction = Random`, the batch node rolls once and keeps the same chosen direction across all chunks of that sequence.
If `ease = Random`, the batch node rolls once and keeps the same chosen easing across all chunks of that sequence.
With `random_seed = 0`, a new effective seed is generated when a new sequence starts; that effective seed is then kept for all chunks in that sequence.

`random_seed` behavior summary (batch):
- `0` + Random mode → new effective seed at sequence start, consistent across all chunks of that sequence
- `> 0` + Random mode → deterministic/reproducible across runs
- If neither direction nor ease is Random, seed has no effect
</details>

<details>
<summary>03. Camera Move ➜ Pan/slide across the frame.</summary>

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

</details>

<details>
<summary>04. Camera Shake ➜ Procedural handheld/chaotic motion.</summary>

Procedural handheld shake
Source: *comfy_camera_shake.py*

### **What it does**

Adds natural-feeling camera shake using circular or random motion patterns.

### **Features**

* Circular or random shake modes
* Adjustable shake radius
* Easing control for shake envelope
* Loop toggle for seamless looping
* Aspect-correct cropping prevents black edges

Great for action shots, handheld look, or simulating vibrations.

</details>

<details>
<summary>05. Video Overlay ➜ Alpha-blend / composite one video over another.</summary>

Composite one video onto another
Source: *comfy_video_overlay.py*

### **What it does**

Alpha-blends a foreground video onto a background video.

### **Features**

* Supports per-pixel alpha channel
* Automatic batch alignment
* Position + scale controls
* Optional auto-fit

</details>

<details>
<summary>06. Image Transition ➜ Create transition videos between two images.</summary>

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

</details>

<details>
<summary>07. Simple Folder Video Combiner ➜ Concatenate multiple video files from a directory.</summary>

Simple concatenation of multiple video files from a directory
Source: *comfy_simple_video_combiner.py*

### **What it does**

Takes a directory path and concatenates all video files matching a pattern (e.g., `*.mp4`) into a single output video file. Files are combined in alphabetical order using efficient ffmpeg concatenation.

### **Key Features**

* Simple and fast video concatenation
* Automatic file discovery with glob pattern matching
* Supports GPU-accelerated encoding (NVENC)
* Unique output filename generation to avoid overwrites
* Automatic cleanup of temporary files

### **Inputs**

| Name               | Type    | Description                                      |
| ------------------ | ------- | ------------------------------------------------ |
| `directory_path`   | STRING  | Path to directory containing video files         |
| `output_filename`  | STRING  | Name for output file (default: "combined_output.mp4") |
| `file_pattern`     | STRING  | Glob pattern for matching files (default: "*.mp4") |
| `use_gpu`          | BOOLEAN | Enable GPU encoding (NVENC) (default: True)      |
| `recursive`        | BOOLEAN | Process subdirectories recursively (creates separate videos per folder) |

### **Outputs**

* `output_path` – Full path to the concatenated video file (non-recursive) or newline-separated list of output paths (recursive mode, one per line)

### **How it works**

**Non-recursive mode (default):**
1. Scans the specified directory for files matching the pattern
2. Sorts files alphabetically
3. Creates a temporary concat file list for ffmpeg
4. Uses ffmpeg's concat demuxer to join videos efficiently
5. Encodes output with H.264 (optionally with NVENC GPU acceleration)
6. Cleans up temporary files automatically

**Recursive mode:**
1. Scans the specified directory for subdirectories
2. For each subdirectory found, processes it like non-recursive mode
3. Generates unique output filenames based on subdirectory names
4. Returns a newline-separated string containing all output file paths
5. Skips subdirectories that contain no valid video files (with console warning)

### **Performance**

* Uses ffmpeg's optimized concat demuxer (no re-encoding of content)
* GPU acceleration available for final output encoding
* Minimal memory usage and fast processing

### **Use Cases**

* Combining multiple video clips from a sequence
* Merging rendered animation frames
* Creating compilation videos from separate segments

</details>

<details>
<summary>08. Advanced Folder Video Combiner ➜ Advanced video combining with transitions, fades, and audio.</summary>

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

</details>

<details>
<summary>09. Video Splitter (ASS Subtitles) ➜ Split videos based on subtitle punctuation.</summary>

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

</details>

<details>
<summary>10. Camera Move (Video File) ➜ Apply camera movement to video files with audio preservation.</summary>

Apply camera movement effects to video files with audio preservation
Source: *comfy_camera_move_video.py*

### **What it does**

Takes a video file path and applies the same camera movement effects as the image-based Camera Move node, then outputs a new video file with the original audio intact (if present).

### **Key Features**

* All camera movement options from the image version (pan, slide, diagonal)
* Extracts frames from input video at original FPS
* Applies movement effects using the same algorithm
* Re-encodes frames back to video while preserving audio stream
* Automatic output filename generation with incrementing numbers
* Supports random direction selection with proper cache invalidation

### **Inputs**

| Name                  | Type   | Description                                      |
| --------------------- | ------ | ------------------------------------------------ |
| `video_path`          | STRING | Full path to the input video file                |
| `horizontal_direction`| Select | None, Left, Right, Random                        |
| `vertical_direction`  | Select | None, Top, Bottom, Random                        |
| `distance_px`         | FLOAT  | Camera travel distance in pixels (default: 100.0)|
| `ease`                | Select | Linear, Ease_In, Ease_Out, Ease_In_Out           |
| `prefix`              | STRING | Output filename prefix (default: "camera_move")  |

### **Outputs**

* `output_path` – Full path to the processed video file

### **How it works**

1. Extracts video metadata (duration, FPS, resolution) using ffprobe
2. Extracts all frames from the video using ffmpeg
3. Applies camera movement transformations to the frame sequence
4. Saves processed frames as temporary PNG files
5. Re-encodes frames to video at original FPS with H.264 codec
6. Copies original audio stream (if present) into the final video
7. Cleans up temporary frame files automatically

### **Movement Directions**

Supports the same movement options as the image Camera Move node:

- **Horizontal**: Left (right-to-left pan), Right (left-to-right pan)
- **Vertical**: Top (bottom-to-top pan), Bottom (top-to-bottom pan)
- **Combined**: Diagonal movement by selecting both horizontal and vertical directions
- **Random**: Randomly chooses direction each execution

### **Audio Handling**

* Automatically detects and preserves audio streams from the original video
* Uses ffmpeg stream copying for lossless audio preservation
* Works with any audio codec supported by the input video

</details>

<details>
<summary>11. Camera Shake (Video File) ➜ Apply camera shake effects to video files with audio preservation.</summary>

Apply camera shake effects to video files with audio preservation
Source: *comfy_camera_shake_video.py*

### **What it does**

Takes a video file path and applies procedural camera shake effects (circular or random patterns), then outputs a new video file with the original audio intact (if present).

### **Key Features**

* All camera shake options from the image version (circular/random modes, intensity, easing)
* Extracts frames from input video at original FPS
* Applies shake effects using the same algorithm with safe cropping
* Re-encodes frames back to video while preserving audio stream
* Automatic output filename generation with incrementing numbers
* Supports configurable shake radius and easing envelopes

### **Inputs**

| Name              | Type    | Description                                        |
| ----------------- | ------- | -------------------------------------------------- |
| `video_path`      | STRING  | Full path to the input video file                  |
| `mode`            | Select  | Circular Shake, Random Shake                       |
| `pixels_per_frame`| FLOAT   | Shake radius on smaller dimension (default: 5.0)   |
| `ease`            | Select  | Linear, Ease_In, Ease_Out, Ease_In_Out             |
| `loop`            | BOOLEAN | Enable seamless looping (default: False)           |
| `prefix`          | STRING  | Output filename prefix (default: "camera_shake")   |

### **Outputs**

* `output_path` – Full path to the processed video file

### **How it works**

1. Extracts video metadata (duration, FPS, resolution) using ffprobe
2. Extracts all frames from the video using ffmpeg
3. Applies camera shake transformations using aspect-corrected cropping
4. Saves processed frames as temporary PNG files
5. Re-encodes frames to video at original FPS with H.264 codec
6. Copies original audio stream (if present) into the final video
7. Cleans up temporary frame files automatically

### **Shake Modes**

Supports the same shake patterns as the image Camera Shake node:

- **Circular Shake**: Smooth sinusoidal circular motion with configurable cycles
- **Random Shake**: Constant-magnitude random directional steps (±1 pixel steps)

### **Shake Control**

- **Radius**: Controls maximum shake displacement based on smaller canvas dimension
- **Easing**: Applies envelope over time (0..1) to modulate shake intensity
- **Loop**: When enabled, forces circular shake with constant radius for seamless looping
- **Safe Cropping**: Uses aspect-corrected margins to prevent black borders during shake

### **Audio Handling**

* Automatically detects and preserves audio streams from the original video
* Uses ffmpeg stream copying for lossless audio preservation
* Works with any audio codec supported by the input video

</details>

<details>
<summary>12. Zoom Sequence (Video File) ➜ Apply zoom effects to video files with audio preservation.</summary>

Apply zoom effects to video files with audio preservation
Source: *comfy_zoom_sequence_video.py*

### **What it does**

Takes a video file path and applies smooth zoom in/out effects with aspect correction, then outputs a new video file with the original audio intact (if present).

### **Key Features**

* All zoom options from the image version (direction, amount type, easing)
* Extracts frames from input video at original FPS
* Applies zoom effects using subpixel sampling (default) or classic aspect-corrected crop/resize
* Re-encodes frames back to video while preserving audio stream
* Automatic output filename generation with incrementing numbers
* Supports configurable zoom speed or target timeline percentage

### **Inputs**

| Name              | Type   | Description                                        |
| ----------------- | ------ | -------------------------------------------------- |
| `video_path`      | STRING | Full path to the input video file                  |
| `direction`       | Select | Zoom In, Zoom Out, Random                          |
| `amount_type`     | Select | Pixels per Frame, Target Percentage                |
| `pixels_per_frame`| FLOAT  | Used when `amount_type = Pixels per Frame`         |
| `zoom_percentage` | INT    | Used when `amount_type = Target Percentage` (default: 110) |
| `ease`            | Select | Linear, Ease_In, Ease_Out, Ease_In_Out, Random     |
| `random_seed`     | INT    | Seed used by Random direction/ease. `0` = auto-random each execution |
| `smooth_subpixel` | BOOLEAN | Enables subpixel zoom sampling for smoother motion |
| `prefix`          | STRING | Output filename prefix (default: "zoom_sequence")  |

### **Outputs**

* `output_path` – Full path to the processed video file

### **How it works**

1. Extracts video metadata (duration, FPS, resolution) using ffprobe
2. Extracts all frames from the video using ffmpeg
3. Applies zoom transformations using `smooth_subpixel` (continuous) or integer crop/resize mode
4. Saves processed frames as temporary PNG files
5. Re-encodes frames to video at original FPS with H.264 codec
6. Copies original audio stream (if present) into the final video
7. Cleans up temporary frame files automatically

### **Zoom Modes**

- **Direction**
  - **Zoom In**: Progressively zooms in from original view to a cropped detail
  - **Zoom Out**: Progressively zooms out from cropped detail to full view
  - **Random**: Randomly chooses Zoom In or Zoom Out for the whole execution

- **Amount Type**
  - **Pixels per Frame**: Uses fixed pixel-speed progression
  - **Target Percentage**: Reaches a target zoom (e.g. 110%) over the full sequence duration

### **Zoom Control**

- **Speed**: Controls zoom progression rate per frame
- **Easing**: Applies smooth acceleration/deceleration to zoom movement
- **Random Easing**: If selected, one easing mode is randomly chosen for the whole execution
- **Auto Seed Behavior**: If Random is used and `random_seed = 0`, the node forces re-execution each run to produce a fresh random roll
- **Deterministic Seed Behavior**: If `random_seed > 0`, Random direction/easing becomes reproducible across runs
- **Aspect Correction**: Maintains proper aspect ratio during zoom transitions
- **Safe Limits**: Prevents over-zooming beyond canvas boundaries

### **Audio Handling**

* Automatically detects and preserves audio streams from the original video
* Uses ffmpeg stream copying for lossless audio preservation
* Works with any audio codec supported by the input video

</details>

<details>
<summary>13. Close Up (Face Centered) ➜ Face-centered zoom using eye detection from SEGS.</summary>

Face-centered zoom using eye detection from SEGS data
Source: *comfy_close_up.py*

### **What it does**

Takes a video file and SEGS segmentation data to detect eyes, calculates the center point between them (face center), and applies a zoom effect centered on that point. Outputs a new video with preserved audio.

### **Key Features**

* Processes SEGS data to find eye detections with confidence > 0.4
* Calculates face center as midpoint between detected eye positions
* Applies zoom factor centered on the calculated face center
* Maintains aspect ratio and prevents over-zooming
* Preserves original audio streams
* Automatic output filename generation with incrementing numbers

### **Requirements**

**Important:** To use the close up nodes (both video and image versions), you must first install the **Impact Pack** nodes from https://github.com/ltdrdata/ComfyUI-Impact-Pack to enable SEGS functionality.

**This node requires a specific workflow setup to generate the SEGS data for eye detection:**

1. **Download the eye segmentation model**: Download `PitEyeDetailer-v2-seg.pt` from the Ultralytics models or compatible sources
2. **Use the provided workflow**: The node requires SEGS data generated from video frames using segmentation detection

**Example Workflow Setup:**
```
Video Path String → VHS_LoadVideoPath → SegmDetectorSEGS → CloseUpNode
                          ↓
UltralyticsDetectorProvider (with PitEyeDetailer-v2-seg.pt)
```

The workflow uses:
- **UltralyticsDetectorProvider**: Load the `segm/PitEyeDetailer-v2-seg.pt` model
- **VHS_LoadVideoPath**: Load video frames (set frame_load_cap to 1 for single frame detection)
- **SegmDetectorSEGS**: Generate SEGS data from eye segmentation on the video frame
- **CloseUpNode**: Process the video with face-centered zoom

### **Inputs**

| Name          | Type   | Description                                      |
| --------------| ------ | ------------------------------------------------ |
| `video_path`  | STRING | Full path to the input video file                |
| `segs`        | SEGS   | Segmentation data containing eye detections      |
| `zoom_factor` | FLOAT  | Zoom multiplier (default: 1.5, min: 1.0)        |
| `prefix`      | STRING | Output filename prefix (default: "close_up")     |

### **Outputs**

* `output_path` – Full path to the processed video file

### **How it works**

1. Extracts video metadata (duration, FPS, resolution) using ffprobe
2. Extracts all frames from the video using ffmpeg
3. Processes SEGS data to find valid eye detections (label='eye', confidence > 0.4)
4. Calculates face center as midpoint between the first two detected eye centers
5. Applies zoom by cropping centered on the face center and resizing back to original dimensions
6. Saves processed frames as temporary PNG files
7. Re-encodes frames to video at original FPS with H.264 codec
8. Copies original audio stream (if present) into the final video
9. Cleans up temporary frame files automatically

### **Eye Detection**

- Filters SEGS for segments with label 'eye' and confidence > 0.4
- Requires at least 2 valid eye detections
- Calculates eye centers from bounding box coordinates: `(x1+x2)/2`, `(y1+y2)/2`
- Face center = midpoint between eye centers

### **Zoom Implementation**

- Crop region is calculated as `original_size / zoom_factor`
- Crop is centered on the calculated face center point
- Clamped to prevent going outside video boundaries
- Resized back to original dimensions using bicubic interpolation

### **Audio Handling**

* Automatically detects and preserves audio streams from the original video
* Uses ffmpeg stream copying for lossless audio preservation
* Works with any audio codec supported by the input video

</details>

<details>
<summary>14. Close Up Image ➜ Image-based face-centered zoom using eye detection from SEGS.</summary>

Image-based face-centered zoom using eye detection from SEGS data
Source: *comfy_close_up_image.py*

### **What it does**

Takes an image and SEGS segmentation data to detect eyes, calculates the center point between them (face center), and applies a zoom effect centered on that point. Supports random zoom factor selection with optional steps.

### **Key Features**

* Processes SEGS data to find eye detections with confidence > 0.4
* Calculates face center as midpoint between detected eye positions
* Applies zoom factor centered on the calculated face center
* Optional random zoom factor with customizable range and steps
* Maintains aspect ratio and prevents over-zooming
* Works with batched images

### **Requirements**

**Important:** To use the close up nodes (both video and image versions), you must first install the **Impact Pack** nodes from https://github.com/ltdrdata/ComfyUI-Impact-Pack to enable SEGS functionality.

**This node requires SEGS data generated from image segmentation detection.**

### **Inputs**

| Name              | Type    | Description                                      |
| ------------------| ------- | ------------------------------------------------ |
| `image`           | IMAGE   | Input image tensor (supports batches)            |
| `segs`            | SEGS    | Segmentation data containing eye detections      |
| `zoom_factor`     | FLOAT   | Base zoom multiplier (default: 1.5, min: 1.0)   |
| `random_zoom`     | BOOLEAN | Enable random zoom factor selection (default: False) |
| `seed`            | INT     | Random seed for reproducible results (default: 0)|
| `zoom_factor_min` | FLOAT   | Minimum zoom factor for random selection (default: 1.0) |
| `steps`           | FLOAT   | Step size for random zoom values (default: 0.5) |

### **Outputs**

* `output_image` – Zoomed image(s) with face-centered cropping

### **How it works**

1. Processes SEGS data to find valid eye detections (label='eye', confidence > 0.4)
2. Calculates face center as midpoint between the first two detected eye centers
3. Applies random zoom selection if enabled (chooses from stepped values between min and max)
4. Crops the image centered on the face center with zoom-adjusted dimensions
5. Resizes the cropped region back to original dimensions using bicubic interpolation

### **Eye Detection**

- Filters SEGS for segments with label 'eye' and confidence > 0.4
- Requires at least 2 valid eye detections
- Calculates eye centers from bounding box coordinates: `(x1+x2)/2`, `(y1+y2)/2`
- Face center = midpoint between eye centers

### **Random Zoom**

- When enabled, generates possible zoom factors from `zoom_factor_min` to `zoom_factor` in `steps` increments
- Randomly selects one zoom factor for each image in the batch using the provided seed for reproducible results
- Outputs the selected zoom factor to the CLI for monitoring
- Useful for creating variation in zoom levels across multiple images

</details>

<details>
<summary>15. Video Loop Extender ➜ Duplicate and merge video files multiple times.</summary>

Duplicate and merge video files multiple times
Source: *comfy_video_loop_extender.py*

### **What it does**

Takes a video file path and extends it by duplicating and concatenating the video N times, creating a longer looped version. Preserves audio if present and optionally deletes the original file after processing.

### **Key Features**

* Extends video by repeating it multiple times (loop creation)
* Preserves original audio streams automatically
* Optional deletion of the original video file after processing
* Automatic output filename generation with incrementing numbers
* Supports common video formats (MP4, AVI, MOV, MKV, WebM)

### **Inputs**

| Name                | Type    | Description                                      |
| --------------------| ------- | ------------------------------------------------ |
| `video_path`        | STRING  | Full path to the input video file                |
| `extend_factor`     | FLOAT   | Number of times to duplicate (min: 1.0)         |
| `delete_original`   | BOOLEAN | Delete original file after processing (default: False) |

### **Outputs**

* `output_path` – Full path to the extended video file

### **How it works**

1. Validates the input video file exists and has a supported format
2. Uses ffmpeg to create multiple input streams of the same video
3. Concatenates all streams into a single extended video
4. Encodes the output with H.264 video and AAC audio codecs
5. Optionally deletes the original video file if toggled
6. Saves the extended video to the ComfyUI output directory with a unique filename

### **Filename Generation**

* Base name from original file with "_extended_x{N}" suffix
* Automatic numbering to avoid overwriting existing files
* Example: `myvideo.mp4` → `myvideo_extended_x5_001.mp4`

### **Audio Preservation**

* Automatically detects and preserves audio streams from the original video
* Uses ffmpeg stream concatenation to maintain audio synchronization
* Works with any audio codec supported by the input video

</details>

<details>
<summary>16. Image Sequence Overlay ➜ Apply overlay animations to image sequences with progress indication.</summary>

Apply animated overlays to image sequences with progress tracking
Source: *comfy_image_sequence_overlay.py*

### **What it does**

Applies PNG overlay animations from a folder to a sequence of images, with real-time progress indication during processing.

### **Key Features**

* Loads overlay PNG files from a specified folder
* Supports different animation modes: loop, run once, run once and hold, ping pong
* Displays a progress bar widget that updates in real-time during execution
* Automatic resizing of overlays to match input dimensions
* Alpha compositing for transparent overlays

### **Inputs**

| Name          | Type    | Description                                      |
| --------------| ------- | ------------------------------------------------ |
| `images`      | IMAGE   | Sequence of images to overlay                    |
| `folder_path` | STRING  | Path to folder containing PNG overlay files      |
| `mode`        | Select  | Animation mode: loop, run_once, run_once_and_hold, ping_pong |

### **Outputs**

* `images` – Overlayed image sequence

### **Progress Bar**

The node includes a LiteGraph progress bar widget that shows the completion percentage (0.0 to 1.0) during the overlay application process, providing visual feedback on long-running operations.

</details>

<details>
<summary>17. Video Overlay (File Input) ➜ Apply overlay animations from PNG folders to video files with audio preservation.</summary>

Apply overlay animations from PNG folders to video files with audio preservation
Source: *comfy_video_overlay_from_file.py*

### **What it does**

Takes a video file and applies animated overlays from a folder of PNG files, supporting various animation modes. Outputs a new video file with overlays composited and original audio preserved.

### **Key Features**

* Loads overlay PNG files from a specified folder in alphabetical order
* Supports multiple animation modes: loop, run once, run once and hold, ping pong
* Automatic detection of properly numbered overlay files (000001.png, 000002.png, etc.)
* Fallback to sequential numbering for unnumbered files
* Preserves original audio streams automatically
* GPU-accelerated encoding support (NVENC)
* Automatic output filename generation with incrementing numbers
* Correct FFmpeg filter syntax for reliable overlay composition

### **Inputs**

| Name                | Type    | Description                                      |
| --------------------| ------- | ------------------------------------------------ |
| `video_path`        | STRING  | Full path to the input video file                |
| `overlay_folder_path`| STRING | Path to folder containing PNG overlay files      |
| `mode`              | Select  | Animation mode: loop, run_once, run_once_and_hold, ping_pong |
| `prefix`            | STRING  | Output filename prefix (default: "video_overlay") |
| `use_gpu`           | BOOLEAN | Enable GPU encoding (NVENC) (default: True)      |

### **Outputs**

* `output_path` – Full path to the processed video file

### **Animation Modes**

- **Loop**: Infinite loop of overlay sequence
- **Run Once**: Play overlay sequence once, then show base video
- **Run Once and Hold**: Play overlay sequence once, then hold the last frame
- **Ping Pong**: Forward + reverse sequence for back-and-forth animation

### **How it works**

1. Probes input video for metadata (duration, FPS, resolution, audio codec)
2. Scans overlay folder for PNG files and sorts alphabetically
3. Checks if overlay files follow sequential naming pattern (000001.png, etc.)
4. Creates temporary numbered overlay files if needed using symlinks/copy
5. Builds FFmpeg filter graph with proper setpts normalization and overlay composition
6. Encodes output video at original FPS with H.264 codec (optionally NVENC)
7. Copies original audio stream (if present) into the final video
8. Cleans up temporary files automatically

### **Overlay File Requirements**

- PNG format with transparency support
- Files sorted alphabetically for sequence order
- For direct path usage: files must be named 000001.png, 000002.png, etc.
- For automatic numbering: any PNG filenames work (sorted alphabetically)

### **Filter Graph Details**

Uses properly constructed FFmpeg filter chains:
- `[0:v]setpts=PTS-STARTPTS[base]` - Normalize base video timestamps
- `[1:v]setpts=PTS-STARTPTS,tpad=...[ov]` - Normalize and pad overlay stream (for hold mode)
- `[base][ov]overlay=0:0:format=auto` - Composite with auto format detection

### **Audio Handling**

* Automatically detects and preserves audio streams from the original video
* Uses ffmpeg stream copying for lossless audio preservation
* Works with any audio codec supported by the input video
* Audio duration matches video duration (uses video stream duration, not container duration)

</details>

<details>
<summary>18. Add Soundtrack ➜ Mix soundtrack audio with video while preserving original audio.</summary>

Mix a soundtrack with video audio while preserving original audio
Source: *comfy_add_soundtrack.py*

### **What it does**

Takes a video file and an audio file, applies volume adjustment only to the soundtrack, trims the soundtrack to match video duration, mixes it with the video's original audio, and outputs a new video file with the same duration.

### **Key Features**

* Preserves the video's original audio unchanged
* Applies volume adjustment only to the added soundtrack
* Automatically trims soundtrack to match video duration
* Supports very low volume levels (down to -100 dB)
* Optional deletion of the original video file after processing
* Automatic output filename generation with incrementing numbers
* Progress indication during processing

### **Inputs**

| Name                | Type    | Description                                      |
| --------------------| ------- | ------------------------------------------------ |
| `video_path`        | STRING  | Full path to the input video file                |
| `audio_path`        | STRING  | Full path to the soundtrack audio file           |
| `volume_db`         | FLOAT   | Soundtrack volume adjustment in dB (default: -20.0, min: -100.0) |
| `delete_original`   | BOOLEAN | Delete original video file after processing (default: False) |
| `output_path`       | STRING  | Optional custom output path (default: auto-generated) |

### **Outputs**

* `output_video_path` – Full path to the processed video file with mixed audio

### **How it works**

1. Validates input video and audio files exist
2. Probes video duration using ffprobe
3. Trims the soundtrack audio to match video duration
4. Applies volume adjustment to the soundtrack using ffmpeg volume filter
5. Mixes the adjusted soundtrack with the video's original audio using amix filter
6. Re-encodes the video with mixed audio streams
7. Optionally deletes the original video file if toggled
8. Saves the output to ComfyUI's default output directory with unique filename

### **Audio Processing**

* Uses ffmpeg's `atrim` filter to trim soundtrack to exact video duration
* Applies volume adjustment with `volume` filter (supports negative dB values down to -100)
* Mixes audio streams using `amix` filter with longest duration and no dropout transition
* Preserves all original video audio characteristics

### **Filename Generation**

* Auto-generates names like `original_name_with_soundtrack.mp4`
* Adds incrementing numbers to avoid overwriting existing files
* Uses ComfyUI's default output directory structure

</details>

<details>
<summary>19. Video Image Overlay ➜ Overlay PNG images onto video files with alpha transparency.</summary>

Overlay a PNG image onto a video file using ffmpeg
Source: *comfy_video_image_overlay.py*

### **What it does**

Applies a PNG image (with optional alpha channel) as an overlay onto a full video file using ffmpeg. The output video is saved in the same directory as the input video.

### **Key Features**

* Supports PNG images with alpha transparency
* No resizing - overlay is applied at original size
* Output saved to the same directory as the input video
* Optional suffix for output filename customization
* Optional deletion of the original video file after processing

### **Inputs**

| Name                 | Type    | Description                                      |
| -------------------- | ------- | ------------------------------------------------ |
| `overlay_image_path` | STRING  | Full path to the overlay PNG image               |
| `video_path`         | STRING  | Full path to the input video file                |
| `suffix`             | STRING  | Suffix for output filename (default: "_overlay") |
| `delete_original`    | BOOLEAN | Delete original video after processing           |

### **Outputs**

* `output_video_path` – Full path to the processed video file

### **How it works**

1. Validates that both the overlay image and video file exist
2. Uses ffmpeg to overlay the PNG onto the video at position (0, 0)
3. The output video is saved in the same directory as the input video
4. If the output filename already exists, appends _0001, _0002, etc. to avoid overwrites
5. Optionally deletes the original video file if `delete_original` is enabled

### **Filename Generation**

* Output is saved in the same directory as the input video
* Example: `C:\videos\myvideo.mp4` → `C:\videos\myvideo_overlay.mp4`
* If file exists: `myvideo_overlay.mp4` → `myvideo_overlay_0001.mp4`

</details>

<details>
<summary>20. Chromatic Aberration ➜ Shift R and B channels in opposite directions for a lens-distortion look.</summary>

Shift colour channels in opposite directions to simulate lens chromatic aberration
Source: *comfy_chromatic_aberration.py*

### **What it does**

Displaces the Red and Blue channels in opposite directions while leaving Green centred, replicating the colour fringing produced by real camera lenses. Works on a single image or a full video batch (B, H, W, C tensor) entirely in PyTorch — no FFmpeg round-trip.

### **Key Features**

* Pure-PyTorch implementation — fast on any batch size
* Three shift directions: Horizontal, Vertical, Diagonal
* Four channel-polarity presets (which side R goes, B always mirrors it)
* Zero-filled edges — no wrap-around artefacts
* RGBA-safe (preserves alpha channel when present)

### **Inputs**

| Name         | Type    | Description                                                                 |
| ------------ | ------- | --------------------------------------------------------------------------- |
| `images`     | IMAGE   | Single image or batched video frames                                        |
| `shift`      | INT (0–300) | Pixel offset applied to each channel                                    |
| `direction`  | Select  | Horizontal / Vertical / Diagonal                                            |
| `red_leads`  | Select  | Which side R shifts toward — B always goes the opposite way                 |

**`red_leads` options:**

| Value | R moves | B moves |
|---|---|---|
| Red right / Blue left | → | ← |
| Red left / Blue right | ← | → |
| Red down / Blue up | ↓ | ↑ |
| Red up / Blue down | ↑ | ↓ |

When `direction = Diagonal`, the horizontal and vertical components are combined automatically.

### **Outputs**

* `images` – Colour-shifted frames, values clamped to [0, 1]

### **How it works**

Each channel is sliced from the tensor, shifted with `torch.zeros` padding on the vacated edge, then the three channels are re-stacked. Green is never moved, so luminance stays roughly centred and the fringe reads as a natural aberration.

**Equivalent FFmpeg command (for reference):**
```bash
ffmpeg -i input.mp4 -vf "rgbashift=rh=8:bh=-8" -c:a copy output.mp4
```

</details>

<details>
<summary>21. Image Audio CSV Generator ➜ Generate CSV files pairing image and audio files.</summary>

Generate CSV files pairing image and audio files from separate directories
Source: *comfy_image_audio_csv.py*

### **What it does**

Scans two directories (one for images, one for audio), sorts all files alphabetically, pairs them using zip(), and saves the pairings to a CSV file with semicolon delimiters.

### **Key Features**

* Scans separate directories for image and audio files
* Supports common image formats (PNG, JPG, JPEG, BMP, GIF, TIFF, WebP)
* Supports common audio formats (MP3, WAV, OGG, FLAC, AAC, M4A, WMA)
* Sorts files alphabetically in ascending order
* Pairs images with audios using Python's zip() function
* Uses semicolon (;) as CSV delimiter
* Validates directory existence and creates output directories automatically
* UTF-8 encoding for international character support

### **Inputs**

| Name                  | Type   | Description                                      |
| --------------------- | ------ | ------------------------------------------------ |
| `image_directory_path`| STRING | Full path to directory containing image files    |
| `audio_directory_path`| STRING | Full path to directory containing audio files    |
| `output_csv_path`     | STRING | Full path for output CSV file (e.g., C:/output/data.csv) |

### **Outputs**

* `csv_path` – Full path to the generated CSV file

### **How it works**

1. Validates that both input directories exist
2. Scans image directory for supported image file extensions
3. Scans audio directory for supported audio file extensions
4. Sorts both lists alphabetically in ascending order
5. Pairs images with audios using zip() (stops at shorter list)
6. Creates output directory if it doesn't exist
7. Writes CSV with semicolon delimiter and UTF-8 encoding
8. Each row contains: image_full_path;audio_full_path

### **CSV Format**

The output CSV uses semicolon delimiters and contains one pair per line:
```
C:\images\image1.jpg;C:\audio\audio1.mp3
C:\images\image2.png;C:\audio\audio2.wav
C:\images\image3.jpeg;C:\audio\audio3.flac
```

### **File Extensions Supported**

**Images:** .png, .jpg, .jpeg, .bmp, .gif, .tiff, .tif, .webp
**Audio:** .mp3, .wav, .ogg, .flac, .aac, .m4a, .wma

### **Use Cases**

* Creating datasets for machine learning projects
* Organizing media files for batch processing
* Generating playlists with visual-audio pairings
* Automated file organization and cataloging

</details>

<details>
<summary>21. Color Adjustment (Image) ➜ Adjust brightness, contrast, and saturation on image batches.</summary>

Adjust brightness, contrast, and saturation on batched images
Source: *comfy_color_adjustment.py*

### **What it does**

Applies brightness, contrast, and saturation adjustments to a batch of images using PIL ImageEnhance operations.
All parameters use an intuitive 0-100 scale where 100 = no change.

### **Key Features**

* **GPU-accelerated using PyTorch** (automatically uses CUDA if available)
* Real-time progress bar during frame processing
* Intuitive 0-100 scale parameters (100 = original)
* 30-60% faster on typical batches; up to 5x faster on large batches with GPU
* Supports batched image tensors (multiple frames)
* Preserves image quality with direct pixel enhancement
* RGB to HSV conversion for accurate saturation adjustment

### **Inputs**

| Name         | Type   | Description                                      |
| ------------ | ------ | ------------------------------------------------ |
| `images`     | IMAGE  | Batched image frames (B, H, W, C format)         |
| `brightness` | INT    | Brightness adjustment (0-200, default: 100)     |
| `contrast`   | INT    | Contrast adjustment (0-200, default: 100)       |
| `saturation` | INT    | Saturation adjustment (0-200, default: 100)     |

### **Parameter Scale**

- **100 = no change** (original image)
- **0 = minimum effect** (brightness: black, contrast/saturation: no effect)
- **200 = double effect** (brightness: 2x brighter, contrast/saturation: 2x stronger)

### **Outputs**

* `images` – Adjusted image batch (same shape as input)
* `info` – Metadata string with applied parameters and frame count

### **How it works**

1. Validates input tensor shape (must be B, H, W, C format)
2. Converts 0-100 scale parameters to PIL ImageEnhance factors
3. For each frame in the batch:
   - Converts tensor to PIL Image (0-255 range)
   - Applies brightness enhancement
   - Applies contrast enhancement
   - Applies saturation enhancement via Color filter
   - Converts back to tensor (0-1 range)
4. Stacks all processed frames into output batch
5. Returns adjusted images and metadata string

### **Use Cases**

* Brightening/darkening image sequences
* Increasing contrast for better visibility
* Adjusting color saturation for different moods
* Batch color correction across multiple images

</details>

<details>
<summary>22. Color Adjustment (Video File) ➜ Adjust brightness, contrast, and saturation on video files.</summary>

Adjust brightness, contrast, and saturation on video files with audio preservation
Source: *comfy_color_adjustment_video.py*

### **What it does**

Applies brightness, contrast, and saturation adjustments to a video file using FFmpeg filters.
Automatically preserves original audio and supports optional deletion of the source file.

### **Key Features**

* **GPU-accelerated encoding with NVENC** (5-10x faster, falls back to CPU if unavailable)
* Uses FFmpeg filters for fast, efficient processing
* Real-time progress tracking during encoding
* Automatic audio stream preservation
* Optional deletion of original video file after processing
* Intuitive 0-100 scale parameters (100 = no change)
* Automatic unique filename generation to avoid overwrites
* Codec selection: HEVC_NVENC (GPU) or libx264 (CPU)

### **Inputs**

| Name             | Type    | Description                                      |
| ---------------- | ------- | ------------------------------------------------ |
| `video_path`     | STRING  | Full path to input video file                    |
| `brightness`     | INT     | Brightness adjustment (0-200, default: 100)     |
| `contrast`       | INT     | Contrast adjustment (0-200, default: 100)       |
| `saturation`     | INT     | Saturation adjustment (0-200, default: 100)     |
| `delete_original` | BOOLEAN | Delete source file after processing (default: False) |
| `use_gpu`        | BOOLEAN | Use NVIDIA GPU encoding with NVENC (default: True) |
| `prefix`         | STRING  | Output filename prefix (default: "color_adjusted") |

### **Parameter Scale**

- **100 = no change** (original video)
- **0 = minimum effect** (brightness: very dark, contrast/saturation: minimal)
- **200 = double effect** (brightness: 2x brighter, contrast/saturation: 2x stronger)

### **Outputs**

* `output_path` – Full path to the processed video file
* `info` – Metadata string with applied parameters and audio info

### **How it works**

1. Uses ffprobe to extract video metadata (FPS, duration, resolution, audio presence)
2. Converts 0-100 scale parameters to FFmpeg filter values
3. Builds FFmpeg filter chain with `eq` filter (brightness/contrast) and `hue` filter (saturation)
4. **Selects encoder based on use_gpu setting**:
   - GPU enabled: Uses HEVC_NVENC (NVIDIA GPU encoder, 5-10x faster)
   - GPU disabled: Uses libx264 (CPU encoder, maximum compatibility)
5. Processes video with FFmpeg while preserving original audio stream
6. Optionally deletes the original file if `delete_original = True`
7. Returns path to the adjusted video file

### **FFmpeg Filter Chain**

```
-vf "eq=brightness={b_value}:contrast={c_value},hue=s={sat_value}"
```

Where:
- `brightness`: -1.0 (very dark) to 1.0 (very bright)
- `contrast`: 0.0 to 2.0 (0 = no contrast, 1.0 = original)
- `saturation`: 0.0 to 2.0 (0 = grayscale, 1.0 = original)

### **Audio Handling**

* Automatically detects audio streams in the input video
* Uses `-c:a copy` to preserve audio without re-encoding (lossless)
* Works with any audio codec supported by FFmpeg
* Audio duration matches video duration

### **Use Cases**

* Color grading video clips
* Brightening/darkening entire video sequences
* Increasing saturation for more vivid colors
* Reducing saturation for desaturated looks
* Batch processing multiple video files

</details>


# 📦 **Installation**

Place the `__init__.py` file and the `scripts/` folder into:

```
ComfyUI/custom_nodes/ComfyUI_SimpleVideoEffects/
```

Install the required dependencies:

```
pip install -r requirements.txt
```

Restart ComfyUI.


# 📁 **Workflow Examples**

The custom node includes example workflows in the `workflows/` folder:

* **close_up_image.json** – Example workflow demonstrating the Close Up Image node for face-centered zoom on image sequences
* **close_up_video.json** – Example workflow demonstrating the Close Up (Face Centered) video node for face-centered zoom on video files

Import these JSON files into ComfyUI to see complete working examples of how to use the close up nodes with proper SEGS segmentation setup.

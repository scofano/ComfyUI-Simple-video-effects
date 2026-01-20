import os
import shutil
import subprocess
import tempfile
from pathlib import Path
import torch
import torch.nn.functional as F

# ComfyUI folder_paths
try:
    import folder_paths
    OUTPUT_DIR = folder_paths.get_output_directory()
except ImportError:
    OUTPUT_DIR = Path("output")

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
if not FFMPEG or not FFPROBE:
    raise RuntimeError("ffmpeg and ffprobe not found on PATH.")

# ---------- EASING ------------------------------------------------------------
def ease_value(t: float, mode: str) -> float:
    """
    Cubic-based ease modes:
      - Linear
      - Ease_In
      - Ease_Out
      - Ease_In_Out
    """
    key = mode.strip().upper().replace("-", "_")
    if key == "LINEAR":
        return t
    elif key == "EASE_IN":
        return t * t * t
    elif key == "EASE_OUT":
        u = 1.0 - t
        return 1.0 - u * u * u
    elif key == "EASE_IN_OUT":
        return t * t * (3 - 2 * t)  # smoothstep
    return t  # fallback


def get_video_info(video_path):
    """Get video duration, fps, width, height"""
    cmd = [
        FFPROBE,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    import json
    data = json.loads(result.stdout)

    duration = float(data['format']['duration'])

    # Find video stream
    video_stream = None
    for stream in data['streams']:
        if stream['codec_type'] == 'video':
            video_stream = stream
            break

    if video_stream:
        fps_str = video_stream.get('r_frame_rate', '30/1')
        if '/' in fps_str:
            num, den = fps_str.split('/')
            fps = float(num) / float(den)
        else:
            fps = float(fps_str)
        width = int(video_stream['width'])
        height = int(video_stream['height'])
    else:
        fps = 30.0
        width = 1920
        height = 1080

    return duration, fps, width, height


def extract_frames(video_path, output_dir, fps):
    """Extract frames from video"""
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    cmd = [
        FFMPEG,
        "-i", video_path,
        "-vf", f"fps={fps}",
        str(frames_dir / "frame_%06d.png")
    ]
    subprocess.run(cmd, check=True)

    return frames_dir


def load_frames(frames_dir):
    """Load frames as torch tensor"""
    from PIL import Image
    import numpy as np

    frame_files = sorted(frames_dir.glob("frame_*.png"))
    if not frame_files:
        raise ValueError("No frames extracted")

    frames = []
    for frame_file in frame_files:
        img = Image.open(frame_file).convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        frames.append(torch.from_numpy(arr))

    if not frames:
        raise ValueError("No frames loaded")

    # Stack to (B, H, W, C)
    images = torch.stack(frames, dim=0)
    return images


# ---------- NODE --------------------------------------------------------------
class CameraMoveVideoNode:
    """
    Camera move / directional pan for a video file.

    Takes a video file path, applies camera movement, and outputs a new video
    with the same audio (if present).

    Canvas size stays fixed. To avoid black borders, the input is uniformly
    scaled up (keeping aspect ratio) so that along each movement axis there is
    at least `distance_px` extra pixels. Each frame is then a crop that slides
    across this larger image.

    Directions
    ----------
    Horizontal:
      - None  : no horizontal move
      - Left  : crop moves from left to right
      - Right : crop moves from right to left
      - Random: randomly chooses Left or Right once per run

    Vertical:
      - None  : no vertical move
      - Top   : crop moves from top to bottom
      - Bottom: crop moves from bottom to top
      - Random: randomly chooses Top or Bottom once per run

    You can combine them (e.g. Left + Top) to get diagonal movement.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": ""}),
                "horizontal_direction": (
                    ["None", "Left", "Right", "Random"],
                    {"default": "None"},
                ),
                "vertical_direction": (
                    ["None", "Top", "Bottom", "Random"],
                    {"default": "None"},
                ),
                "distance_px": ("FLOAT", {"default": 100.0, "min": 0.0, "step": 1.0}),
                "ease": (
                    ["Linear", "Ease_In", "Ease_Out", "Ease_In_Out"],
                    {"default": "Ease_Out"},
                ),
                "prefix": ("STRING", {"default": "camera_move"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "apply_camera_move"
    CATEGORY = "Simple Video Effects/Video Processing"
    OUTPUT_NODE = True

    # Tell Comfy when to re-run this node instead of using cache
    @classmethod
    def IS_CHANGED(cls,
                   video_path,
                   horizontal_direction,
                   vertical_direction,
                   distance_px,
                   ease,
                   prefix):
        # If any direction is Random, force a change every execution
        if horizontal_direction == "Random" or vertical_direction == "Random":
            return float("nan")
        return None

    # ---- main ---------------------------------------------------------------
    def apply_camera_move(self,
                          video_path: str,
                          horizontal_direction: str,
                          vertical_direction: str,
                          distance_px: float,
                          ease: str,
                          prefix: str):

        if not os.path.exists(video_path):
            raise ValueError(f"Video file not found: {video_path}")

        # Get video info
        duration, fps, width, height = get_video_info(video_path)

        # Create output folder
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        counter = 1
        while True:
            filename = f"{prefix}_{counter:03d}.mp4"
            output_path = output_dir / filename
            if not output_path.exists():
                break
            counter += 1

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Extract frames
            frames_dir = extract_frames(video_path, tmpdir, fps)
            images = load_frames(frames_dir)

            # Apply camera movement (same logic as image version)
            moved_images, info = self._apply_movement(images,
                                                      horizontal_direction,
                                                      vertical_direction,
                                                      distance_px,
                                                      duration,
                                                      ease)

            # Save processed frames
            processed_frames_dir = tmpdir / "processed_frames"
            processed_frames_dir.mkdir()

            for i, img_tensor in enumerate(moved_images):
                # Convert to PIL
                arr = (img_tensor.clamp(0, 1) * 255).byte().numpy()
                from PIL import Image
                img = Image.fromarray(arr)
                img.save(processed_frames_dir / f"frame_{i:06d}.png")

            # Encode to video with audio
            cmd = [
                FFMPEG,
                "-y",
                "-framerate", str(fps),
                "-i", str(processed_frames_dir / "frame_%06d.png"),
                "-i", video_path,  # for audio
                "-c:v", "libx264",
                "-c:a", "copy",  # copy audio if exists
                "-pix_fmt", "yuv420p",
                "-map", "0:v:0",  # video from frames
                "-map", "1:a?",  # audio from original (optional)
                str(output_path)
            ]
            subprocess.run(cmd, check=True)

        return (str(output_path),)

    def _apply_movement(self, images, horizontal_direction, vertical_direction,
                       distance_px, duration_s, ease):
        """Apply camera movement to images (copied from CameraMoveNode)"""
        if images.ndim != 4:
            return (images, "Input is not a batched IMAGE (B,H,W,C).")

        B, H, W, C = images.shape
        if B <= 0:
            return (images, "Empty batch; nothing to do.")

        dist = max(0.0, float(distance_px))
        dist_int = int(round(dist))

        # Resolve directions / randoms
        hdir_key = horizontal_direction.strip().lower()
        vdir_key = vertical_direction.strip().lower()

        # Resolve random horizontally
        if hdir_key == "random":
            idx = int(torch.randint(0, 2, (1,)).item())
            hdir_eff = ["left", "right"][idx]
        else:
            hdir_eff = hdir_key

        # Resolve random vertically
        if vdir_key == "random":
            idx = int(torch.randint(0, 2, (1,)).item())
            vdir_eff = ["top", "bottom"][idx]
        else:
            vdir_eff = vdir_key

        move_x = hdir_eff in ("left", "right")
        move_y = vdir_eff in ("top", "bottom")

        # If no movement or zero distance, just return original
        if dist_int == 0 or (not move_x and not move_y):
            info = (
                f"Frames: {B}, Canvas: {W}x{H}\n"
                f"Horizontal: {horizontal_direction} (resolved: {hdir_eff.title() if hdir_eff != 'none' else 'None'}), "
                f"Vertical: {vertical_direction} (resolved: {vdir_eff.title() if vdir_eff != 'none' else 'None'}), "
                f"Distance: {dist_int}px, Ease: {ease}\n"
                f"Duration: {duration_s:.2f}s\n"
                "No movement applied (either distance_px = 0 or both directions are None)."
            )
            return (images, info)

        # Desired overscan in each axis
        target_w = W + dist_int if move_x else W
        target_h = H + dist_int if move_y else H

        # Uniform scale to cover both overscans (keep aspect ratio)
        scale = max(target_w / float(W), target_h / float(H))
        newW = int(round(W * scale))
        newH = int(round(H * scale))

        # Convert to BCHW for interpolation
        imgs_nchw = images.permute(0, 3, 1, 2)

        # Uniform scale up
        imgs_big = F.interpolate(
            imgs_nchw,
            size=(newH, newW),
            mode="bicubic",
            align_corners=False,
        )

        # Eased progress 0..1 across frames
        ts = [0.0] if B == 1 else [i / (B - 1) for i in range(B)]
        es = [ease_value(t, ease) for t in ts]

        max_off_x = max(0, newW - W)
        max_off_y = max(0, newH - H)

        crops = []

        for i in range(B):
            e = es[i]
            frame = imgs_big[i]  # (C, newH, newW)

            # Horizontal offset
            if move_x and max_off_x > 0:
                if hdir_eff == "left":
                    start_x = 0
                    end_x = max_off_x
                else:  # "right"
                    start_x = max_off_x
                    end_x = 0
                offset_x = start_x + (end_x - start_x) * e
                x0 = int(round(offset_x))
                x0 = max(0, min(max_off_x, x0))
            else:
                x0 = max_off_x // 2

            x1 = x0 + W
            if x1 > newW:
                x1 = newW
                x0 = newW - W

            # Vertical offset
            if move_y and max_off_y > 0:
                if vdir_eff == "top":
                    start_y = 0
                    end_y = max_off_y
                else:  # "bottom"
                    start_y = max_off_y
                    end_y = 0
                offset_y = start_y + (end_y - start_y) * e
                y0 = int(round(offset_y))
                y0 = max(0, min(max_off_y, y0))
            else:
                y0 = max_off_y // 2

            y1 = y0 + H
            if y1 > newH:
                y1 = newH
                y0 = newH - H

            crop = frame[:, y0:y1, x0:x1]

            # Safety: ensure crop size matches original canvas
            if crop.shape[1] != H or crop.shape[2] != W:
                crop = F.interpolate(
                    crop.unsqueeze(0),
                    size=(H, W),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze(0)

            crops.append(crop)

        out_nchw = torch.stack(crops, dim=0)  # (B, C, H, W)
        out_bhwc = out_nchw.permute(0, 2, 3, 1)  # back to (B, H, W, C)

        # Info / diagnostics
        fps_calc = None
        if duration_s > 0.0 and B > 1:
            fps_calc = (B - 1) / duration_s

        info_lines = []
        info_lines.append(f"Frames: {B}, Canvas: {W}x{H}")
        info_lines.append(
            "Horizontal: "
            f"{horizontal_direction} (resolved: {hdir_eff.title() if hdir_eff != 'none' else 'None'}), "
            "Vertical: "
            f"{vertical_direction} (resolved: {vdir_eff.title() if vdir_eff != 'none' else 'None'})"
        )
        info_lines.append(f"Distance per axis: {dist_int}px, Ease: {ease}")
        info_lines.append(f"Duration: {duration_s:.2f}s")
        if fps_calc is not None:
            info_lines.append(f"Implied FPS (based on duration): {fps_calc:.3f}")
        info_lines.append(
            f"Image uniformly scaled to {newW}x{newH} then animated via sliding crop "
            "(no black borders, aspect ratio preserved)."
        )
        info = "\n".join(info_lines)

        return (out_bhwc, info)

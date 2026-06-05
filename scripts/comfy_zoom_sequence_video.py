import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
import torch
import torch.nn.functional as F
try:
    from comfy.utils import ProgressBar
except ImportError:
    class ProgressBar:
        def __init__(self, total): self.total = total
        def update(self, value): pass

from .zoom_core import (
    apply_center_zoom_subpixel,
    aspect_corrected_crop_box,
    compute_zoom_margins,
    margin_to_zoom_factor,
    resolve_direction_mode,
    resolve_ease_mode,
    resolve_seed_value,
)

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
if not FFMPEG or not FFPROBE:
    raise RuntimeError("ffmpeg and ffprobe not found on PATH.")

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
class ZoomSequenceVideoNode:
    """
    Zooms IN or OUT across a video file. The canvas size stays fixed and
    the original aspect ratio is preserved for every frame.

    Takes a video file path and applies zoom effects, then outputs a new video
    with the same audio (if present).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": ""}),
                "direction": (["Zoom In", "Zoom Out", "Random"], {"default": "Zoom In"}),
                "amount_type": (["Pixels per Frame", "Target Percentage"], {"default": "Pixels per Frame"}),
                "pixels_per_frame": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.1}),
                "zoom_percentage": ("INT", {"default": 110, "min": 100, "max": 10000, "step": 1}),
                "ease": (["Linear", "Ease_In", "Ease_Out", "Ease_In_Out", "Random"], {"default": "Linear"}),
                "random_seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "smooth_subpixel": ("BOOLEAN", {"default": True}),
                "prefix": ("STRING", {"default": "zoom_sequence"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "apply_zoom_sequence"
    CATEGORY = "Simple Video Effects/Video Processing"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(
        cls,
        video_path,
        direction,
        amount_type,
        pixels_per_frame,
        zoom_percentage,
        ease,
        random_seed,
        smooth_subpixel,
        prefix,
    ):
        random_mode = (
            str(direction).strip().lower() == "random"
            or str(ease).strip().upper().replace("-", "_") == "RANDOM"
        )
        if random_mode and int(random_seed) == 0:
            # Force re-execution each run when using auto-seed random mode.
            # A monotonic nonce is more explicit/reliable than NaN for cache invalidation.
            return ("AUTO_RANDOM_NONCE", time.time_ns())
        return (
            str(video_path),
            str(direction),
            str(amount_type),
            float(pixels_per_frame),
            int(zoom_percentage),
            str(ease),
            int(random_seed),
            bool(smooth_subpixel),
            str(prefix),
        )

    # ---- helpers -------------------------------------------------------------
    def _resize_to(self, frame_hwc: torch.Tensor, size_hw):
        H, W = size_hw
        x = frame_hwc.permute(2, 0, 1).unsqueeze(0)  # 1,C,H,W
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        return x.squeeze(0).permute(1, 2, 0)

    # ---- main ---------------------------------------------------------------
    def apply_zoom_sequence(self,
                          video_path: str,
                          direction: str,
                          amount_type: str,
                          pixels_per_frame: float,
                          zoom_percentage: int,
                          ease: str,
                          random_seed: int,
                          smooth_subpixel: bool,
                          prefix: str):

        if not os.path.exists(video_path):
            raise ValueError(f"Video file not found: {video_path}")

        # Get video info
        duration, fps, width, height = get_video_info(video_path)

        seed_i, auto_seed = resolve_seed_value(random_seed)
        effective_direction, rolled_direction = resolve_direction_mode(direction, seed_i)
        effective_ease, rolled_random = resolve_ease_mode(ease, seed_i + 1)

        # Create output folder next to source video
        output_dir = Path(video_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        counter = 1
        while True:
            filename = f"{prefix}_{counter:03d}.mp4"
            output_path = output_dir / filename
            if not output_path.exists():
                break
            counter += 1

        pbar = ProgressBar(3)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Extract frames
            frames_dir = extract_frames(video_path, tmpdir, fps)
            images = load_frames(frames_dir)
            pbar.update(1)

            # Apply zoom sequence (same logic as image version)
            zoomed_images, info = self._apply_zoom(
                images,
                effective_direction,
                amount_type,
                pixels_per_frame,
                zoom_percentage,
                effective_ease,
                smooth_subpixel,
                rolled_direction,
                rolled_random,
                seed_i,
                auto_seed,
            )

            # Save processed frames
            processed_frames_dir = tmpdir / "processed_frames"
            processed_frames_dir.mkdir()

            for i, img_tensor in enumerate(zoomed_images):
                # Convert to PIL
                arr = (img_tensor.clamp(0, 1) * 255).byte().numpy()
                from PIL import Image
                img = Image.fromarray(arr)
                img.save(processed_frames_dir / f"frame_{i:06d}.png")
            
            pbar.update(1)

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
            pbar.update(1)

        return (str(output_path),)

    def _apply_zoom(self, images, direction, amount_type, pixels_per_frame, zoom_percentage, ease, smooth_subpixel, rolled_direction, rolled_random, random_seed, auto_seed):
        """Apply zoom sequence to images (copied from ZoomSequenceNode)"""
        if images.ndim != 4:
            return (images, "Input is not a batched IMAGE (B,H,W,C).")
        B, H, W, C = images.shape
        if B <= 0:
            return (images, "Empty batch; nothing to do.")

        small = min(W, H)
        m_smalls_f, meta = compute_zoom_margins(
            frame_count=B,
            small_dim=small,
            direction=direction,
            amount_type=amount_type,
            pixels_per_frame=pixels_per_frame,
            zoom_percentage=zoom_percentage,
            ease=ease,
            timeline_start=0,
            timeline_total=B,
        )

        # Apply proportional, aspect-corrected crops
        out_frames = []
        clamped = bool(meta.get("clamped", False))
        for i in range(B):
            if smooth_subpixel:
                zf = margin_to_zoom_factor(m_smalls_f[i], small)
                out_frames.append(apply_center_zoom_subpixel(images[i], zf, mode="bilinear"))
            else:
                # Round to integer ONLY where slicing happens
                m_small_int = int(round(m_smalls_f[i]))

                x0, y0, x1, y1, _, _ = aspect_corrected_crop_box(W, H, m_small_int)

                if x1 <= x0 or y1 <= y0:
                    out_frames.append(images[i])
                    continue

                cropped = images[i, y0:y1, x0:x1, :]
                resized = self._resize_to(cropped, (H, W))
                out_frames.append(resized)

        out = torch.stack(out_frames, dim=0)

        info_lines = []
        info_lines.append(
            f"Frames: {B}, Canvas: {W}x{H}, Direction: {direction}, Amount Type: {amount_type}, Ease: {ease}"
        )
        if rolled_direction:
            info_lines.append(f"Direction Random roll selected: {direction}")
        if rolled_random:
            info_lines.append(f"Ease Random roll selected:      {ease}")
        if rolled_direction or rolled_random:
            info_lines.append(
                f"Random seed:                   {int(random_seed)}{' (auto from 0)' if auto_seed else ''}"
            )
        info_lines.append(f"Requested small-dim max margin: {meta['requested_small_margin_max_f']:.2f} px")
        info_lines.append(
            f"Applied small-dim max margin:   {meta['applied_small_margin_max_f']:.2f} px "
            f"(safe limit: {meta['max_safe_small_margin']} px)"
        )
        info_lines.append(f"Effective max zoom reached:     {meta['effective_max_zoom']:.4f}x")
        info_lines.append(f"Transform mode:                 {'Subpixel (grid_sample)' if smooth_subpixel else 'Integer crop/resize'}")
        if str(amount_type).strip().lower() == "target percentage":
            info_lines.append(f"Target zoom percentage:         {int(zoom_percentage)}%")
        if smooth_subpixel:
            info_lines.append("Note: margins are converted to continuous zoom factors and applied with subpixel sampling.")
        else:
            info_lines.append("Note: margins are proportional per axis to preserve aspect; integer crop indices are used.")
        if clamped:
            info_lines.append("Warning: requested margins exceeded safe bounds and were clamped.")
        info = "\n".join(info_lines)

        return (out, info)

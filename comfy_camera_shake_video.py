import os
import shutil
import subprocess
import tempfile
import math
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

def normalize_mode(mode: str) -> str:
    """
    Interpret UI mode string as one of our internal shake modes.
    """
    m = mode.strip().lower()
    if "circle" in m:
        return "CIRCULAR"
    if "random" in m:
        return "RANDOM"
    # Default to circular if unclear
    return "CIRCULAR"

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
class CameraShakeVideoNode:
    """
    Camera shake / procedural handheld motion for a video file.

    Takes a video file path and applies shake effects, then outputs a new video
    with the same audio (if present).

    Supports circular or random shake patterns with configurable intensity and easing.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": ""}),
                "mode": (["Circular Shake", "Random Shake"], {"default": "Circular Shake"}),
                "pixels_per_frame": ("FLOAT", {"default": 5.0, "min": 0.0, "step": 0.1}),
                "ease": (["Linear", "Ease_In", "Ease_Out", "Ease_In_Out"], {"default": "Linear"}),
                "prefix": ("STRING", {"default": "camera_shake"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "apply_camera_shake"
    CATEGORY = "Simple Video Effects/Video Processing"
    OUTPUT_NODE = True

    # ---- helpers -------------------------------------------------------------
    def _resize_to(self, frame_hwc: torch.Tensor, size_hw):
        H, W = size_hw
        x = frame_hwc.permute(2, 0, 1).unsqueeze(0)  # 1,C,H,W
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        return x.squeeze(0).permute(1, 2, 0)

    def _aspect_corrected_crop_box(self, W: int, H: int, m_small_int: int):
        """
        Given an integer scalar margin on the smaller dimension (per-side),
        compute integer crop box (x0,y0,x1,y1) that:
          - preserves aspect (within ±1 px due to rounding),
          - never exceeds bounds.
        """
        aspect = W / H if H > 0 else 1.0

        # Which side is smaller?
        small = min(W, H)
        if small <= 2:
            return 0, 0, W, H, 0, 0

        # Safe guard: ensure at least 1 px interior
        max_small_margin = max(0, (small // 2) - 1)
        m_small = max(0, min(int(m_small_int), max_small_margin))

        # Fractional level based on smaller half-size
        half_small = small / 2.0
        frac = 0.0 if half_small <= 0 else (m_small / half_small)  # 0..~1

        # Proportional margins per axis to keep aspect (before integer correction)
        m_x = int(round(frac * (W / 2.0)))
        m_y = int(round(frac * (H / 2.0)))

        # Compute crop size
        cw = W - 2 * m_x
        ch = H - 2 * m_y

        # Enforce minimum interior
        cw = max(1, cw)
        ch = max(1, ch)

        # ---- Aspect correction (integer) ------------------------------------
        # target: cw / ch == W / H  =>  cw == round(ch * aspect)
        ideal_cw = int(round(ch * aspect))
        ideal_cw = max(1, min(ideal_cw, W - 2))  # keep some border to be safe in edge cases

        if ideal_cw != cw:
            # Recompute m_x from ideal_cw, keep m_y if possible
            cw = ideal_cw
            m_x = (W - cw) // 2
            # ensure symmetric crop; parity fix if needed
            if W - 2 * m_x != cw:
                m_x2 = max(0, min(m_x + 1, (W - 1) // 2))
                if W - 2 * m_x2 == cw:
                    m_x = m_x2
                else:
                    # else recompute ch from cw to restore symmetry
                    ch = max(1, int(round(cw / aspect)))
                    m_y = (H - ch) // 2

        # Final clamp to safe range
        m_x = max(0, min(m_x, (W - 1) // 2))
        m_y = max(0, min(m_y, (H - 1) // 2))

        x0, y0 = m_x, m_y
        x1, y1 = W - m_x, H - m_y
        return x0, y0, x1, y1, m_x, m_y

    # ---- main ---------------------------------------------------------------
    def apply_camera_shake(self,
                          video_path: str,
                          mode: str,
                          pixels_per_frame: float,
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

            # Apply camera shake (same logic as image version)
            shaken_images, info = self._apply_shake(images, mode, pixels_per_frame, ease)

            # Save processed frames
            processed_frames_dir = tmpdir / "processed_frames"
            processed_frames_dir.mkdir()

            for i, img_tensor in enumerate(shaken_images):
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

    def _apply_shake(self, images, mode, pixels_per_frame, ease):
        """Apply camera shake to images (copied from CameraShakeNode)"""
        if images.ndim != 4:
            return (images, "Input is not a batched IMAGE (B,H,W,C).")
        B, H, W, C = images.shape
        if B <= 0:
            return (images, "Empty batch; nothing to do.")

        # Normalize
        shake_mode = normalize_mode(mode)
        i_ease = ease

        # Compute scalar radius on the smaller dimension (per-side)
        small = min(W, H)
        max_safe_small_margin = max(0, (small // 2) - 1)

        requested_radius_small_f = max(0.0, float(pixels_per_frame))
        radius_small_f = min(requested_radius_small_f, float(max_safe_small_margin))

        # Convert to integer margin and compute the "safe inner canvas"
        m_small_int = int(round(radius_small_f))
        # This gives us a central crop with margins m_x, m_y
        x0c, y0c, x1c, y1c, m_x, m_y = self._aspect_corrected_crop_box(W, H, m_small_int)
        cw = x1c - x0c
        ch = y1c - y0c

        if cw <= 0 or ch <= 0:
            return (images, "Invalid crop; probably too large shake vs resolution.")

        # Eased progress 0..1 across frames
        ts = [0.0] if B == 1 else [i / (B - 1) for i in range(B)]
        es = [ease_value(t, i_ease) for t in ts]

        out_frames = []
        # How many circles across the sequence for circular shake
        num_cycles = 2.0

        for i in range(B):
            t = ts[i]
            # Envelope (0..1) over time for radius
            radius_factor = es[i] if B > 1 else 1.0

            # Per-axis max amplitude for this frame
            ax = m_x * radius_factor
            ay = m_y * radius_factor

            if shake_mode == "CIRCULAR":
                # Smooth sinusoidal circular path (like smooth_camera_shake, no jitter)
                # num_cycles = how many full circles over the whole sequence
                angle = 2.0 * math.pi * num_cycles * t
                dx_f = ax * math.sin(angle)
                dy_f = ay * math.sin(angle + math.pi * 0.5)

            elif shake_mode == "RANDOM":
                # Constant-magnitude random shake.
                # If pixels_per_frame = 1 on the small dimension, this results in
                # a 1-pixel step (left/right/up/down/diagonal) every frame,
                # and it does NOT ramp up over time.

                # Base step size from margins (already derived from requested radius)
                step_x = m_x if m_x > 0 else 0
                step_y = m_y if m_y > 0 else 0

                # Ensure we still get motion if margins are tiny but non-zero
                if step_x == 0 and m_x > 0:
                    step_x = 1
                if step_y == 0 and m_y > 0:
                    step_y = 1

                # Random direction in {-1, 0, +1} for each axis
                dir_x = int(torch.randint(-1, 2, (1,)).item())
                dir_y = int(torch.randint(-1, 2, (1,)).item())

                # Ignore the easing envelope for magnitude here – keep step size constant
                dx_f = dir_x * step_x
                dy_f = dir_y * step_y

            else:
                # Fallback: no movement
                dx_f = 0.0
                dy_f = 0.0

            # Clamp to safe integer offsets (so crop stays inside original)
            dx = int(round(max(-m_x, min(m_x, dx_f))))
            dy = int(round(max(-m_y, min(m_y, dy_f))))

            # Base central crop is at (m_x, m_y); we offset it by dx, dy
            x0 = m_x + dx
            y0 = m_y + dy
            x1 = x0 + cw
            y1 = y0 + ch

            # Extra safety clamp against rounding drift
            if x0 < 0:
                x0 = 0
                x1 = cw
            if y0 < 0:
                y0 = 0
                y1 = ch
            if x1 > W:
                x1 = W
                x0 = W - cw
            if y1 > H:
                y1 = H
                y0 = H - ch

            cropped = images[i, y0:y1, x0:x1, :]
            resized = self._resize_to(cropped, (H, W))
            out_frames.append(resized)

        out = torch.stack(out_frames, dim=0)

        info_lines = []
        info_lines.append(f"Frames: {B}, Canvas: {W}x{H}")
        info_lines.append(f"Mode: {shake_mode}, Ease: {ease}")
        info_lines.append(f"Requested small-dim radius: {requested_radius_small_f:.2f} px")
        info_lines.append(f"Applied small-dim radius:   {radius_small_f:.2f} px (safe limit: {max_safe_small_margin} px)")
        info_lines.append(f"Per-axis margins (max shake): m_x={m_x}, m_y={m_y}")
        info_lines.append("Note: frames are cropped with a safe border then resized back to avoid black edges during shake.")
        info = "\n".join(info_lines)

        return (out, info)

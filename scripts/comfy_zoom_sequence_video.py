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
    m = mode.strip().lower()
    if m in ("zoom in", "in"):
        return "IN"
    if m in ("zoom out", "out"):
        return "OUT"
    return "IN"

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
                "mode": (["Zoom In", "Zoom Out"], {"default": "Zoom In"}),
                "pixels_per_frame": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.1}),
                "ease": (["Linear", "Ease_In", "Ease_Out", "Ease_In_Out"], {"default": "Linear"}),
                "prefix": ("STRING", {"default": "zoom_sequence"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "apply_zoom_sequence"
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

        # Fractional zoom level based on smaller half-size
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
    def apply_zoom_sequence(self,
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

            # Apply zoom sequence (same logic as image version)
            zoomed_images, info = self._apply_zoom(images, mode, pixels_per_frame, ease)

            # Save processed frames
            processed_frames_dir = tmpdir / "processed_frames"
            processed_frames_dir.mkdir()

            for i, img_tensor in enumerate(zoomed_images):
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

    def _apply_zoom(self, images, mode, pixels_per_frame, ease):
        """Apply zoom sequence to images (copied from ZoomSequenceNode)"""
        if images.ndim != 4:
            return (images, "Input is not a batched IMAGE (B,H,W,C).")
        B, H, W, C = images.shape
        if B <= 0:
            return (images, "Empty batch; nothing to do.")

        # Normalize
        i_mode = normalize_mode(mode)
        i_ease = ease

        # Compute max scalar margin ON THE SMALLER DIMENSION (per-side)
        small = min(W, H)
        max_safe_small_margin = max(0, (small // 2) - 1)

        # keep as float (allows fractional speed), clamp later when actually cropping
        requested_small_margin_max_f = pixels_per_frame * max(0, B - 1)
        small_margin_max_f = min(requested_small_margin_max_f, float(max_safe_small_margin))

        # Eased progress 0..1 across frames
        ts = [0.0] if B == 1 else [i / (B - 1) for i in range(B)]
        es = [ease_value(t, i_ease) for t in ts]

        # Per-frame scalar margins (float, on the smaller dimension)
        if i_mode == "IN":
            m_smalls_f = [small_margin_max_f * e for e in es]
        else:  # OUT
            m_smalls_f = [small_margin_max_f * (1.0 - e) for e in es]

        # Apply proportional, aspect-corrected crops
        out_frames = []
        clamped = False
        used_mx = []
        used_my = []
        for i in range(B):
            # Round to integer ONLY where slicing happens
            m_small_int = int(round(m_smalls_f[i]))
            # Detect clamping relative to theoretical safe limit
            if m_small_int != max(0, min(m_small_int, (small // 2) - 1)):
                clamped = True

            x0, y0, x1, y1, m_x, m_y = self._aspect_corrected_crop_box(W, H, m_small_int)

            if x1 <= x0 or y1 <= y0:
                out_frames.append(images[i])
                continue

            cropped = images[i, y0:y1, x0:x1, :]
            resized = self._resize_to(cropped, (H, W))
            out_frames.append(resized)
            used_mx.append(m_x)
            used_my.append(m_y)

        out = torch.stack(out_frames, dim=0)

        info_lines = []
        info_lines.append(f"Frames: {B}, Canvas: {W}x{H}, Mode: {mode}, Ease: {ease}")
        info_lines.append(f"Requested small-dim max margin: {requested_small_margin_max_f:.2f} px")
        info_lines.append(f"Applied small-dim max margin:   {small_margin_max_f:.2f} px (safe limit: {max_safe_small_margin} px)")
        info_lines.append("Note: margins are proportional per axis to preserve aspect; integer crop indices are used.")
        if clamped:
            info_lines.append("Warning: requested margins exceeded safe bounds and were clamped.")
        info = "\n".join(info_lines)

        return (out, info)

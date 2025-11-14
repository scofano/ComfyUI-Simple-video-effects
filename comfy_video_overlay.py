import os
import shutil
import subprocess
import json
import tempfile
from pathlib import Path

import torch
import numpy as np
from PIL import Image

# tqdm for progress (with safe fallback)
try:
    from tqdm import tqdm
except ImportError:  # if tqdm isn't installed, just pass through
    def tqdm(x, *args, **kwargs):
        return x

# ---- ffmpeg helpers (adapted from videoverlay.py) ----

def which_or_die(name: str) -> str:
    p = shutil.which(name)
    if not p:
        raise RuntimeError(f"Error: '{name}' not found on PATH.")
    return p

FFMPEG = which_or_die("ffmpeg")
FFPROBE = which_or_die("ffprobe")


def run(cmd: list[str]) -> None:
    print("[VideoOverlay] Running command:", " ".join(cmd), flush=True)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        pretty = " ".join(f'"{c}"' if " " in c else c for c in cmd)
        raise RuntimeError(f"Command failed ({e.returncode}):\n{pretty}\n{e}") from e


def ffprobe_json(path: Path) -> dict:
    prog = FFPROBE if "ffprobe" in FFPROBE else FFMPEG.replace("ffmpeg", "ffprobe")
    out = subprocess.check_output([
        prog,
        "-v", "error",
        "-show_streams",
        "-show_format",      # include format so we can get duration reliably
        "-print_format", "json",
        str(path),
    ])
    return json.loads(out.decode("utf-8"))


def get_video_stream_info(path: Path) -> dict | None:
    info = ffprobe_json(path)
    for s in info.get("streams", []):
        if s.get("codec_type") == "video":
            return s
    return None


def get_dimensions(path: Path) -> tuple[int, int]:
    s = get_video_stream_info(path)
    if not s:
        raise RuntimeError(f"No video stream found in: {path}")
    try:
        w, h = int(s["width"]), int(s["height"])
        if w <= 0 or h <= 0:
            raise ValueError
        return w, h
    except Exception:
        raise RuntimeError(f"Could not determine dimensions for: {path}")


def get_duration(path: Path) -> float:
    """
    Returns duration in seconds (float) using ffprobe format/stream info.
    """
    info = ffprobe_json(path)

    dur = None
    # Prefer container (format) duration if available
    fmt = info.get("format", {})
    if "duration" in fmt:
        try:
            dur = float(fmt["duration"])
        except Exception:
            dur = None

    # Fallback to video stream duration
    if dur is None:
        for s in info.get("streams", []):
            if s.get("codec_type") == "video" and "duration" in s:
                try:
                    dur = float(s["duration"])
                    break
                except Exception:
                    pass

    if dur is None or dur <= 0:
        raise RuntimeError(f"Could not determine duration for: {path}")

    return dur


def build_filter_complex(base_w: int, base_h: int, opacity: float) -> str:
    """
    Simpler, robust filter graph:
    - Convert both inputs to planar RGB.
    - Scale overlay to match base dimensions.
    - Make overlay grayscale (so multiply behaves as a neutral luminance mask).
    - Multiply blend.
    - Convert to yuv420p for libx264.
    """
    return (
        # Base: just get to RGB planar, fix SAR
        f"[0:v]format=gbrp,setsar=1[base];"
        # Overlay: scale to base size, gray -> gbrp so channels match
        f"[1:v]scale={base_w}:{base_h},format=gray,format=gbrp,setsar=1[ov];"
        # Multiply blend
        f"[base][ov]blend=all_mode=multiply:all_opacity={opacity}[rgb];"
        # Back to yuv420p for encoding
        f"[rgb]format=yuv420p[v]"
    )


def run_ffmpeg_overlay(base: Path, overlay: Path, out_path: Path,
                       opacity: float = 0.75,
                       crf: int = 18,
                       preset: str = "medium") -> None:
    print("[VideoOverlay] Probing base video for dimensions and duration...", flush=True)
    base_w, base_h = get_dimensions(base)
    base_dur = get_duration(base)
    overlay_dur = get_duration(overlay)
    print(f"[VideoOverlay] Base size: {base_w}x{base_h}, duration: {base_dur:.3f}s", flush=True)
    print(f"[VideoOverlay] Overlay duration: {overlay_dur:.3f}s", flush=True)

    fc = build_filter_complex(base_w, base_h, opacity)

    # Decide how to handle overlay vs base duration
    overlay_for_blend = overlay
    eps = 0.01  # small epsilon for float comparison

    if overlay_dur + eps < base_dur:
        # Overlay is shorter -> create a looped version exactly as long as base
        print("[VideoOverlay] Overlay shorter than base. Creating looped overlay to match base duration...", flush=True)
        looped_overlay = out_path.with_name("overlay_looped.mp4")
        cmd_loop = [
            FFMPEG,
            "-hide_banner", "-loglevel", "error",
            "-y",
            "-stream_loop", "-1",    # loop overlay
            "-i", str(overlay),
            "-t", f"{base_dur:.3f}", # cut to base duration
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(looped_overlay),
        ]
        run(cmd_loop)
        overlay_for_blend = looped_overlay
    elif overlay_dur > base_dur + eps:
        print("[VideoOverlay] Overlay longer than base. It will be trimmed to base duration.", flush=True)
    else:
        print("[VideoOverlay] Overlay and base durations are approximately equal.", flush=True)

    # Final blend: always cut output to base duration
    cmd = [
        FFMPEG,
        "-hide_banner", "-loglevel", "error",
        "-y",
        "-i", str(base),
        "-i", str(overlay_for_blend),
        "-filter_complex", fc,
        "-map", "[v]",
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", preset,
        "-t", f"{base_dur:.3f}",     # ensure output duration == base duration
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        "-colorspace", "bt709",
        "-color_range", "tv",
        str(out_path),
    ]
    run(cmd)


# ---- IMAGE <-> VIDEO helpers for ComfyUI ----

def tensor_to_png_sequence(images: torch.Tensor, out_dir: Path) -> None:
    """
    images: [N, H, W, C] float32 0..1 (ComfyUI IMAGE)
    Saves as out_dir/frame_000001.png ...
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    images = images.detach().cpu()
    n = images.shape[0]

    print(f"[VideoOverlay] Saving {n} input frames to {out_dir}", flush=True)
    for i in tqdm(range(n), desc="[VideoOverlay] Saving input frames"):
        frame = images[i]
        # Clamp & convert to uint8
        frame = (frame.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).numpy()
        # Ensure 3 channels
        if frame.shape[-1] == 1:
            frame = np.repeat(frame, 3, axis=-1)
        elif frame.shape[-1] == 4:
            frame = frame[..., :3]
        img = Image.fromarray(frame)
        img.save(out_dir / f"frame_{i:06d}.png")


def png_sequence_to_tensor(in_dir: Path) -> torch.Tensor:
    """
    Loads frame_*.png and returns [N, H, W, C] float32 0..1 tensor.
    """
    frames = sorted(
        [p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"]
    )
    if not frames:
        raise RuntimeError(f"No PNG frames found in {in_dir}")

    print(f"[VideoOverlay] Loading {len(frames)} output frames from {in_dir}", flush=True)
    imgs = []
    for p in tqdm(frames, desc="[VideoOverlay] Loading output frames"):
        img = Image.open(p).convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        imgs.append(arr)

    arr = np.stack(imgs, axis=0)  # [N, H, W, C]
    return torch.from_numpy(arr)


def png_sequence_to_video(in_dir: Path, out_path: Path, fps: int = 24) -> None:
    """
    Uses ffmpeg to encode PNG sequence -> mp4.
    """
    print(f"[VideoOverlay] Encoding PNGs -> video at {out_path}", flush=True)
    cmd = [
        FFMPEG,
        "-hide_banner", "-loglevel", "error",
        "-y",
        "-framerate", str(fps),
        "-i", str(in_dir / "frame_%06d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    run(cmd)


def video_to_png_sequence(video_path: Path, out_dir: Path) -> None:
    """
    Uses ffmpeg to decode video -> PNG frames.
    """
    print(f"[VideoOverlay] Decoding video -> PNGs from {video_path} to {out_dir}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        FFMPEG,
        "-hide_banner", "-loglevel", "error",
        "-y",
        "-i", str(video_path),
        str(out_dir / "frame_%06d.png"),
    ]
    run(cmd)


# ---- ComfyUI Node ----

class VideoOverlay:
    """
    ComfyUI node:
      - Input:  IMAGE (video sequence)
      - Input:  overlay_video_path (STRING)
      - Output: IMAGE (overlayed sequence)

    Video operations are done only with ffmpeg.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "overlay_video_path": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "",
                    },
                ),
                "opacity": (
                    "FLOAT",
                    {
                        "default": 0.75,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_overlay"
    CATEGORY = "Video"

    def apply_overlay(self, images, overlay_video_path: str, opacity: float):
        if not overlay_video_path:
            raise RuntimeError("overlay_video_path is empty.")
        overlay_video = Path(overlay_video_path)
        if not overlay_video.exists():
            raise RuntimeError(f"Overlay video not found: {overlay_video}")

        # Clamp opacity
        opacity = float(max(0.0, min(1.0, opacity)))

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            print(f"[VideoOverlay] Working in temp dir: {tmpdir}", flush=True)

            # 1) IMAGE tensor -> PNG sequence -> base video
            png_in_dir = tmpdir / "in_frames"
            tensor_to_png_sequence(images, png_in_dir)

            base_video_path = tmpdir / "base.mp4"
            png_sequence_to_video(png_in_dir, base_video_path, fps=24)

            # 2) ffmpeg overlay + duration logic
            overlaid_video_path = tmpdir / "overlaid.mp4"
            run_ffmpeg_overlay(
                base=base_video_path,
                overlay=overlay_video,
                out_path=overlaid_video_path,
                opacity=opacity,
                crf=18,
                preset="medium",
            )

            # 3) overlaid video -> PNG sequence -> IMAGE tensor
            png_out_dir = tmpdir / "out_frames"
            video_to_png_sequence(overlaid_video_path, png_out_dir)
            out_tensor = png_sequence_to_tensor(png_out_dir)

        # ComfyUI expects a batch of images: [N, H, W, C]
        print("[VideoOverlay] Done, returning tensor with shape", tuple(out_tensor.shape), flush=True)
        return (out_tensor,)
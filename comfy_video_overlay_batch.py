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

# ComfyUI output helper
try:
    import folder_paths
except ImportError:
    folder_paths = None

# ---- ffmpeg helpers ----

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


def run_ffmpeg_with_progress(cmd: list[str], total_duration: float) -> None:
    """
    Run ffmpeg command and show tqdm progress from 0% to 100% based on out_time_ms.
    """
    print("[VideoOverlay] Running command with progress:", " ".join(cmd), flush=True)
    # We rely on -progress pipe:1 (stdout)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    total_duration = max(float(total_duration), 0.001)
    last_t = 0.0

    with tqdm(
        total=total_duration,
        desc="[VideoOverlay] ffmpeg overlay",
        unit="s",
        leave=True,
    ) as pbar:
        if proc.stdout is not None:
            for line in proc.stdout:
                line = line.strip()
                if line.startswith("out_time_ms="):
                    try:
                        ms = float(line.split("=", 1)[1])
                        t = ms / 1_000_000.0
                    except Exception:
                        continue
                    if t > total_duration:
                        t = total_duration
                    if t > last_t:
                        pbar.update(t - last_t)
                        last_t = t

        # Ensure bar reaches the end
        if last_t < total_duration:
            pbar.update(total_duration - last_t)

    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"ffmpeg command failed with return code {ret}")


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
    temp_overlay = None          # track temp loop file
    eps = 0.01  # small epsilon for float comparison

    if overlay_dur + eps < base_dur:
        # Overlay is shorter -> create a looped version exactly as long as base
        print("[VideoOverlay] Overlay shorter than base. Creating looped overlay to match base duration...", flush=True)
        looped_overlay = out_path.with_name(out_path.stem + "_overlay_looped.mp4")
        temp_overlay = looped_overlay
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
    # Video from filter [v], audio from base input (0:a)
    cmd = [
        FFMPEG,
        "-hide_banner", "-loglevel", "error",
        "-y",
        "-progress", "pipe:1",
        "-nostats",
        "-i", str(base),              # 0: base (video+audio)
        "-i", str(overlay_for_blend), # 1: overlay (video only used)
        "-filter_complex", fc,
        "-map", "[v]",                # blended video
        "-map", "0:a?",               # ORIGINAL audio from base (if present)
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", preset,
        "-c:a", "copy",               # keep original audio encoding
        "-t", f"{base_dur:.3f}",      # ensure output duration == base duration
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        "-colorspace", "bt709",
        "-color_range", "tv",
        str(out_path),
    ]
    run_ffmpeg_with_progress(cmd, total_duration=base_dur)

    # Clean up temp looped overlay if we created one
    if temp_overlay is not None:
        try:
            if temp_overlay.exists():
                temp_overlay.unlink()
                print(f"[VideoOverlay] Deleted temp overlay file: {temp_overlay}", flush=True)
        except Exception as e:
            print(f"[VideoOverlay] Warning: could not delete temp overlay file {temp_overlay}: {e}", flush=True)


# ---- (legacy) IMAGE <-> VIDEO helpers (kept for compatibility, not used by new node) ----

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


# ---- ComfyUI Node (updated) ----

def _sanitize_prefix(prefix: str) -> str:
    if not prefix:
        return ""
    invalid_chars = '<>:"/\\|?*'
    for ch in invalid_chars:
        prefix = prefix.replace(ch, "_")
    return prefix


class VideoOverlayBatch:
    """
    Updated ComfyUI node:

      Inputs:
        - base_video_path: STRING (full path to original/base video)
        - overlay_video_path: STRING (full path to overlay video)
        - prefix: STRING (filename prefix in ComfyUI output folder)
        - opacity: FLOAT

      Behavior:
        - Creates an overlaid video using ffmpeg.
        - Saves result into default ComfyUI output folder.
        - Shows tqdm progress from 0% to 100% based on ffmpeg progress.
        - Returns the full output filename as two STRING outputs
          (node works fine even if the second/string-only output is not connected).
    """
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_video_path": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "",
                    },
                ),
                "overlay_video_path": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "",
                    },
                ),
                "prefix": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "overlay_",
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

    # Two STRING outputs: same filename, one can be used purely as "string" output
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_output_path",)
    FUNCTION = "apply_overlay"
    CATEGORY = "Simple Video Effects"

    def apply_overlay(self,
                      base_video_path: str,
                      overlay_video_path: str,
                      prefix: str,
                      opacity: float):

        if not base_video_path:
            raise RuntimeError("base_video_path is empty.")
        if not overlay_video_path:
            raise RuntimeError("overlay_video_path is empty.")

        base_video = Path(base_video_path)
        overlay_video = Path(overlay_video_path)

        if not base_video.exists():
            raise RuntimeError(f"Base video not found: {base_video}")
        if not overlay_video.exists():
            raise RuntimeError(f"Overlay video not found: {overlay_video}")

        # Clamp opacity
        opacity = float(max(0.0, min(1.0, opacity)))

        # Resolve ComfyUI output directory
        if folder_paths is not None:
            output_dir = Path(folder_paths.get_output_directory())
        else:
            # Fallback to ./output if folder_paths isn't available (e.g. running standalone)
            output_dir = Path("./output")
        output_dir.mkdir(parents=True, exist_ok=True)

        prefix_sanitized = _sanitize_prefix(prefix)
        out_name = f"{prefix_sanitized}{base_video.stem}_overlay.mp4"
        out_path = output_dir / out_name

        print(f"[VideoOverlay] Output video will be written to: {out_path}", flush=True)

        run_ffmpeg_overlay(
            base=base_video,
            overlay=overlay_video,
            out_path=out_path,
            opacity=opacity,
            crf=18,
            preset="medium",
        )

        full_path_str = str(out_path.resolve())
        print(f"[VideoOverlay] Done. Output video: {full_path_str}", flush=True)

        # Node works fine even if either output is not connected.
        return (full_path_str,)

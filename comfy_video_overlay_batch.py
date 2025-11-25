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


# Try to import ComfyUI's folder_paths helper
try:
    import folder_paths  # type: ignore
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
    """
    Run ffprobe and parse JSON output.
    """
    prog = FFPROBE if "ffprobe" in Path(FFPROBE).name else FFMPEG.replace("ffmpeg", "ffprobe")
    out = subprocess.check_output([
        prog,
        "-v", "error",
        "-show_streams",
        "-show_format",
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
    Get duration in seconds, preferring container duration but falling back to
    the video stream if necessary.
    """
    info = ffprobe_json(path)

    dur = None
    fmt = info.get("format", {})
    if "duration" in fmt:
        try:
            dur = float(fmt["duration"])
        except Exception:
            dur = None

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
    Build a robust filter graph that:
      - scales & pads base and overlay to the same size
      - applies alpha to overlay
      - blends overlay on top of base
    """
    # Keep it simple: scale+pad both streams to base_w x base_h, then blend.
    fc = (
        "[0:v]scale={w}:{h}:force_original_aspect_ratio=decrease,"
        "pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,format=rgba[base];"
        "[1:v]scale={w}:{h}:force_original_aspect_ratio=decrease,"
        "pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,format=rgba,"
        "colorchannelmixer=aa={op}[ovr];"
        "[base][ovr]overlay=0:0:format=auto[v]"
    ).format(w=base_w, h=base_h, op=opacity)
    return fc


def run_ffmpeg_with_progress(cmd: list[str], total_duration: float | None = None) -> None:
    """
    Run ffmpeg, parsing -progress output and displaying a tqdm progress bar
    based on out_time_ms.
    """
    print("[VideoOverlay] Running ffmpeg with progress:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    if proc.stdout is None:
        # Shouldn't happen, but just in case
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"ffmpeg command failed with return code {ret}")
        return

    if total_duration is None or total_duration <= 0:
        # No progress bar possible, just echo lines
        for line in proc.stdout:
            line = line.rstrip("\n")
            if line:
                print("[ffmpeg]", line, flush=True)
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"ffmpeg command failed with return code {ret}")
        return

    with tqdm(total=int(total_duration), unit="s", desc="[VideoOverlay] ffmpeg") as pbar:
        last_t = 0.0
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue

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

            elif line.startswith("progress="):
                value = line.split("=", 1)[1]
                if value == "end" and last_t < total_duration:
                    pbar.update(total_duration - last_t)

        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"ffmpeg command failed with return code {ret}")


def run_ffmpeg_overlay(
    base: Path,
    overlay: Path,
    out_path: Path,
    opacity: float = 0.75,
    crf: int = 18,
    preset: str = "medium",
    use_gpu: bool = True,
) -> None:
    print("[VideoOverlay] Probing base video for dimensions and duration...", flush=True)
    base_w, base_h = get_dimensions(base)
    base_dur = get_duration(base)
    overlay_dur = get_duration(overlay)
    print(
        f"[VideoOverlay] Base size: {base_w}x{base_h}, duration: {base_dur:.3f}s",
        flush=True,
    )
    print(f"[VideoOverlay] Overlay duration: {overlay_dur:.3f}s", flush=True)

    # Decide codec / hwaccel based on GPU toggle and encoder availability
    video_codec = "libx264"
    hwaccel_args: list[str] = []

    if use_gpu:
        try:
            encoders = subprocess.check_output(
                [FFMPEG, "-hide_banner", "-encoders"],
                stderr=subprocess.STDOUT,
                text=True,
            )
            if "h264_nvenc" in encoders:
                video_codec = "h264_nvenc"
                hwaccel_args = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
                print("[VideoOverlay] Using GPU encoder: h264_nvenc", flush=True)
            else:
                print(
                    "[VideoOverlay] USE_GPU is enabled, but h264_nvenc encoder was "
                    "not found. Falling back to libx264.",
                    flush=True,
                )
        except Exception as e:
            print(
                f"[VideoOverlay] USE_GPU is enabled, but encoder probe failed: {e}. "
                "Falling back to libx264.",
                flush=True,
            )

    fc = build_filter_complex(base_w, base_h, opacity)

    # Decide how to handle overlay vs base duration
    overlay_for_blend = overlay
    temp_overlay: Path | None = None
    eps = 0.01  # small epsilon for float comparison

    if overlay_dur + eps < base_dur:
        # Overlay is shorter -> create a looped version exactly as long as base
        print(
            "[VideoOverlay] Overlay shorter than base. Creating looped overlay to "
            "match base duration...",
            flush=True,
        )
        looped_overlay = out_path.with_name(out_path.stem + "_overlay_looped.mp4")
        temp_overlay = looped_overlay
        cmd_loop = [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
#            *hwaccel_args,
            "-stream_loop",
            "-1",
            "-i",
            str(overlay),
            "-t",
            f"{base_dur:.3f}",
            "-c:v",
            video_codec,
            "-pix_fmt",
            "yuv420p",
            str(looped_overlay),
        ]
        run(cmd_loop)
        overlay_for_blend = looped_overlay
    elif overlay_dur > base_dur + eps:
        print(
            "[VideoOverlay] Overlay longer than base. It will be trimmed to base "
            "duration.",
            flush=True,
        )
    else:
        print(
            "[VideoOverlay] Overlay and base durations are approximately equal.",
            flush=True,
        )

    # Final blend: always cut output to base duration
    # Video from filter [v], audio from base input (0:a)
    cmd = [
        FFMPEG,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-progress",
        "pipe:1",
        "-nostats",
        *hwaccel_args,
        "-i",
        str(base),
        "-i",
        str(overlay_for_blend),
        "-filter_complex",
        fc,
        "-map",
        "[v]",
        "-map",
        "0:a?",
        "-c:v",
        video_codec,
        "-crf",
        str(crf),
        "-preset",
        preset,
        "-c:a",
        "copy",
        "-t",
        f"{base_dur:.3f}",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-colorspace",
        "bt709",
        "-color_range",
        "tv",
        str(out_path),
    ]
    run_ffmpeg_with_progress(cmd, total_duration=base_dur)

    # Clean up temp looped overlay if we created one
    if temp_overlay is not None:
        try:
            if temp_overlay.exists():
                temp_overlay.unlink()
                print(
                    f"[VideoOverlay] Deleted temp overlay file: {temp_overlay}",
                    flush=True,
                )
        except Exception as e:
            print(
                f"[VideoOverlay] Warning: could not delete temp overlay file "
                f"{temp_overlay}: {e}",
                flush=True,

            )
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
        - delete_original: BOOLEAN (if True, delete the input/base video after processing)
        - use_gpu: BOOLEAN (if True, use GPU encoders when available)

      Behavior:
        - Creates an overlaid video using ffmpeg.
        - Saves result into default ComfyUI output folder.
        - Shows tqdm progress from 0% to 100% based on ffmpeg progress.
        - Returns the full output filename as STRING.
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
                "delete_original": (
                    "BOOLEAN",
                    {
                        "default": False,
                    },
                ),
                "use_gpu": (
                    "BOOLEAN",
                    {
                        "default": True,
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_output_path",)
    FUNCTION = "apply_overlay"
    CATEGORY = "Simple Video Effects"

    def apply_overlay(
        self,
        base_video_path: str,
        overlay_video_path: str,
        prefix: str,
        opacity: float,
        delete_original: bool = False,
        use_gpu: bool = True,
    ):
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
            use_gpu=use_gpu,
        )

        full_path_str = str(out_path.resolve())
        print(f"[VideoOverlay] Done. Output video: {full_path_str}", flush=True)

        # delete original base video file if requested
        if delete_original:
            try:
                base_video.unlink()
                print(
                    f"[VideoOverlay] Deleted original base video: {base_video}",
                    flush=True,
                )
            except Exception as e:
                print(
                    f"[VideoOverlay] Warning: could not delete base video "
                    f"{base_video}: {e}",
                    flush=True,
                )

        return (full_path_str,)


# Optional: ComfyUI-style mappings
NODE_CLASS_MAPPINGS = {
    "VideoOverlayBatch": VideoOverlayBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoOverlayBatch": "Video Overlay (Batch)",
}

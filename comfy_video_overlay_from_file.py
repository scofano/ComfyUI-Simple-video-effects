import os
import shutil
import subprocess
import tempfile
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import ffmpeg
import comfy.utils
import math

class VideoOverlayFromFile:
    """
    ComfyUI node:
      - Input: video_path (STRING, path to input video file)
      - Input: overlay_folder_path (STRING, path to folder with transparent PNG files)
      - Input: mode (STRING, animation mode for overlays)
      - Input: prefix (STRING, output filename prefix)
      - Output: output_path (STRING, path to processed video file with overlays and preserved audio)
    """

    # Cache GPU availability to avoid repeated encoder probing
    _gpu_available = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "Path to input video file"
                }),
                "overlay_folder_path": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "",
                        "placeholder": "Path to folder with PNG overlays"
                    },
                ),
                "mode": (
                    ["loop", "run_once", "run_once_and_hold", "ping_pong"],
                    {
                        "default": "loop",
                    },
                ),
                "prefix": ("STRING", {
                    "default": "video_overlay",
                    "multiline": False,
                    "placeholder": "output filename prefix"
                }),
                "use_gpu": ("BOOLEAN", {
                    "default": True,
                    "label": "Use GPU (NVENC encoder)"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "apply_overlay"
    CATEGORY = "Simple Video Effects"
    OUTPUT_NODE = True

    def __init__(self):
        self.output_dir = self._get_default_output_directory()

    def _get_default_output_directory(self) -> str:
        try:
            current_path = Path(__file__).resolve()
            comfy_root = None
            for parent in current_path.parents:
                if parent.name == "ComfyUI":
                    comfy_root = parent.parent
                    break

            if comfy_root:
                fallback_dir = comfy_root / "ComfyUI" / "output"
            else:
                fallback_dir = Path(os.getcwd()) / "output"

            os.makedirs(fallback_dir, exist_ok=True)
            return str(fallback_dir)
        except Exception:
            fallback_dir = os.getcwd()
            print(f"Using fallback output directory: {fallback_dir}")
            return fallback_dir

    def get_unique_filename(self, output_path: str) -> str:
        base, ext = os.path.splitext(output_path)
        index = 1
        unique_path = output_path

        while os.path.exists(unique_path):
            unique_path = f"{base}_{index}{ext}"
            index += 1

        return unique_path

    def _extract_frames(self, video_path: str, temp_dir: str) -> tuple:
        """Extract all frames from video and return fps and frame count"""
        try:
            probe = ffmpeg.probe(video_path)
        except ffmpeg.Error as e:
            msg = e.stderr.decode() if getattr(e, "stderr", None) else str(e)
            raise RuntimeError(f"Failed to probe video '{video_path}': {msg}")

        streams = probe.get("streams", [])
        vstreams = [s for s in streams if s.get("codec_type") == "video"]
        if not vstreams:
            raise RuntimeError(f"No video stream found in '{video_path}'.")

        v0 = vstreams[0]
        fps_str = v0.get("r_frame_rate") or v0.get("avg_frame_rate") or "30/1"
        try:
            num, den = fps_str.split("/")
            fps = float(num) / float(den) if den != "0" else 30.0
        except Exception:
            fps = 30.0

        # Extract frames
        frame_pattern = os.path.join(temp_dir, "frame_%06d.png")
        stream = ffmpeg.input(video_path).output(frame_pattern, start_number=0)
        stream.run(quiet=True)

        # Count extracted frames
        frame_files = sorted([f for f in os.listdir(temp_dir) if f.startswith("frame_") and f.endswith(".png")])
        frame_count = len(frame_files)

        if frame_count == 0:
            raise RuntimeError("No frames were extracted from the video")

        return fps, frame_count, frame_files

    def apply_overlay(self, video_path: str, overlay_folder_path: str, mode: str, prefix: str = "video_overlay", use_gpu: bool = True):
        if not video_path or not os.path.exists(video_path):
            raise RuntimeError(f"Video file not found: {video_path}")

        if not overlay_folder_path:
            raise RuntimeError("overlay_folder_path is empty.")
        overlay_folder = Path(overlay_folder_path)
        if not overlay_folder.exists() or not overlay_folder.is_dir():
            raise RuntimeError(f"Overlay folder not found: {overlay_folder}")

        # Check for overlay files
        overlay_files = sorted([f for f in os.listdir(overlay_folder) if f.lower().endswith('.png')])
        if not overlay_files:
            raise RuntimeError(f"No PNG files found in {overlay_folder}")

        num_overlays = len(overlay_files)

        # Check if overlay files already follow the expected naming pattern (000001.png, 000002.png, etc.)
        expected_pattern = all(f == f"{i:06d}.png" for i, f in enumerate(overlay_files, 1))
        use_direct_path = expected_pattern and len(overlay_files) > 0

        # Get video info
        try:
            probe = ffmpeg.probe(video_path)
        except ffmpeg.Error as e:
            msg = e.stderr.decode() if getattr(e, "stderr", None) else str(e)
            raise RuntimeError(f"Failed to probe video '{video_path}': {msg}")

        streams = probe.get("streams", [])
        vstreams = [s for s in streams if s.get("codec_type") == "video"]
        if not vstreams:
            raise RuntimeError(f"No video stream found in '{video_path}'.")

        v0 = vstreams[0]
        fps_str = v0.get("r_frame_rate") or v0.get("avg_frame_rate") or "30/1"
        try:
            num, den = fps_str.split("/")
            fps = float(num) / float(den) if den != "0" else 30.0
        except Exception:
            fps = 30.0

        # Get video duration and audio codec info
        fmt = probe.get("format", {})
        # Use video stream duration if available, otherwise format duration
        duration = float(v0.get("duration", fmt.get("duration", 0)))
        if duration <= 0:
            raise RuntimeError("Could not determine video duration")

        # Check if input audio is already AAC (to avoid unnecessary re-encoding)
        audio_codec = None
        for stream in streams:
            if stream.get("codec_type") == "audio":
                audio_codec = stream.get("codec_name")
                break

        # Use copy if already AAC, otherwise re-encode
        audio_encoding = "copy" if audio_codec == "aac" else "aac"

        # Check for GPU encoder availability with caching
        video_codec = "libx264"
        if use_gpu:
            if self._gpu_available is None:
                # Cache GPU availability check
                try:
                    encoders = subprocess.check_output(
                        ["ffmpeg", "-hide_banner", "-encoders"],
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    self._gpu_available = "h264_nvenc" in encoders
                except Exception:
                    self._gpu_available = False

            if self._gpu_available:
                video_codec = "h264_nvenc"
                print("Using GPU encoder: h264_nvenc")
            else:
                print("USE_GPU is enabled, but h264_nvenc encoder was not found. Falling back to libx264.")

        # Create output path
        prefix = prefix.strip() or "video_overlay"
        output_filename = f"{prefix}.mp4"
        output_path = os.path.join(self.output_dir, output_filename)
        output_path = self.get_unique_filename(output_path)

        # Build FFmpeg command for overlay
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                if use_direct_path:
                    # Use overlay folder directly if files are already properly numbered
                    overlay_pattern = os.path.join(overlay_folder_path, "%06d.png")
                else:
                    # Create properly numbered overlay sequence for FFmpeg
                    overlay_temp_dir = os.path.join(temp_dir, "overlays")
                    os.makedirs(overlay_temp_dir)

                    # Create symlinks with sequential numbering (000001.png, 000002.png, etc.)
                    for i, overlay_file in enumerate(overlay_files, 1):
                        src_path = os.path.join(overlay_folder_path, overlay_file)
                        dst_name = f"{i:06d}.png"
                        dst_path = os.path.join(overlay_temp_dir, dst_name)
                        try:
                            os.symlink(src_path, dst_path)
                        except OSError:
                            # Fallback to copy if symlink fails (e.g., on Windows without privileges)
                            shutil.copy2(src_path, dst_path)

                    overlay_pattern = os.path.join(overlay_temp_dir, "%06d.png")

                # Build filter based on mode
                base_filter = "[0:v]setpts=PTS-STARTPTS[base];[1:v]setpts=PTS-STARTPTS"
                overlay_options = "overlay=0:0:format=auto"

                if mode == "loop":
                    # Infinite loop of overlays
                    stream_loop = "-1"
                elif mode == "run_once":
                    # Play overlays once, then show base video without overlay
                    stream_loop = "0"
                elif mode == "run_once_and_hold":
                    # Play overlays once, then hold last frame using tpad filter
                    stream_loop = "0"
                    base_filter = base_filter.replace("[1:v]setpts=PTS-STARTPTS", "[1:v]tpad=stop_mode=clone:stop_duration=86400,setpts=PTS-STARTPTS")
                elif mode == "ping_pong":
                    # Create ping-pong by creating a forward+reverse sequence in temp dir
                    pingpong_temp_dir = os.path.join(temp_dir, "pingpong")
                    os.makedirs(pingpong_temp_dir)

                    # Create forward sequence
                    forward_files = overlay_files
                    # Create reverse sequence (excluding first to avoid duplication at the turn)
                    reverse_files = list(reversed(overlay_files))[1:] if len(overlay_files) > 1 else []

                    # Combine forward + reverse for ping-pong
                    pingpong_sequence = forward_files + reverse_files

                    # Create sequentially numbered files for ping-pong
                    for i, overlay_file in enumerate(pingpong_sequence, 1):
                        src_path = os.path.join(overlay_folder_path, overlay_file)
                        dst_name = f"{i:06d}.png"
                        dst_path = os.path.join(pingpong_temp_dir, dst_name)
                        try:
                            os.symlink(src_path, dst_path)
                        except OSError:
                            shutil.copy2(src_path, dst_path)

                    # Update overlay pattern to use ping-pong sequence
                    overlay_pattern = os.path.join(pingpong_temp_dir, "%06d.png")
                    stream_loop = "-1"  # Loop the ping-pong sequence
                else:
                    raise RuntimeError(f"Unknown mode: {mode}")

                filter_complex = f"{base_filter}[ov];[base][ov]{overlay_options}[v]"

                # Build the FFmpeg command
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel", "error",
                    "-i", video_path,
                    "-framerate", str(fps),
                    "-stream_loop", stream_loop,
                    "-i", overlay_pattern,
                    "-filter_complex",
                    filter_complex,
                    "-map", "[v]",
                    "-map", "0:a?",
                    "-c:v", video_codec,
                    "-c:a", audio_encoding,
                    "-pix_fmt", "yuv420p",
                    "-t", str(duration),  # Match input duration
                    output_path
                ]

                print(f"Applying overlay with mode '{mode}' using FFmpeg...")
                pbar = comfy.utils.ProgressBar(1)  # Simple progress for the FFmpeg operation

                try:
                    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                    pbar.update(1)
                except subprocess.CalledProcessError as e:
                    pbar.update(1)
                    error_msg = e.stderr.strip()
                    if error_msg:
                        raise RuntimeError(f"FFmpeg overlay failed: {error_msg}")
                    else:
                        raise RuntimeError(f"FFmpeg overlay failed with return code {e.returncode}")

                print(f"Video overlay applied. Output: {output_path}")
                return (output_path,)

            except Exception as e:
                raise RuntimeError(f"Error processing video overlay: {str(e)}")

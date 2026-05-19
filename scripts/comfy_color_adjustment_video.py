import os
import json
import shutil
import subprocess
from pathlib import Path

try:
    import folder_paths
    OUTPUT_DIR = folder_paths.get_output_directory()
except ImportError:
    OUTPUT_DIR = "output"


class ColorAdjustmentVideoNode:
    """
    Adjusts brightness, contrast, and saturation on video files using FFmpeg filters.
    Parameters use 0-100 scale where 100 = no change.
    Preserves original audio and includes option to delete original file.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": ""}),
                "brightness": ("INT", {"default": 100, "min": 0, "max": 200, "step": 1}),
                "contrast": ("INT", {"default": 100, "min": 0, "max": 200, "step": 1}),
                "saturation": ("INT", {"default": 100, "min": 0, "max": 200, "step": 1}),
                "delete_original": ("BOOLEAN", {"default": False}),
                "prefix": ("STRING", {"default": "color_adjusted"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("output_path", "info")
    FUNCTION = "run"
    CATEGORY = "Simple Video Effects/Utility & Special Effects"

    def run(
        self,
        video_path: str,
        brightness: int,
        contrast: int,
        saturation: int,
        delete_original: bool,
        prefix: str,
    ):
        # Locate ffmpeg and ffprobe
        FFMPEG = shutil.which("ffmpeg")
        FFPROBE = shutil.which("ffprobe")

        if not FFMPEG or not FFPROBE:
            raise RuntimeError("ffmpeg and ffprobe not found on PATH.")

        # Validate input video exists
        video_path = Path(video_path)
        if not video_path.exists():
            raise ValueError(f"Video file not found: {video_path}")

        # Extract video metadata using ffprobe
        try:
            cmd = [
                FFPROBE,
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(video_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            probe_data = json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to probe video: {e}")

        # Extract video stream info
        video_stream = None
        for stream in probe_data.get("streams", []):
            if stream.get("codec_type") == "video":
                video_stream = stream
                break

        if not video_stream:
            raise ValueError(f"No video stream found in {video_path}")

        # Get FPS from r_frame_rate (format: "30/1" or "24000/1001")
        fps_str = video_stream.get("r_frame_rate", "30/1")
        try:
            num, den = map(int, fps_str.split("/"))
            fps = num / den
        except (ValueError, ZeroDivisionError):
            fps = 30.0

        # Get duration
        duration = float(probe_data.get("format", {}).get("duration", 0))

        # Check for audio stream
        has_audio = any(s.get("codec_type") == "audio" for s in probe_data.get("streams", []))

        # Convert parameters from 0-100 scale to filter values
        # brightness: 100 = 0.0 (no change), range -1.0 to 1.0
        brightness_val = (brightness - 100) / 100.0

        # contrast: 100 = 1.0 (no change), range 0.0 to 2.0
        contrast_val = contrast / 100.0

        # saturation: 100 = 1.0 (no change), range 0.0 to 2.0
        saturation_val = saturation / 100.0

        # Build FFmpeg filter chain
        # eq filter for brightness/contrast, hue filter for saturation
        vf = f"eq=brightness={brightness_val}:contrast={contrast_val},hue=s={saturation_val}"

        # Generate unique output filename
        output_path = self._get_unique_output_path(prefix, output_dir=OUTPUT_DIR)

        # Build FFmpeg command
        cmd = [
            FFMPEG,
            "-y",  # overwrite output file
            "-i",
            str(video_path),
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
        ]

        # Preserve audio
        if has_audio:
            cmd.extend(["-c:a", "copy", "-map", "0:v:0", "-map", "0:a?"])

        cmd.append(str(output_path))

        # Execute FFmpeg
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg failed: {e.stderr.decode() if e.stderr else str(e)}")

        # Delete original if requested
        if delete_original:
            try:
                video_path.unlink()
            except Exception as e:
                print(f"Warning: Failed to delete original file: {e}")

        info = (
            f"ColorAdjustment: brightness={brightness} (val: {brightness_val:.2f}), "
            f"contrast={contrast} (val: {contrast_val:.2f}), "
            f"saturation={saturation} (val: {saturation_val:.2f}), "
            f"audio={'preserved' if has_audio else 'none'}"
        )

        return (str(output_path), info)

    def _get_unique_output_path(self, prefix: str, output_dir: str = "output") -> Path:
        """Generate unique output filename to avoid overwrites."""
        os.makedirs(output_dir, exist_ok=True)
        counter = 1
        while True:
            filename = f"{prefix}_{counter:03d}.mp4"
            output_path = Path(output_dir) / filename
            if not output_path.exists():
                return output_path
            counter += 1

import os
import subprocess
from pathlib import Path
import tempfile
import ffmpeg

class VideoLoopExtenderNode:
    """
    A ComfyUI-compatible node that:
    - Takes a video file path and extends it by duplicating and merging it N times.
    - Keeps audio if present.
    - Optionally deletes the original video file after processing.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Path to input video file"
                }),
                "extend_factor": ("FLOAT", {
                    "default": 1.0,
                    "min": 1.0,
                    "max": 100.0,
                    "step": 1.0,
                    "round": 0.0,
                    "label": "Number of times to duplicate (loop)"
                }),
            },
            "optional": {
                "delete_original": ("BOOLEAN", {
                    "default": False,
                    "label": "Delete original video file after processing"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "extend_video_loop"
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

    def _get_video_duration(self, video_path: str) -> float:
        try:
            probe = ffmpeg.probe(video_path)
        except ffmpeg.Error as e:
            msg = e.stderr.decode() if getattr(e, "stderr", None) else str(e)
            raise RuntimeError(f"Failed to probe video '{video_path}': {msg}")
        except Exception as e:
            raise RuntimeError(f"Failed to probe video '{video_path}': {str(e)}")

        fmt = probe.get("format", {})
        if "duration" not in fmt:
            raise RuntimeError(
                f"Could not determine duration for video '{video_path}' "
                f"(no 'duration' in ffprobe output)."
            )

        try:
            return float(fmt["duration"])
        except Exception as e:
            raise RuntimeError(
                f"Invalid duration value for video '{video_path}': "
                f"{fmt['duration']} ({e})"
            )

    def get_unique_filename(self, output_path: str) -> str:
        base, ext = os.path.splitext(output_path)
        index = 1
        unique_path = output_path

        while os.path.exists(unique_path):
            unique_path = f"{base}_{index}{ext}"
            index += 1

        return unique_path

    def extend_video_loop(
        self,
        video_path: str,
        extend_factor: float = 1.0,
        delete_original: bool = False,
    ) -> tuple:
        if not os.path.exists(video_path):
            raise ValueError(f"Video file does not exist: {video_path}")

        if not video_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
            raise ValueError(f"Unsupported video format for '{video_path}'. Supported: mp4, avi, mov, mkv, webm")

        extend_factor = max(1, int(extend_factor))  # Ensure at least 1, and integer for simplicity

        # Get original filename for output
        video_name = Path(video_path).stem
        output_filename = f"{video_name}_extended_x{extend_factor}.mp4"
        output_path = os.path.join(self.output_dir, output_filename)
        output_path = self.get_unique_filename(output_path)

        # Create list of inputs (same video repeated)
        inputs = [ffmpeg.input(video_path) for _ in range(extend_factor)]

        # Concatenate videos
        if len(inputs) == 1:
            stream = inputs[0]
        else:
            stream = ffmpeg.concat(*inputs, v=1, a=1)  # v=1 for video, a=1 for audio

        # Output
        stream = ffmpeg.output(stream, output_path, vcodec='libx264', acodec='aac')
        stream = stream.overwrite_output()

        try:
            print(f"Extending video loop: duplicating '{video_path}' {extend_factor} times...")
            ffmpeg.run(stream, quiet=True)
            print("Video loop extension completed.")
        except ffmpeg.Error as e:
            msg = e.stderr.decode() if getattr(e, "stderr", None) else str(e)
            raise RuntimeError(f"FFmpeg error during video extension: {msg}")

        # Delete original if toggled
        if delete_original:
            try:
                os.remove(video_path)
                print(f"Deleted original video file: {video_path}")
            except Exception as e:
                print(f"Warning: Failed to delete original file '{video_path}': {e}")

        return (output_path,)

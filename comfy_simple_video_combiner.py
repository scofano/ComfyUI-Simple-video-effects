import os
import subprocess
from pathlib import Path
import tempfile
import ffmpeg

class ComfySimpleVideoCombiner:
    """
    A simple ComfyUI-compatible node that:
    - Scans a directory for video files matching a pattern (e.g. *.mp4)
    - Concatenates them in sorted order
    - Optionally uses GPU-accelerated encoding (NVENC) when available
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Path to video directory"
                }),
                "output_filename": ("STRING", {
                    "default": "combined_output.mp4",
                    "multiline": False,
                    "placeholder": "output.mp4"
                }),
                "file_pattern": ("STRING", {
                    "default": "*.mp4",
                    "multiline": False,
                    "placeholder": "*.mp4"
                }),
                "use_gpu": ("BOOLEAN", {
                    "default": True,
                    "label": "Use GPU (NVENC encoder)"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "combine_videos"
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

    def combine_videos(
        self,
        directory_path: str,
        output_filename: str,
        file_pattern: str,
        use_gpu: bool = True,
    ) -> tuple:

        directory = Path(directory_path).expanduser().resolve()
        if not directory.is_dir():
            raise ValueError(f"Directory does not exist: {directory}")

        video_files = sorted(directory.glob(file_pattern))
        if not video_files:
            raise ValueError(
                f"No video files found in directory '{directory}' "
                f"matching pattern '{file_pattern}'"
            )

        output_filename = output_filename.strip() or "combined_output.mp4"
        if not output_filename.lower().endswith(".mp4"):
            output_filename += ".mp4"

        output_path = os.path.join(self.output_dir, output_filename)
        output_path = self.get_unique_filename(output_path)

        try:
            # Create inputs for each video file
            inputs = [ffmpeg.input(str(video_file)) for video_file in video_files]

            v_streams = [i.video for i in inputs]
            a_streams = [i.audio for i in inputs]

            # Normalize audio (to avoid breaking due to 24000/mono vs 44100/stereo)
            a_streams = [
                a.filter("aresample", 44100)
                 .filter("aformat", sample_fmts="fltp", channel_layouts="stereo")
                for a in a_streams
            ]

            # IMPORTANT: interleave v/a per clip
            streams = []
            for v, a in zip(v_streams, a_streams):
                streams.extend([v, a])

            out_nodes = ffmpeg.concat(*streams, v=1, a=1).node
            v_out, a_out = out_nodes[0], out_nodes[1]

            if use_gpu:
                out = ffmpeg.output(v_out, a_out, output_path, vcodec="h264_nvenc", acodec="aac")
            else:
                out = ffmpeg.output(v_out, a_out, output_path, vcodec="libx264", acodec="aac")

            out = out.overwrite_output()
            out.run(quiet=True)

        except ffmpeg.Error as e:
            if getattr(e, "stderr", None) is not None:
                raise RuntimeError(f"FFmpeg error: {e.stderr.decode()}")
            else:
                raise RuntimeError(f"FFmpeg error: {str(e)}")

        return (output_path,)

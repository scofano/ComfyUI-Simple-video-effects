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

        # Create concat file list
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            for video_file in video_files:
                path_str = str(video_file.absolute()).replace("'", r"'\''")
                f.write(f"file '{path_str}'\n")
            temp_list_path = f.name

        try:
            # Simple concat
            stream = ffmpeg.input(temp_list_path, f="concat", safe=0)

            if use_gpu:
                stream = ffmpeg.output(
                    stream, output_path, vcodec="h264_nvenc"
                )
            else:
                stream = ffmpeg.output(stream, output_path)

            stream = stream.overwrite_output()
            print("Rendering video...")
            stream.run(quiet=True)
            print("Rendering finished.")

        except ffmpeg.Error as e:
            if getattr(e, "stderr", None) is not None:
                raise RuntimeError(f"FFmpeg error: {e.stderr.decode()}")
            else:
                raise RuntimeError(f"FFmpeg error: {str(e)}")

        finally:
            if temp_list_path and os.path.exists(temp_list_path):
                try:
                    os.remove(temp_list_path)
                except Exception:
                    pass

        return (output_path,)

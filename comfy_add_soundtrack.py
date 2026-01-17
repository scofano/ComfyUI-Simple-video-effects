import os
import ffmpeg
from pathlib import Path

class ComfyAddSoundtrack:
    """
    A ComfyUI-compatible node that adds a soundtrack to a video without changing the original audio.
    Applies volume adjustment only to the soundtrack and mixes it with the video's audio.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Full path to video file"
                }),
                "audio_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Full path to audio file"
                }),
                "volume_db": ("FLOAT", {
                    "default": -20.0,
                    "min": -100.0,
                    "step": 1.0,
                    "label": "Soundtrack volume (dB)"
                }),
                "delete_original": ("BOOLEAN", {
                    "default": False,
                    "label": "Delete original video after processing"
                }),
            },
            "optional": {
                "output_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Output video path (optional)"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_video_path",)
    FUNCTION = "add_soundtrack"
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

    def add_soundtrack(
        self,
        video_path: str,
        audio_path: str,
        volume_db: float = -20.0,
        delete_original: bool = False,
        output_path: str = "",
    ) -> tuple:
        video_path = video_path.strip()
        audio_path = audio_path.strip()

        if not os.path.isfile(video_path):
            raise ValueError(f"Video file does not exist: {video_path}")

        if not os.path.isfile(audio_path):
            raise ValueError(f"Audio file does not exist: {audio_path}")

        # Get video duration
        video_duration = self._get_video_duration(video_path)

        # Set output path
        if not output_path.strip():
            base_name = os.path.splitext(os.path.basename(video_path))[0] + "_with_soundtrack.mp4"
            output_path = os.path.join(self.output_dir, base_name)

        if not output_path.lower().endswith(".mp4"):
            output_path += ".mp4"

        output_path = self.get_unique_filename(output_path)

        try:
            print("Starting soundtrack addition processing...")
            # Load inputs
            video = ffmpeg.input(video_path)
            audio = ffmpeg.input(audio_path)

            # Trim audio to video duration and apply volume
            audio = audio.filter('atrim', 0, video_duration).filter('volume', f'{volume_db}dB')

            # Mix audios
            mixed_audio = ffmpeg.filter([video.audio, audio], 'amix', inputs=2, duration='longest', dropout_transition=0)

            # Output with mixed audio
            out = ffmpeg.output(video.video, mixed_audio, output_path, vcodec="libx264", acodec="aac")
            out = out.overwrite_output()
            out.run(quiet=True)
            print("Soundtrack addition processing complete.")

        except ffmpeg.Error as e:
            if getattr(e, "stderr", None) is not None:
                raise RuntimeError(f"FFmpeg error: {e.stderr.decode()}")
            else:
                raise RuntimeError(f"FFmpeg error: {str(e)}")

        # Delete original video if requested
        if delete_original:
            try:
                os.remove(video_path)
                print(f"Deleted original video: {video_path}")
            except Exception as e:
                print(f"Warning: Could not delete original video '{video_path}': {e}")

        return (output_path,)

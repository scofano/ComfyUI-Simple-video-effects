# comfy_merge_video_audio.py

import os
import subprocess

class MergeVideoAudioNode:
    """
    Merge a video file and an audio file into a new video.
    - Receives full paths for video and audio.
    - Receives a filename (text) for the output.
    - If the filename already exists, appends _001, _002, etc.
    - Has a trim_to_audio toggle (default ON), which trims video to audio length.
    - Outputs the final full file path as a STRING (output node).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "",
                        "placeholder": "Full path to input video file",
                    },
                ),
                "audio_path": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "",
                        "placeholder": "Full path to input audio file",
                    },
                ),
                "output_filename": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "output.mp4",
                        "placeholder": "filename or full path for output (e.g. final.mp4)",
                    },
                ),
                "trim_to_audio": (
                    "BOOLEAN",
                    {
                        "default": True,
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "merge"
    OUTPUT_NODE = True
    CATEGORY = "Simple Video Effects"

    def _resolve_output_path(self, video_path: str, output_filename: str) -> str:
        """
        Determine the full output path with collision handling (_001, _002, ...).
        - If `output_filename` is an absolute path, use it as the base.
        - If it contains path separators, treat it as relative or absolute and normalize.
        - Otherwise, save in the same directory as the video file.
        """
        # If user gave an absolute path, keep its directory.
        if os.path.isabs(output_filename):
            base_output_path = output_filename
        else:
            # If contains a path separator, join and normalize
            if os.sep in output_filename or (os.altsep and os.altsep in output_filename):
                base_output_path = os.path.abspath(output_filename)
            else:
                # Just a name: put it next to the video
                video_dir = os.path.dirname(os.path.abspath(video_path)) or os.getcwd()
                base_output_path = os.path.join(video_dir, output_filename)

        directory = os.path.dirname(base_output_path) or os.getcwd()
        filename = os.path.basename(base_output_path)

        base_name, ext = os.path.splitext(filename)
        if ext == "":
            # default to .mp4 if no extension given
            ext = ".mp4"

        candidate = os.path.join(directory, base_name + ext)

        # If it exists, append _001, _002, ...
        counter = 1
        while os.path.exists(candidate):
            candidate = os.path.join(
                directory,
                f"{base_name}_{counter:03d}{ext}",
            )
            counter += 1

        # Ensure directory exists
        os.makedirs(directory, exist_ok=True)

        return candidate

    def merge(self, video_path: str, audio_path: str, output_filename: str, trim_to_audio: bool):
        video_path = os.path.abspath(video_path)
        audio_path = os.path.abspath(audio_path)

        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        output_path = self._resolve_output_path(video_path, output_filename)

        # Build ffmpeg command
        cmd = [
            "ffmpeg",
            "-y",              # overwrite intermediate, but our name is always unique
            "-i", video_path,
            "-i", audio_path,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",    # keep original video
            "-c:a", "aac",     # encode audio as AAC (widely supported)
            "-b:a", "192k",
        ]

        # Trim video to audio length if selected
        if trim_to_audio:
            cmd.append("-shortest")

        cmd.append(output_path)

        try:
            # Run ffmpeg and capture errors if any
            completed = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to run ffmpeg: {e}")

        if completed.returncode != 0:
            # Include some of the ffmpeg error output for debugging
            raise RuntimeError(
                f"ffmpeg merge failed with code {completed.returncode}\n"
                f"stderr:\n{completed.stderr}"
            )

        return (output_path,)
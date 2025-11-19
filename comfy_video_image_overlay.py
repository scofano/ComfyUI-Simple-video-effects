import os
import subprocess
from typing import Tuple

from tqdm import tqdm

import folder_paths


class VideoImageOverlay:
    """
    ComfyUI custom node: Video Image Overlay (No Resize Version)

    - Applies a PNG image (with optional alpha) as an overlay over a full video
      using ffmpeg.
    - Removes all functionality related to resizing the overlay image.
    - Saves the result as an MP4 in the ComfyUI output directory.
    - Returns the full path to the generated video as a STRING.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "overlay_image_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "Full path to overlay image (PNG with alpha)",
                    },
                ),
                "video_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "Full path to input video file",
                    },
                ),
                "suffix": (
                    "STRING",
                    {
                        "default": "_overlay",
                        "multiline": False,
                        "placeholder": "Suffix for output video filename (before .mp4)",
                    },
                ),
            }
        }

    # Returns one STRING: the full path to the output video
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_video_path",)

    # This function gets called by ComfyUI
    FUNCTION = "apply_overlay"

    # Category in the node menu
    CATEGORY = "video"

    # Mark as an OUTPUT node so it can run without needing any input
    OUTPUT_NODE = True

    def _validate_paths(self, overlay_image_path: str, video_path: str) -> None:
        if not overlay_image_path:
            raise ValueError("overlay_image_path is empty.")
        if not video_path:
            raise ValueError("video_path is empty.")

        if not os.path.isfile(overlay_image_path):
            raise FileNotFoundError(f"Overlay image not found: {overlay_image_path}")
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

    def _build_output_path(self, video_path: str, suffix: str) -> str:
        """Build the output .mp4 path in ComfyUI's output directory, avoiding overwrites."""
        output_dir = folder_paths.get_output_directory()

        base_name = os.path.basename(video_path)
        stem, _ext = os.path.splitext(base_name)

        # Base name + suffix, e.g. myvideo_overlay
        suffix = suffix or ""
        stem_with_suffix = f"{stem}{suffix}"

        # First candidate: myvideo_overlay.mp4
        output_path = os.path.join(output_dir, f"{stem_with_suffix}.mp4")

        # If it doesn't exist yet, use it directly
        if not os.path.exists(output_path):
            return output_path

        # If it exists, append _0001, _0002, ...
        counter = 1
        while True:
            # e.g. myvideo_overlay_0001.mp4, myvideo_overlay_0002.mp4, ...
            candidate = os.path.join(output_dir, f"{stem_with_suffix}_{counter:04d}.mp4")
            if not os.path.exists(candidate):
                return candidate
            counter += 1

    def _build_ffmpeg_command(
        self,
        overlay_image_path: str,
        video_path: str,
        output_path: str,
    ) -> list:

        cmd = [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-i", overlay_image_path,
        ]

        # No resizing: direct overlay
        filter_complex = "[0:v][1:v]overlay=0:0:format=auto[vout]"

        cmd.extend([
            "-filter_complex", filter_complex,
            "-map", "[vout]",
            "-map", "0:a?",  # optional audio
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            output_path,
        ])

        return cmd

    def _run_ffmpeg_with_progress(self, cmd: list) -> None:
        with tqdm(total=1, desc="Applying video overlay (ffmpeg)", unit="step") as pbar:
            try:
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )
            finally:
                pbar.update(1)

        if result.returncode != 0:
            err_snippet = (result.stderr or "").splitlines()[-10:]
            raise RuntimeError(
                "ffmpeg failed with code {}.\nCommand: {}\nStderr:\n{}".format(
                    result.returncode,
                    " ".join(cmd),
                    "\n".join(err_snippet),
                )
            )

    def apply_overlay(
        self,
        overlay_image_path: str,
        video_path: str,
        suffix: str,
    ) -> Tuple[str]:

        self._validate_paths(overlay_image_path, video_path)

        output_path = self._build_output_path(video_path, suffix)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        cmd = self._build_ffmpeg_command(
            overlay_image_path=overlay_image_path,
            video_path=video_path,
            output_path=output_path,
        )

        try:
            self._run_ffmpeg_with_progress(cmd)
        except FileNotFoundError:
            raise RuntimeError(
                "ffmpeg executable not found. Make sure ffmpeg is installed and available in PATH."
            )

        return (output_path,)

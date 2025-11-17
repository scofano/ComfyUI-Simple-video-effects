import os
import subprocess
from pathlib import Path
import tempfile
import random
import ffmpeg
from concurrent.futures import ThreadPoolExecutor

# Import numpy and soundfile here, as they are needed for audio processing
import numpy as np
import soundfile as sf

# Optional progress bar
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


class ComfyVideoCombiner:
    """
    A ComfyUI-compatible node that:
    - Scans a directory for video files matching a pattern (e.g. *.mp4)
    - Sorts them or randomizes order
    - Optionally adds crossfade transitions between them
    - Optionally fades in from a color and/or fades out to a color
    - Optionally overlays a music track (VideoHelperSuite audio format)
    """

    # Removed @classmethod OUTPUT_NODE(cls): return True

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
                "transition": (["none", "fade"], {"default": "none"}),
                "transition_duration": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1,
                    "round": 0.1,
                }),
            },
            "optional": {
                "sort_files": ("BOOLEAN", {
                    "default": True,
                    "label": "Sort files alphabetically"
                }),
                "random_order": ("BOOLEAN", {
                    "default": True,
                    "label": "Randomize order (no repeats, overrides sort)"
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2**31 - 1,
                    "step": 1,
                    "label": "Random seed (-1 = random)"
                }),
                "music_track": ("AUDIO",),  # VideoHelperSuite audio format

                "trim_to_audio": ("BOOLEAN", {
                    "default": True,
                    "label": "Trim video to audio length"
                }),

                # Fade in/out controls
                "fade_in_enabled": ("BOOLEAN", {
                    "default": False,
                    "label": "Enable fade in from color"
                }),
                "fade_in_color": ("STRING", {
                    "default": "#000000",
                    "multiline": False,
                    "placeholder": "#000000",
                    "label": "Fade in color (hex)"
                }),
                "fade_in_duration": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "round": 0.1,
                    "label": "Fade in duration (s)"
                }),
                "fade_out_enabled": ("BOOLEAN", {
                    "default": False,
                    "label": "Enable fade out to color"
                }),
                "fade_out_color": ("STRING", {
                    "default": "#000000",
                    "multiline": False,
                    "placeholder": "#000000",
                    "label": "Fade out color (hex)"
                }),
                "fade_out_duration": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "round": 0.1,
                    "label": "Fade out duration (s)"
                }),
            }
        }

    # Restored output path return types
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "combine_videos"
    CATEGORY = "Video"

    def __init__(self):
        # Use a default output directory (ComfyUI's default output, if available).
        self.output_dir = self._get_default_output_directory()

    def _get_default_output_directory(self) -> str:
        """
        Try to get ComfyUI's default output directory (e.g. 'output') from environment
        variables or fallback to current directory.
        """
        # Adjust this logic as needed in your actual ComfyUI environment.
        comfy_env_output = os.environ.get("COMFYUI_OUTPUT_DIR")
        if comfy_env_output and os.path.isdir(comfy_env_output):
            return comfy_env_output

        try:
            # Try a typical ComfyUI-style 'output' directory
            fallback_dir = os.path.join(os.getcwd(), "output")
            os.makedirs(fallback_dir, exist_ok=True)
            return fallback_dir
        except Exception:
            # Last resort: current directory
            fallback_dir = os.getcwd()
            print(f"Using fallback output directory: {fallback_dir}")
            return fallback_dir

    def _get_video_duration(self, video_path: str) -> float:
        """
        Get duration of video file using ffmpeg, with robust error handling.
        """
        try:
            # no stderr= here – let ffmpeg-python handle the subprocess
            probe = ffmpeg.probe(video_path)
        except ffmpeg.Error as e:
            msg = e.stderr.decode() if getattr(e, "stderr", None) else str(e)
            raise RuntimeError(f"Failed to probe video '{video_path}': {msg}")
        except Exception as e:
            raise RuntimeError(f"Failed to probe video '{video_path}': {str(e)}")

        fmt = probe.get("format", {})
        if "duration" not in fmt:
            raise RuntimeError(f"Could not determine duration for video '{video_path}' (no 'duration' in ffprobe output).")

        try:
            return float(fmt["duration"])
        except Exception as e:
            raise RuntimeError(f"Invalid duration value for video '{video_path}': {fmt['duration']} ({e})")

    def _get_video_resolution_and_fps(self, video_path: str) -> tuple[int, int, float]:
        """
        Get (width, height, fps_float) of the first video stream.
        """
        try:
            probe = ffmpeg.probe(video_path)
        except ffmpeg.Error as e:
            msg = e.stderr.decode() if getattr(e, "stderr", None) else str(e)
            raise RuntimeError(f"Failed to probe video '{video_path}': {msg}")
        except Exception as e:
            raise RuntimeError(f"Failed to probe video '{video_path}': {str(e)}")

        # Find first video stream
        streams = probe.get("streams", [])
        vstreams = [s for s in streams if s.get("codec_type") == "video"]
        if not vstreams:
            raise RuntimeError(f"No video stream found in '{video_path}'.")

        v0 = vstreams[0]
        width = int(v0.get("width", 0))
        height = int(v0.get("height", 0))

        # Derive fps from r_frame_rate or avg_frame_rate
        fps_str = v0.get("r_frame_rate") or v0.get("avg_frame_rate") or "0/0"
        try:
            num, den = fps_str.split("/")
            num, den = float(num), float(den)
            fps = num / den if den != 0 else 0.0
        except Exception:
            fps = 0.0

        if width <= 0 or height <= 0:
            raise RuntimeError(f"Invalid resolution for '{video_path}': {width}x{height}")
        if fps <= 0:
            # fallback: 30
            fps = 30.0

        return width, height, fps

    def _compute_durations_parallel(self, video_files):
        """
        Compute durations of multiple video files in parallel, returning list of floats.
        """
        durations = [0.0] * len(video_files)
        exceptions = []

        def worker(idx, path):
            nonlocal durations, exceptions
            try:
                durations[idx] = self._get_video_duration(str(path))
            except Exception as e:
                exceptions.append((path, e))

        with ThreadPoolExecutor(max_workers=min(8, len(video_files))) as executor:
            futures = []
            for idx, vf in enumerate(video_files):
                futures.append(executor.submit(worker, idx, vf))

            for f in futures:
                f.result()

        if exceptions:
            # Report the first error (or aggregate if desired)
            path, e = exceptions[0]
            raise RuntimeError(f"Error computing duration of '{path}': {e}")

        return durations

    def _process_vhs_audio(self, audio_dict):
        """
        Process VideoHelperSuite-style audio input.

        This function converts the audio tensor to a usable temporary WAV file,
        handling the 3D shape common in VFS/ComfyUI audio output.
        """

        if not isinstance(audio_dict, dict):
            raise ValueError("music_track must be a dict in VideoHelperSuite format.")

        path = audio_dict.get("path") or audio_dict.get("output_path")
        if path and os.path.exists(path):
            # Just use the existing audio file.
            return path, None

        # If no ready-made path is given, fall back to waveform-based writing if present.
        waveform = audio_dict.get("waveform")
        sample_rate = audio_dict.get("sample_rate")

        if waveform is None or sample_rate is None:
            raise ValueError("VideoHelperSuite audio must include either 'path' or ('waveform' and 'sample_rate').")

        # 1. Convert to NumPy array
        if not isinstance(waveform, np.ndarray):
            try:
                # Attempt to convert PyTorch/TensorFlow tensor to numpy
                waveform = waveform.cpu().numpy() if hasattr(waveform, 'cpu') else np.array(waveform)
            except Exception as e:
                print(f"Warning: Could not convert waveform to numpy for processing. Error: {e}")

        # 2. Reshape/Squeeze: Remove any leading 'batch' dimension of size 1 (e.g., (1, 2, N) -> (2, N)).
        if hasattr(waveform, 'ndim') and waveform.ndim > 2:
            original_shape = waveform.shape
            if hasattr(waveform, 'squeeze'):
                 waveform = waveform.squeeze()
            elif original_shape[0] == 1:
                 # Fallback for manual removal of batch dim
                 waveform = waveform.reshape(original_shape[1:])
            print(f"Squeezed 3D waveform from {original_shape} to {waveform.shape}")

        # 3. Transpose: Convert from (channels, samples) to (samples, channels) 
        # as expected by soundfile for stereo audio.
        if waveform.ndim == 2 and waveform.shape[0] < waveform.shape[1]:
             waveform = waveform.T
             print(f"Transposed 2D waveform to shape: {waveform.shape} (samples, channels)")
        
        # Ensure sample rate is an integer for soundfile
        sample_rate = int(sample_rate)

        # Write to temporary WAV file
        temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_audio.close()

        sf.write(temp_audio.name, waveform, sample_rate)

        return temp_audio.name, temp_audio

    def get_unique_filename(self, output_path: str) -> str:
        """
        Ensure the final output path does not overwrite existing files by appending an index if needed.
        """
        base, ext = os.path.splitext(output_path)
        index = 1
        unique_path = output_path

        while os.path.exists(unique_path):
            unique_path = f"{base}_{index}{ext}"
            index += 1

        return unique_path

    def _sanitize_hex_color(self, color: str) -> str:
        """
        Ensure color is a valid hex string in the form "#RRGGBB".
        Pads incomplete hex values with leading zeros.
        """
        color = color.strip().lstrip('#')
        if not color:
            return "#000000"
        
        # Ensure it contains only valid hex characters and doesn't exceed 6 digits
        color = ''.join(c for c in color if c.lower() in '0123456789abcdef')
        if len(color) > 6:
            color = color[:6]
        
        # Pad with leading zeros to ensure 6 characters (e.g., '1' -> '000001')
        color = color.zfill(6)
        
        return "#" + color

    def combine_videos(
        self,
        directory_path: str,
        output_filename: str,
        file_pattern: str,
        transition: str = "none",
        transition_duration: float = 0.5,
        sort_files: bool = True,
        random_order: bool = True,
        seed: int = -1,
        music_track: dict = None,
        trim_to_audio: bool = True,
        fade_in_enabled: bool = False,
        fade_in_color: str = "#000000",
        fade_in_duration: float = 1.0,
        fade_out_enabled: bool = False,
        fade_out_color: str = "#000000",
        fade_out_duration: float = 1.0,
    ) -> tuple:
        """
        Combine multiple video files from a directory into a single video file
        with optional transitions, fade in/out from/to color, and optional music.
        """

        directory = Path(directory_path).expanduser().resolve()
        if not directory.is_dir():
            raise ValueError(f"Directory does not exist: {directory}")

        # Gather matching video files
        video_files = sorted(directory.glob(file_pattern))
        if not video_files:
            raise ValueError(f"No video files found in directory '{directory}' matching pattern '{file_pattern}'")

        # Optional randomization (no repeats)
        if random_order:
            if seed != -1:
                random.seed(seed)
            random.shuffle(video_files)
        elif sort_files:
            video_files.sort()

        # Process VHS audio format
        audio_path = None
        temp_audio = None
        audio_duration = None

        if music_track is not None:
            audio_path, temp_audio = self._process_vhs_audio(music_track)
            if audio_path and trim_to_audio:
                try:
                    # Works for audio as well; just uses ffprobe format.duration
                    audio_duration = self._get_video_duration(audio_path)
                except Exception as e:
                    print(f"Warning: could not probe audio duration: {e}")
                    audio_duration = None

        # Set output path with uniqueness
        output_filename = output_filename.strip() or "combined_output.mp4"
        if not output_filename.lower().endswith(".mp4"):
            output_filename += ".mp4"

        output_path = os.path.join(self.output_dir, output_filename)
        output_path = self.get_unique_filename(output_path)

        # Sanitize colors
        fade_in_color = self._sanitize_hex_color(fade_in_color)
        fade_out_color = self._sanitize_hex_color(fade_out_color)

        # Decide whether we need advanced filter-graph:
        # - If transition == "fade" and we have at least 2 videos, OR
        # - We have fade in/out enabled
        need_filter_graph = (
            (transition == "fade" and len(video_files) >= 2) or
            fade_in_enabled or
            fade_out_enabled
        )

        try:
            if not need_filter_graph:
                # Basic concatenation without transitions or fade in/out
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                    for video_file in video_files:
                        # Escape single quotes in path for ffmpeg concat file
                        path_str = str(video_file.absolute()).replace("'", r"'\''")
                        f.write(f"file '{path_str}'\n")
                    temp_list_path = f.name

                stream = ffmpeg.input(temp_list_path, f='concat', safe=0)

                if audio_path:
                    audio_stream = ffmpeg.input(audio_path)
                    output_args = {
                        'acodec': 'aac',
                        'vcodec': 'copy',
                    }
                    if trim_to_audio:
                        # -shortest → cut to the *shorter* of audio/video (usually the audio)
                        output_args['shortest'] = None

                    stream = ffmpeg.output(
                        stream,
                        audio_stream,
                        output_path,
                        **output_args,
                    )
                else:
                    stream = ffmpeg.output(stream, output_path, c='copy')

            else:
                # Build a filter-graph with optional crossfades and optional fade in/out color clips
                # Step 1: compute durations for all real video files (parallel)
                durations = self._compute_durations_parallel(video_files)

                # Clamp/check transition_duration vs clip length for real videos
                if transition == "fade" and len(video_files) >= 2:
                    min_duration = min(durations)
                    if transition_duration >= min_duration:
                        # Clamp rather than outright fail
                        clamped = max(0.1, min_duration - 0.1)
                        print(
                            f"Requested transition_duration ({transition_duration:.3f}s) "
                            f"is >= shortest clip ({min_duration:.3f}s). "
                            f"Clamping to {clamped:.3f}s."
                        )
                        transition_duration = clamped

                # Get base resolution AND base fps from first video
                width, height, base_fps = self._get_video_resolution_and_fps(str(video_files[0]))
                print(f"Using base resolution {width}x{height} at {base_fps:.3f} fps for all clips/color fades.")

                # Build list of all streams and durations (including optional color clips)
                all_streams = []
                all_durations = []

                # Optional fade in color clip (normalized to base_fps)
                if fade_in_enabled:
                    # Sanitize is already called outside this block
                    color_in = ffmpeg.input(
                        f"color=c={fade_in_color}:s={width}x{height}:r={base_fps}:d={fade_in_duration}",
                        f='lavfi'
                    )
                    all_streams.append(color_in)
                    all_durations.append(float(fade_in_duration))

                # Real video streams, normalized to base_fps
                for vf, dur in zip(video_files, durations):
                    s = ffmpeg.input(str(vf)).filter('fps', fps=base_fps)
                    all_streams.append(s)
                    all_durations.append(float(dur))

                # For fade-out, we have two modes:
                # - When NOT trimming to audio: append a color clip and crossfade into it.
                # - When trimming to audio: we'll do a final fade based on the audio length,
                #   so we do NOT append a separate color clip here.
                append_fadeout_clip = fade_out_enabled and not (audio_path and trim_to_audio)

                # Optional fade out color clip (normalized to base_fps)
                if append_fadeout_clip:
                    # Sanitize is already called outside this block
                    color_out = ffmpeg.input(
                        f"color=c={fade_out_color}:s={width}x{height}:r={base_fps}:d={fade_out_duration}",
                        f='lavfi'
                    )
                    all_streams.append(color_out)
                    all_durations.append(float(fade_out_duration))

                if len(all_streams) == 1:
                    # Only one thing in the chain
                    current = all_streams[0]
                else:
                    # Chain xfade transitions between adjacent clips
                    current = all_streams[0]
                    offset_accum = 0.0

                    total_clips = len(all_streams)
                    clip_indices = range(1, total_clips)
                    iterator = clip_indices

                    for i in iterator:
                        prev_dur = all_durations[i - 1]
                        cur_dur = all_durations[i]

                        # -----------------------------
                        # Choose transition type + duration
                        # -----------------------------
                        if fade_in_enabled and i == 1:
                            # Color -> first real video
                            xfade_transition = "fade"
                            desired = fade_in_duration

                        elif append_fadeout_clip and i == total_clips - 1:
                            # Last real video -> fade into the color clip
                            xfade_transition = "fade"
                            desired = fade_out_duration

                        else:
                            # Regular inter-video transitions
                            if transition == "fade" and len(video_files) >= 2:
                                xfade_transition = "fade"
                                desired = transition_duration
                            else:
                                # For "none", approximate a cut with a tiny crossfade
                                xfade_transition = "fade"
                                desired = min(0.05, prev_dur, cur_dur)

                        # -----------------------------
                        # Clamp duration so it's valid
                        # -----------------------------
                        max_allowed = max(0.0, min(prev_dur, cur_dur) - 0.01)
                        if max_allowed <= 0.0:
                            raise RuntimeError(
                                f"Transition duration cannot be determined for pair {i-1} -> {i} "
                                f"with clip durations {prev_dur:.3f}s and {cur_dur:.3f}s."
                            )

                        actual = min(desired, max_allowed)
                        if actual <= 0.0:
                            raise RuntimeError(
                                f"Computed non-positive transition duration ({actual}) "
                                f"for pair {i-1} -> {i}."
                            )

                        # Compute offset for xfade
                        offset = offset_accum + prev_dur - actual

                        # Apply xfade between current and the next clip
                        current = ffmpeg.filter(
                            [current, all_streams[i]],
                            'xfade',
                            transition=xfade_transition,
                            duration=actual,
                            offset=offset
                        )

                        offset_accum += prev_dur - actual

                # If we're trimming to audio and fade-out is enabled,
                # apply a final fade based on the audio length so it
                # is guaranteed to happen before the render stops.
                if fade_out_enabled and audio_path and trim_to_audio and audio_duration:
                    # Clamp fade duration so it fits inside the audio
                    effective_fade = min(
                        fade_out_duration,
                        max(0.01, audio_duration - 0.01)
                    )
                    fade_start = max(0.0, audio_duration - effective_fade)

                    current = current.filter(
                        'fade',
                        type='out',
                        start_time=fade_start,
                        duration=effective_fade,
                        color=fade_out_color # This uses the already sanitized color!
                    )

                # Setup output with audio if provided
                if audio_path:
                    audio_stream = ffmpeg.input(audio_path)
                    output_args = {
                        'acodec': 'aac',
                    }
                    if trim_to_audio:
                        output_args['shortest'] = None  # -shortest

                    stream = ffmpeg.output(
                        current,
                        audio_stream,
                        output_path,
                        **output_args,
                    )
                else:
                    stream = ffmpeg.output(current, output_path)

            # Run the ffmpeg command
            stream = stream.overwrite_output()
            print("Rendering video...")
            stream.run(overwrite_output=True, quiet=True)
            print("Rendering finished.")

        except ffmpeg.Error as e:
            if e.stderr is not None:
                raise RuntimeError(f"FFmpeg error: {e.stderr.decode()}")
            else:
                raise RuntimeError(f"FFmpeg error: {str(e)}")

        finally:
            # Clean up temporary files
            if (not need_filter_graph) and 'temp_list_path' in locals():
                if os.path.exists(temp_list_path):
                    os.unlink(temp_list_path)
            # temp_audio cleanup needs to be robust as the object might not exist
            if 'temp_audio' in locals() and temp_audio is not None:
                try:
                    temp_audio.close()
                    if os.path.exists(temp_audio.name):
                        os.unlink(temp_audio.name)
                except Exception as e:
                    print(f"Warning: Failed to cleanup temp audio file: {e}")

        # Return the output path as a tuple (required for ComfyUI nodes returning one value)
        return (output_path,)
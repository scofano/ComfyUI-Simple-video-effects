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
                "clip_duration": ("FLOAT", {
                    "default": 5.0,
                    "min": 0.0,
                    "max": 3600.0,
                    "step": 0.1,
                    "round": 0.1,
                    "label": "Random segment length (s, 0=full video)"
                }),

                "transition": (["none", "fade"], {"default": "fade"}),
                "transition_duration": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1,
                    "round": 0.1,
                }),
                # NEW: target resolution + resize mode
                "width": ("INT", {
                    "default": 1080,
                    "min": 1,
                    "max": 8192,
                    "step": 1,
                    "label": "Target width"
                }),
                "height": ("INT", {
                    "default": 1920,
                    "min": 1,
                    "max": 8192,
                    "step": 1,
                    "label": "Target height"
                }),
                "resize_mode": (["crop", "padding"], {
                    "default": "crop",
                    "label": "Resize mode"
                }),
                "resize_padding_color": ("STRING", {
                    "default": "#000000",
                    "multiline": False,
                    "placeholder": "#000000",
                    "label": "Padding background color (hex)"
                }),
            },
            "optional": {
                "sort_files": ("BOOLEAN", {
                    "default": False,
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
                    "default": True,
                    "label": "Enable fade in from color"
                }),
                "fade_in_color": ("STRING", {
                    "default": "#000000",
                    "multiline": False,
                    "placeholder": "#000000",
                    "label": "Fade in color (hex)"
                }),
                "fade_in_duration": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "round": 0.1,
                    "label": "Fade in duration (s)"
                }),
                "fade_out_enabled": ("BOOLEAN", {
                    "default": True,
                    "label": "Enable fade out to color"
                }),
                "fade_out_color": ("STRING", {
                    "default": "#000000",
                    "multiline": False,
                    "placeholder": "#000000",
                    "label": "Fade out color (hex)"
                }),
                "fade_out_duration": ("FLOAT", {
                    "default": 0.5,
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
    CATEGORY = "Simple Video Effects"
    OUTPUT_NODE = True

    def __init__(self):
        # Use a default output directory (ComfyUI's default output, if available).
        self.output_dir = self._get_default_output_directory()

    def _get_default_output_directory(self) -> str:
        """
        Try to get ComfyUI's default output directory (e.g. 'output' inside 'ComfyUI' root)
        or fallback to current directory.
        """
        # --- FIX: Directly target 'ComfyUI/output' relative to the ComfyUI root if possible ---
        try:
            # Assuming the script runs from 'ComfyUI/custom_nodes/...'
            # We want to go up to the ComfyUI root and then to 'output'
            
            # Find the root (where ComfyUI.exe or run_webui.bat/sh would be)
            # This is a robust way to find the main ComfyUI folder:
            current_path = Path(__file__).resolve()
            comfy_root = None
            for parent in current_path.parents:
                if parent.name == "ComfyUI":
                    comfy_root = parent.parent  # Go up one more to the root *containing* ComfyUI
                    break
            
            if comfy_root:
                # Standard ComfyUI output is 'ComfyUI/output'
                fallback_dir = comfy_root / "ComfyUI" / "output"
            else:
                # Fallback to current working directory if root can't be inferred
                fallback_dir = Path(os.getcwd()) / "output"

            os.makedirs(fallback_dir, exist_ok=True)
            return str(fallback_dir)
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

    def _get_video_resolution_and_fps(self, video_path: str):
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
            path, e = exceptions[0]
            raise RuntimeError(f"Error computing duration of '{path}': {e}")

        return durations

    def _process_vhs_audio(self, audio_dict):
        """
        Process VideoHelperSuite-style audio input.
        """

        if not isinstance(audio_dict, dict):
            raise ValueError("music_track must be a dict in VideoHelperSuite format.")

        path = audio_dict.get("path") or audio_dict.get("output_path")
        if path and os.path.exists(path):
            return path, None

        waveform = audio_dict.get("waveform")
        sample_rate = audio_dict.get("sample_rate")

        if waveform is None or sample_rate is None:
            raise ValueError("VideoHelperSuite audio must include either 'path' or ('waveform' and 'sample_rate').")

        if not isinstance(waveform, np.ndarray):
            try:
                waveform = waveform.cpu().numpy() if hasattr(waveform, 'cpu') else np.array(waveform)
            except Exception as e:
                print(f"Warning: Could not convert waveform to numpy for processing. Error: {e}")

        if hasattr(waveform, 'ndim') and waveform.ndim > 2:
            original_shape = waveform.shape
            if hasattr(waveform, 'squeeze'):
                waveform = waveform.squeeze()
            elif original_shape[0] == 1:
                waveform = waveform.reshape(original_shape[1:])
            print(f"Squeezed 3D waveform from {original_shape} to {waveform.shape}")

        if waveform.ndim == 2 and waveform.shape[0] < waveform.shape[1]:
            waveform = waveform.T
            print(f"Transposed 2D waveform to shape: {waveform.shape} (samples, channels)")
        
        sample_rate = int(sample_rate)

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
        
        color = ''.join(c for c in color if c.lower() in '0123456789abcdef')
        if len(color) > 6:
            color = color[:6]
        
        color = color.zfill(6)
        
        return "#" + color

    def _apply_resize_filters(self, video_stream, target_width: int, target_height: int,
                              resize_mode: str, pad_color: str):
        """
        Resize a video stream to target_width x target_height while preserving aspect ratio.
        - crop: scale to cover, then center-crop.
        - padding: scale to fit, then pad with pad_color.
        """
        if target_width <= 0 or target_height <= 0:
            return video_stream

        pad_color = self._sanitize_hex_color(pad_color)
        if resize_mode not in ("crop", "padding"):
            resize_mode = "crop"

        if resize_mode == "crop":
            video_stream = video_stream.filter(
                "scale",
                target_width,
                target_height,
                force_original_aspect_ratio="increase"
            )
            video_stream = video_stream.filter(
                "crop",
                str(target_width),
                str(target_height),
                "(in_w-out_w)/2",
                "(in_h-out_h)/2"
            )
        else:
            video_stream = video_stream.filter(
                "scale",
                target_width,
                target_height,
                force_original_aspect_ratio="decrease"
            )
            video_stream = video_stream.filter(
                "pad",
                str(target_width),
                str(target_height),
                "(ow-iw)/2",
                "(oh-ih)/2",
                color=pad_color
            )

        return video_stream

    def _build_segment_stream(
        self,
        video_path: str,
        base_fps: float,
        target_width: int,
        target_height: int,
        resize_mode: str,
        pad_color: str,
        clip_duration: float,
        original_duration: float,
    ):
        """
        Build a video stream for a single clip:
        - If clip_duration <= 0: use full video.
        - If 0 < clip_duration <= original_duration: pick a random segment of that length.
        - If clip_duration > original_duration: loop the full video until reaching clip_duration.

        Returns (stream, effective_duration).
        """
        if original_duration <= 0:
            raise RuntimeError(
                f"Invalid duration ({original_duration}) for video '{video_path}'"
            )

        pad_color = self._sanitize_hex_color(pad_color)

        # Helper: build one trimmed segment from the file
        def make_segment(start: float, end: float):
            s = ffmpeg.input(video_path)
            s = s.filter("trim", start=start, end=end)
            s = s.filter("setpts", "PTS-STARTPTS")
            return s

        # 1) Use full video
        if clip_duration <= 0:
            seg_stream = ffmpeg.input(video_path)
            effective_duration = original_duration

        # 2) Single random sub-segment
        elif clip_duration <= original_duration:
            max_start = max(0.0, original_duration - clip_duration)
            start = random.uniform(0.0, max_start) if max_start > 0 else 0.0
            end = start + clip_duration
            seg_stream = make_segment(start, end)
            effective_duration = clip_duration

        # 3) Loop video to fill clip_duration
        else:
            full_copies = int(clip_duration // original_duration)
            remainder = clip_duration - full_copies * original_duration

            segments = []
            copies = max(1, full_copies)

            # Full-length copies
            for _ in range(copies):
                segments.append(make_segment(0.0, original_duration))

            # Remainder segment, if needed
            if remainder > 0.01:
                segments.append(make_segment(0.0, remainder))

            if len(segments) == 1:
                seg_stream = segments[0]
            else:
                # concat v-only; fps will be enforced AFTER concat
                seg_stream = ffmpeg.concat(*segments, v=1, a=0)

            effective_duration = clip_duration

        # Normalize FPS once here so downstream filters (xfade, etc.) always
        # see a valid frame rate, with no branching off a shared fps node.
        seg_stream = seg_stream.filter("fps", fps=base_fps)

        # Apply resize (crop or padding) preserving aspect ratio
        seg_stream = self._apply_resize_filters(
            seg_stream,
            target_width,
            target_height,
            resize_mode,
            pad_color,
        )

        return seg_stream, effective_duration

    def combine_videos(
        self,
        directory_path: str,
        output_filename: str,
        file_pattern: str,
        transition: str = "none",
        transition_duration: float = 0.5,
        width: int = 1920,
        height: int = 1080,
        resize_mode: str = "crop",
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
        resize_padding_color: str = "#000000",
        clip_duration: float = 5.0,
    ) -> tuple:
        """
        Combine multiple video files from a directory into a single video file
        with optional transitions, fade in/out from/to color, and optional music.

        NEW:
        - width / height: target resolution of the output video.
        - resize_mode: "crop" (center crop) or "padding" (letterbox/pillarbox with color).
        - resize_padding_color: hex color used for padding when resize_mode="padding".
        - clip_duration: if > 0, use a random segment of this length (seconds) from each clip.
                         If the requested duration is longer than the clip, loop the clip
                         until that duration is reached. If 0, use full video as before.
        - If an audio track is present and its duration is longer than
          (number_of_files × seconds_per_file), the node will keep randomly
          adding segments until the total video duration reaches the audio duration.
        """

        directory = Path(directory_path).expanduser().resolve()
        if not directory.is_dir():
            raise ValueError(f"Directory does not exist: {directory}")

        video_files = sorted(directory.glob(file_pattern))
        if not video_files:
            raise ValueError(f"No video files found in directory '{directory}' matching pattern '{file_pattern}'")

        if random_order:
            if seed != -1:
                random.seed(seed)
            random.shuffle(video_files)
        elif sort_files:
            video_files.sort()

        audio_path = None
        temp_audio = None
        audio_duration = None

        if music_track is not None:
            audio_path, temp_audio = self._process_vhs_audio(music_track)
            if audio_path and trim_to_audio:
                try:
                    audio_duration = self._get_video_duration(audio_path)
                except Exception as e:
                    print(f"Warning: could not probe audio duration: {e}")
                    audio_duration = None

        output_filename = output_filename.strip() or "combined_output.mp4"
        if not output_filename.lower().endswith(".mp4"):
            output_filename += ".mp4"

        output_path = os.path.join(self.output_dir, output_filename)
        output_path = self.get_unique_filename(output_path)

        fade_in_color = self._sanitize_hex_color(fade_in_color)
        fade_out_color = self._sanitize_hex_color(fade_out_color)
        resize_padding_color = self._sanitize_hex_color(resize_padding_color)

        first_w, first_h, base_fps = self._get_video_resolution_and_fps(str(video_files[0]))

        if width <= 0 or height <= 0:
            target_width, target_height = first_w, first_h
        else:
            target_width, target_height = width, height

        print(
            f"Using target resolution {target_width}x{target_height} at {base_fps:.3f} fps "
            f"with resize mode '{resize_mode}'."
        )

        # Use advanced graph if transitions/fades OR we're using random clip durations
        need_filter_graph = (
            (transition == "fade" and len(video_files) >= 2) or
            fade_in_enabled or
            fade_out_enabled or
            clip_duration > 0
        )

        try:
            if not need_filter_graph:
                # Simple path: full clips, no random trimming, no color fades
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                    for video_file in video_files:
                        path_str = str(video_file.absolute()).replace("'", r"'\''")
                        f.write(f"file '{path_str}'\n")
                    temp_list_path = f.name

                stream = ffmpeg.input(temp_list_path, f='concat', safe=0)

                stream = stream.filter('fps', fps=base_fps)
                stream = self._apply_resize_filters(
                    stream,
                    target_width,
                    target_height,
                    resize_mode,
                    resize_padding_color
                )

                if audio_path:
                    audio_stream = ffmpeg.input(audio_path)
                    output_args = {
                        'acodec': 'aac',
                    }
                    if trim_to_audio:
                        output_args['shortest'] = None

                    stream = ffmpeg.output(
                        stream,
                        audio_stream,
                        output_path,
                        **output_args,
                    )
                else:
                    stream = ffmpeg.output(stream, output_path)

            else:
                # Advanced path: transitions / fades / random segments
                original_durations = self._compute_durations_parallel(video_files)

                # "Nominal" per-clip duration (used for clamping transitions)
                if clip_duration > 0:
                    per_clip_duration = clip_duration
                else:
                    per_clip_duration = min(original_durations)

                if transition == "fade" and len(video_files) >= 2:
                    if transition_duration >= per_clip_duration:
                        clamped = max(0.1, per_clip_duration - 0.1)
                        print(
                            f"Requested transition_duration ({transition_duration:.3f}s) "
                            f"is >= effective clip length ({per_clip_duration:.3f}s). "
                            f"Clamping to {clamped:.3f}s."
                        )
                        transition_duration = clamped

                print(
                    f"Using target resolution {target_width}x{target_height} at {base_fps:.3f} fps "
                    f"for all clips/color fades."
                )

                # --- Build a playlist that may extend to cover full audio duration ---
                # Start with each file once
                playlist = list(zip(video_files, original_durations))

                if audio_path and trim_to_audio and audio_duration is not None:
                    # Base total duration of one pass through all files (no overlaps)
                    if clip_duration > 0:
                        base_total = len(video_files) * clip_duration
                        nominal_clip_len = clip_duration
                    else:
                        base_total = sum(original_durations)
                        nominal_clip_len = base_total / max(1, len(original_durations))

                    target = audio_duration

                    # Estimate how much we lose per transition due to xfade
                    if transition == "fade" and len(video_files) >= 2:
                        nominal_xfade = transition_duration
                    else:
                        # the "no transition" path still uses a tiny xfade (~0.05s)
                        nominal_xfade = min(0.05, nominal_clip_len)

                    def estimate_effective(raw, n_clips):
                        n_xfades = max(0, n_clips - 1)
                        return raw - n_xfades * nominal_xfade

                    cum_raw = base_total
                    clips_count = len(playlist)
                    cum_effective = estimate_effective(cum_raw, clips_count)

                    if cum_effective < target:
                        print(
                            f"Audio is longer ({audio_duration:.3f}s) than base video duration "
                            f"({cum_effective:.3f}s after crossfades). Extending with random clips..."
                        )

                        # Keep adding random clips until the *effective* duration
                        # (after estimated overlaps) reaches the audio length
                        while cum_effective < target:
                            idx = random.randrange(len(video_files))
                            vf = video_files[idx]
                            dur = original_durations[idx]
                            playlist.append((vf, dur))
                            clips_count += 1

                            if clip_duration > 0:
                                cum_raw += clip_duration
                            else:
                                cum_raw += dur

                            cum_effective = estimate_effective(cum_raw, clips_count)

                        print(
                            f"Extended playlist to approx {cum_effective:.3f}s "
                            f"(raw {cum_raw:.3f}s) to cover audio duration ({audio_duration:.3f}s)."
                        )

                all_streams = []
                all_durations = []

                if fade_in_enabled:
                    color_in = ffmpeg.input(
                        f"color=c={fade_in_color}:s={target_width}x{target_height}:r={base_fps}:d={fade_in_duration}",
                        f='lavfi'
                    )
                    all_streams.append(color_in)
                    all_durations.append(float(fade_in_duration))

                # Build per-clip segment streams (random/looped) and durations
                for vf, orig_dur in playlist:
                    seg_stream, eff_dur = self._build_segment_stream(
                        str(vf),
                        base_fps,
                        target_width,
                        target_height,
                        resize_mode,
                        resize_padding_color,
                        clip_duration,
                        orig_dur,
                    )
                    all_streams.append(seg_stream)
                    all_durations.append(float(eff_dur))

                # Only add a final color clip if we're NOT trimming to audio
                append_fadeout_clip = fade_out_enabled and not (audio_path and trim_to_audio)

                if append_fadeout_clip:
                    color_out = ffmpeg.input(
                        f"color=c={fade_out_color}:s={target_width}x{target_height}:r={base_fps}:d={fade_out_duration}",
                        f='lavfi'
                    )
                    all_streams.append(color_out)
                    all_durations.append(float(fade_out_duration))

                # Chain with xfade
                if len(all_streams) == 1:
                    current = all_streams[0]
                else:
                    current = all_streams[0]
                    offset_accum = 0.0

                    total_clips = len(all_streams)
                    for i in range(1, total_clips):
                        prev_dur = all_durations[i - 1]
                        cur_dur = all_durations[i]

                        if fade_in_enabled and i == 1:
                            xfade_transition = "fade"
                            desired = fade_in_duration

                        elif append_fadeout_clip and i == total_clips - 1:
                            xfade_transition = "fade"
                            desired = fade_out_duration

                        else:
                            if transition == "fade" and len(playlist) >= 2:
                                xfade_transition = "fade"
                                desired = transition_duration
                            else:
                                xfade_transition = "fade"
                                desired = min(0.05, prev_dur, cur_dur)

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

                        offset = offset_accum + prev_dur - actual

                        current = ffmpeg.filter(
                            [current, all_streams[i]],
                            'xfade',
                            transition=xfade_transition,
                            duration=actual,
                            offset=offset
                        )

                        offset_accum += prev_dur - actual

                # If we have audio and are trimming to its length, optionally fade video out
                if fade_out_enabled and audio_path and trim_to_audio and audio_duration:
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
                        color=fade_out_color
                    )

                if audio_path:
                    audio_stream = ffmpeg.input(audio_path)
                    output_args = {
                        'acodec': 'aac',
                    }
                    if trim_to_audio:
                        output_args['shortest'] = None

                    stream = ffmpeg.output(
                        current,
                        audio_stream,
                        output_path,
                        **output_args,
                    )
                else:
                    stream = ffmpeg.output(current, output_path)

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
            if (not need_filter_graph) and 'temp_list_path' in locals():
                if os.path.exists(temp_list_path):
                    os.unlink(temp_list_path)
            if 'temp_audio' in locals() and temp_audio is not None:
                try:
                    temp_audio.close()
                    if os.path.exists(temp_audio.name):
                        os.unlink(temp_audio.name)
                except Exception as e:
                    print(f"Warning: Failed to cleanup temp audio file: {e}")

        return (output_path,)

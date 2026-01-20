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
    - Optionally uses GPU-accelerated encoding (NVENC) when available
    - Uses tqdm to show progress in the console
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
                # Target resolution + resize mode
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
                # GPU toggle
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

    # ---------- tqdm + ffmpeg helpers ----------

    def _make_tqdm(self, *args, **kwargs):
        if tqdm is None:
            return None
        return tqdm(*args, **kwargs)

    def _run_ffmpeg_with_progress(self, stream, total_duration=None):
        """
        Run ffmpeg while parsing 'time=' from stderr and updating tqdm.
        Falls back to quiet run if tqdm or total_duration is not available.
        """
        if (tqdm is None) or (total_duration is None) or (total_duration <= 0):
            stream.run(overwrite_output=True, quiet=True)
            return

        import re
        time_re = re.compile(r'time=(\d+):(\d+):(\d+\.\d+)')

        pbar = self._make_tqdm(total=total_duration, desc="Rendering", unit="sec")

        process = ffmpeg.run_async(
            stream,
            pipe_stdin=False,
            pipe_stdout=True,
            pipe_stderr=True,
            overwrite_output=True,
        )

        stderr_chunks = []

        try:
            while True:
                line = process.stderr.readline()
                if not line:
                    break
                stderr_chunks.append(line)

                if isinstance(line, bytes):
                    line_str = line.decode(errors="ignore")
                else:
                    line_str = line

                m = time_re.search(line_str)
                if m:
                    h, m_, s = m.groups()
                    seconds = int(h) * 3600 + int(m_) * 60 + float(s)
                    if pbar:
                        pbar.n = min(seconds, total_duration)
                        pbar.refresh()
            process.wait()
        finally:
            if pbar:
                pbar.close()

        if process.returncode != 0:
            stderr_text = b"".join(
                ch if isinstance(ch, bytes) else ch.encode()
                for ch in stderr_chunks
            ).decode(errors="ignore")
            raise RuntimeError(
                f"FFmpeg process failed with code {process.returncode}:\n{stderr_text}"
            )

    # ---------- filesystem / probing helpers ----------

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

    def _get_video_resolution_and_fps(self, video_path: str):
        try:
            probe = ffmpeg.probe(video_path)
        except ffmpeg.Error as e:
            msg = e.stderr.decode() if getattr(e, "stderr", None) else str(e)
            raise RuntimeError(f"Failed to probe video '{video_path}': {msg}")
        except Exception as e:
            raise RuntimeError(f"Failed to probe video '{video_path}': {str(e)}")

        streams = probe.get("streams", [])
        vstreams = [s for s in streams if s.get("codec_type") == "video"]
        if not vstreams:
            raise RuntimeError(f"No video stream found in '{video_path}'.")

        v0 = vstreams[0]
        width = int(v0.get("width", 0))
        height = int(v0.get("height", 0))

        fps_str = v0.get("r_frame_rate") or v0.get("avg_frame_rate") or "0/0"
        try:
            num, den = fps_str.split("/")
            num, den = float(num), float(den)
            fps = num / den if den != 0 else 0.0
        except Exception:
            fps = 0.0

        if width <= 0 or height <= 0:
            raise RuntimeError(
                f"Invalid resolution for '{video_path}': {width}x{height}"
            )
        if fps <= 0:
            fps = 30.0

        return width, height, fps

    def _compute_durations_parallel(self, video_files):
        durations = [0.0] * len(video_files)
        exceptions = []

        pbar = self._make_tqdm(
            total=len(video_files),
            desc="Probing clip durations",
            unit="file"
        )

        def worker(idx, path):
            nonlocal durations, exceptions
            try:
                durations[idx] = self._get_video_duration(str(path))
            except Exception as e:
                exceptions.append((path, e))
            finally:
                if pbar:
                    pbar.update(1)

        with ThreadPoolExecutor(max_workers=min(8, len(video_files))) as executor:
            futures = []
            for idx, vf in enumerate(video_files):
                futures.append(executor.submit(worker, idx, vf))

            for f in futures:
                f.result()

        if pbar:
            pbar.close()

        if exceptions:
            path, e = exceptions[0]
            raise RuntimeError(f"Error computing duration of '{path}': {e}")

        return durations

    # ---------- audio & color helpers ----------

    def _process_vhs_audio(self, audio_dict):
        if not isinstance(audio_dict, dict):
            raise ValueError("music_track must be a dict in VideoHelperSuite format.")

        path = audio_dict.get("path") or audio_dict.get("output_path")
        if path and os.path.exists(path):
            return path, None

        waveform = audio_dict.get("waveform")
        sample_rate = audio_dict.get("sample_rate")

        if waveform is None or sample_rate is None:
            raise ValueError(
                "VideoHelperSuite audio must include either 'path' or "
                "('waveform' and 'sample_rate')."
            )

        if not isinstance(waveform, np.ndarray):
            try:
                waveform = (
                    waveform.cpu().numpy()
                    if hasattr(waveform, "cpu")
                    else np.array(waveform)
                )
            except Exception as e:
                print(
                    f"Warning: Could not convert waveform to numpy for "
                    f"processing. Error: {e}"
                )

        if hasattr(waveform, "ndim") and waveform.ndim > 2:
            original_shape = waveform.shape
            if hasattr(waveform, "squeeze"):
                waveform = waveform.squeeze()
            elif original_shape[0] == 1:
                waveform = waveform.reshape(original_shape[1:])
            print(f"Squeezed 3D waveform from {original_shape} to {waveform.shape}")

        if waveform.ndim == 2 and waveform.shape[0] < waveform.shape[1]:
            waveform = waveform.T
            print(
                f"Transposed 2D waveform to shape: {waveform.shape} "
                "(samples, channels)"
            )

        sample_rate = int(sample_rate)

        temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_audio.close()

        sf.write(temp_audio.name, waveform, sample_rate)

        return temp_audio.name, temp_audio

    def get_unique_filename(self, output_path: str) -> str:
        base, ext = os.path.splitext(output_path)
        index = 1
        unique_path = output_path

        while os.path.exists(unique_path):
            unique_path = f"{base}_{index}{ext}"
            index += 1

        return unique_path

    def _sanitize_hex_color(self, color: str) -> str:
        color = color.strip().lstrip("#")
        if not color:
            return "#000000"

        color = "".join(c for c in color if c.lower() in "0123456789abcdef")
        if len(color) > 6:
            color = color[:6]

        color = color.zfill(6)
        return "#" + color

    # ---------- resize & segment helpers ----------

    def _apply_resize_filters(self, video_stream, target_width: int, target_height: int,
                              resize_mode: str, pad_color: str):
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
                force_original_aspect_ratio="increase",
            )
            video_stream = video_stream.filter(
                "crop",
                str(target_width),
                str(target_height),
                "(in_w-out_w)/2",
                "(in_h-out_h)/2",
            )
        else:
            video_stream = video_stream.filter(
                "scale",
                target_width,
                target_height,
                force_original_aspect_ratio="decrease",
            )
            video_stream = video_stream.filter(
                "pad",
                str(target_width),
                str(target_height),
                "(ow-iw)/2",
                "(oh-ih)/2",
                color=pad_color,
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
        if original_duration <= 0:
            raise RuntimeError(
                f"Invalid duration ({original_duration}) for video '{video_path}'"
            )

        pad_color = self._sanitize_hex_color(pad_color)

        def make_segment(start: float, end: float):
            s = ffmpeg.input(video_path)
            s = s.filter("trim", start=start, end=end)
            s = s.filter("setpts", "PTS-STARTPTS")
            return s

        if clip_duration <= 0:
            seg_stream = ffmpeg.input(video_path)
            effective_duration = original_duration

        elif clip_duration <= original_duration:
            max_start = max(0.0, original_duration - clip_duration)
            start = random.uniform(0.0, max_start) if max_start > 0 else 0.0
            end = start + clip_duration
            seg_stream = make_segment(start, end)
            effective_duration = clip_duration

        else:
            full_copies = int(clip_duration // original_duration)
            remainder = clip_duration - full_copies * original_duration

            segments = []
            copies = max(1, full_copies)

            for _ in range(copies):
                segments.append(make_segment(0.0, original_duration))

            if remainder > 0.01:
                segments.append(make_segment(0.0, remainder))

            if len(segments) == 1:
                seg_stream = segments[0]
            else:
                seg_stream = ffmpeg.concat(*segments, v=1, a=0)

            effective_duration = clip_duration

        # Normalize FPS so downstream xfade sees consistent frame rate
        seg_stream = seg_stream.filter("fps", fps=base_fps)
        # 🔧 Normalize timebase so xfade inputs match
        seg_stream = seg_stream.filter("settb", "AVTB")

        seg_stream = self._apply_resize_filters(
            seg_stream,
            target_width,
            target_height,
            resize_mode,
            pad_color,
        )

        return seg_stream, effective_duration

    # ---------- chunked xfade renderer ----------

    def _render_chunk(
        self,
        prev_file,
        prev_duration,
        chunk_segments,
        *,
        base_fps,
        target_width,
        target_height,
        resize_mode,
        resize_padding_color,
        clip_duration,
        fade_in_enabled,
        fade_in_color,
        fade_in_duration,
        fade_out_enabled,
        fade_out_color,
        fade_out_duration,
        transition,
        transition_duration,
        is_first_chunk,
        is_last_chunk_no_audio_trim,
        playlist_length,
        use_gpu,
    ):
        """
        Build a smaller filter graph for a subset of segments + (optional) previous result,
        then render to a temp video file.
        """
        all_streams = []
        all_durations = []

        # Fade-in color clip (first chunk only, if enabled)
        if is_first_chunk and fade_in_enabled:
            color_in = ffmpeg.input(
                (
                    f"color=c={fade_in_color}:"
                    f"s={target_width}x{target_height}:"
                    f"r={base_fps}:d={fade_in_duration}"
                ),
                f="lavfi",
            )
            # match fps/timebase
            color_in = color_in.filter("fps", fps=base_fps).filter("settb", "AVTB")
            all_streams.append(color_in)
            all_durations.append(float(fade_in_duration))

        # Previous chunk input
        if prev_file is not None:
            prev_stream = ffmpeg.input(prev_file)
            prev_stream = prev_stream.filter("fps", fps=base_fps).filter("settb", "AVTB")
            all_streams.append(prev_stream)
            all_durations.append(float(prev_duration))

        # New segments for this chunk
        for vf, orig_dur in chunk_segments:
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

        # Optional fade-out color clip in video-only mode (no trim_to_audio)
        append_fadeout_clip = is_last_chunk_no_audio_trim and fade_out_enabled
        if append_fadeout_clip:
            color_out = ffmpeg.input(
                (
                    f"color=c={fade_out_color}:"
                    f"s={target_width}x{target_height}:"
                    f"r={base_fps}:d={fade_out_duration}"
                ),
                f="lavfi",
            )
            color_out = color_out.filter("fps", fps=base_fps).filter("settb", "AVTB")
            all_streams.append(color_out)
            all_durations.append(float(fade_out_duration))

        if not all_streams:
            raise RuntimeError("Empty chunk passed to _render_chunk")

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

                # Special handling for first fade-in
                if (
                    is_first_chunk
                    and fade_in_enabled
                    and prev_file is None
                    and i == 1
                ):
                    xfade_transition = "fade"
                    desired = fade_in_duration

                # Special handling for last fade-out color in video-only mode
                elif append_fadeout_clip and i == total_clips - 1:
                    xfade_transition = "fade"
                    desired = fade_out_duration

                else:
                    if transition == "fade" and playlist_length >= 2:
                        xfade_transition = "fade"
                        desired = transition_duration
                    else:
                        xfade_transition = "fade"
                        desired = min(0.05, prev_dur, cur_dur)

                max_allowed = max(0.0, min(prev_dur, cur_dur) - 0.01)
                if max_allowed <= 0.0:
                    raise RuntimeError(
                        f"Transition duration cannot be determined for pair "
                        f"{i-1} -> {i} with clip durations "
                        f"{prev_dur:.3f}s and {cur_dur:.3f}s."
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
                    "xfade",
                    transition=xfade_transition,
                    duration=actual,
                    offset=offset,
                )
                offset_accum += prev_dur - actual

        total_chunk_duration = sum(all_durations)

        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp_path = tmp.name
        tmp.close()

        out_kwargs = {}
        if use_gpu:
            out_kwargs["vcodec"] = "h264_nvenc"

        stream = ffmpeg.output(current, tmp_path, **out_kwargs).overwrite_output()
        print(f"Rendering chunk to {tmp_path}...")
        self._run_ffmpeg_with_progress(stream, total_chunk_duration)
        print("Chunk rendered.")

        new_duration = self._get_video_duration(tmp_path)
        return tmp_path, new_duration

    # ---------- main entry point ----------

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

        first_w, first_h, base_fps = self._get_video_resolution_and_fps(
            str(video_files[0])
        )

        if width <= 0 or height <= 0:
            target_width, target_height = first_w, first_h
        else:
            target_width, target_height = width, height

        print(
            f"Using target resolution {target_width}x{target_height} at {base_fps:.3f} fps "
            f"with resize mode '{resize_mode}'. GPU: {'ON' if use_gpu else 'OFF'}"
        )

        need_filter_graph = (
            (transition == "fade" and len(video_files) >= 2)
            or fade_in_enabled
            or fade_out_enabled
            or clip_duration > 0
        )

        temp_list_path = None
        temp_chunk_path = None

        try:
            if not need_filter_graph:
                # Simple concat path
                try:
                    simple_durations = self._compute_durations_parallel(video_files)
                    total_render_duration = sum(simple_durations)
                except Exception:
                    total_render_duration = None

                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".txt", delete=False
                ) as f:
                    for video_file in video_files:
                        path_str = str(video_file.absolute()).replace("'", r"'\''")
                        f.write(f"file '{path_str}'\n")
                    temp_list_path = f.name

                stream = ffmpeg.input(temp_list_path, f="concat", safe=0)
                stream = stream.filter("fps", fps=base_fps)
                stream = self._apply_resize_filters(
                    stream,
                    target_width,
                    target_height,
                    resize_mode,
                    resize_padding_color,
                )

                if audio_path:
                    audio_stream = ffmpeg.input(audio_path)
                    output_args = {"acodec": "aac"}
                    if trim_to_audio:
                        output_args["shortest"] = None
                    if use_gpu:
                        output_args["vcodec"] = "h264_nvenc"

                    stream = ffmpeg.output(
                        stream,
                        audio_stream,
                        output_path,
                        **output_args,
                    )
                else:
                    if use_gpu:
                        stream = ffmpeg.output(
                            stream, output_path, vcodec="h264_nvenc"
                        )
                    else:
                        stream = ffmpeg.output(stream, output_path)

                stream = stream.overwrite_output()
                print("Rendering video...")
                self._run_ffmpeg_with_progress(stream, total_render_duration)
                print("Rendering finished.")

            else:
                # Advanced path: transitions, fades, random segments, chunked
                original_durations = self._compute_durations_parallel(video_files)

                if clip_duration > 0:
                    nominal_clip_len = clip_duration
                else:
                    nominal_clip_len = sum(original_durations) / max(
                        1, len(original_durations)
                    )

                if transition == "fade" and len(video_files) >= 2:
                    nominal_xfade = transition_duration
                else:
                    nominal_xfade = min(0.05, nominal_clip_len)

                def estimate_effective(raw, n_clips):
                    n_xfades = max(0, n_clips - 1)
                    return raw - n_xfades * nominal_xfade

                # Build playlist with or without trimming to audio
                if audio_path and trim_to_audio and audio_duration is not None:
                    target = audio_duration
                    playlist = []
                    cum_raw = 0.0
                    clips_count = 0
                    MAX_CLIPS = 100

                    while clips_count < MAX_CLIPS:
                        idx = random.randrange(len(video_files))
                        vf = video_files[idx]
                        orig_dur = original_durations[idx]

                        playlist.append((vf, orig_dur))
                        clips_count += 1

                        clip_len = clip_duration if clip_duration > 0 else orig_dur
                        cum_raw += clip_len

                        cum_effective = estimate_effective(cum_raw, clips_count)
                        if cum_effective >= target:
                            break
                else:
                    playlist = list(zip(video_files, original_durations))
                    cum_raw = 0.0
                    clips_count = 0
                    for _, dur in playlist:
                        if clip_duration > 0:
                            cum_raw += clip_duration
                        else:
                            cum_raw += dur
                        clips_count += 1
                    cum_effective = estimate_effective(cum_raw, clips_count)

                    if audio_path and audio_duration is not None and cum_effective < audio_duration:
                        print(
                            f"Audio is longer ({audio_duration:.3f}s) than base video duration "
                            f"({cum_effective:.3f}s after crossfades). Extending with random clips..."
                        )
                        target = audio_duration
                        while cum_effective < target and clips_count < 100:
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

                # ---- CHUNKED RENDERING ----
                CHUNK_SIZE = 12  # tune this if needed. Maybe it will be an INT input. 8 is a safe number.
                temp_chunk_path = None
                temp_chunk_duration = 0.0
                playlist_length = len(playlist)

                pbar_build = self._make_tqdm(
                    total=playlist_length,
                    desc="Building & rendering chunks",
                    unit="clip",
                )

                idx = 0
                while idx < playlist_length:
                    chunk_segments = playlist[idx: idx + CHUNK_SIZE]
                    is_first_chunk = temp_chunk_path is None
                    is_last_chunk = (idx + CHUNK_SIZE) >= playlist_length
                    is_last_chunk_no_audio_trim = (
                        is_last_chunk and not (audio_path and trim_to_audio)
                    )

                    new_path, new_duration = self._render_chunk(
                        temp_chunk_path,
                        temp_chunk_duration,
                        chunk_segments,
                        base_fps=base_fps,
                        target_width=target_width,
                        target_height=target_height,
                        resize_mode=resize_mode,
                        resize_padding_color=resize_padding_color,
                        clip_duration=clip_duration,
                        fade_in_enabled=fade_in_enabled,
                        fade_in_color=fade_in_color,
                        fade_in_duration=fade_in_duration,
                        fade_out_enabled=fade_out_enabled,
                        fade_out_color=fade_out_color,
                        fade_out_duration=fade_out_duration,
                        transition=transition,
                        transition_duration=transition_duration,
                        is_first_chunk=is_first_chunk,
                        is_last_chunk_no_audio_trim=is_last_chunk_no_audio_trim,
                        playlist_length=playlist_length,
                        use_gpu=use_gpu,
                    )

                    if temp_chunk_path is not None and os.path.exists(temp_chunk_path):
                        try:
                            os.remove(temp_chunk_path)
                        except Exception:
                            pass

                    temp_chunk_path = new_path
                    temp_chunk_duration = new_duration

                    if pbar_build:
                        pbar_build.update(len(chunk_segments))

                    idx += CHUNK_SIZE

                if pbar_build:
                    pbar_build.close()

                if temp_chunk_path is None:
                    raise RuntimeError("Chunked rendering produced no output video.")

                final_video_path = temp_chunk_path
                final_video_duration = temp_chunk_duration

                # ---- FINAL PASS: attach audio, optional fade-out to audio length ----
                if audio_path:
                    video_in = ffmpeg.input(final_video_path)

                    if (
                        fade_out_enabled
                        and audio_path
                        and trim_to_audio
                        and audio_duration
                    ):
                        effective_fade = min(
                            fade_out_duration,
                            max(0.01, audio_duration - 0.01),
                        )
                        fade_start = max(0.0, audio_duration - effective_fade)

                        video_in = video_in.filter(
                            "fade",
                            type="out",
                            start_time=fade_start,
                            duration=effective_fade,
                            color=fade_out_color,
                        )

                    audio_in = ffmpeg.input(audio_path)

                    output_args = {"acodec": "aac"}
                    if trim_to_audio:
                        output_args["shortest"] = None
                    if use_gpu:
                        output_args["vcodec"] = "h264_nvenc"

                    stream = ffmpeg.output(
                        video_in,
                        audio_in,
                        output_path,
                        **output_args,
                    )

                    total_render_duration = (
                        audio_duration if audio_duration else final_video_duration
                    )
                    stream = stream.overwrite_output()
                    print("Rendering final video with audio...")
                    self._run_ffmpeg_with_progress(stream, total_render_duration)
                    print("Rendering finished.")

                else:
                    video_in = ffmpeg.input(final_video_path)
                    if use_gpu:
                        stream = ffmpeg.output(
                            video_in, output_path, vcodec="h264_nvenc"
                        )
                    else:
                        stream = ffmpeg.output(video_in, output_path)

                    total_render_duration = final_video_duration
                    stream = stream.overwrite_output()
                    print("Rendering final video...")
                    self._run_ffmpeg_with_progress(stream, total_render_duration)
                    print("Rendering finished.")

        except (ffmpeg.Error, RuntimeError) as e:
            if isinstance(e, ffmpeg.Error) and getattr(e, "stderr", None) is not None:
                raise RuntimeError(f"FFmpeg error: {e.stderr.decode()}")
            else:
                raise RuntimeError(f"FFmpeg error: {str(e)}")

        finally:
            if temp_list_path and os.path.exists(temp_list_path):
                try:
                    os.remove(temp_list_path)
                except Exception:
                    pass

            if temp_chunk_path and os.path.exists(temp_chunk_path):
                try:
                    os.remove(temp_chunk_path)
                except Exception:
                    pass

            if temp_audio is not None:
                try:
                    temp_audio.close()
                    if os.path.exists(temp_audio.name):
                        os.remove(temp_audio.name)
                except Exception as e:
                    print(f"Warning: Failed to cleanup temp audio file: {e}")

        return (output_path,)

import os
import subprocess
from pathlib import Path
import tempfile
import random
import re
import ffmpeg
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional, List, Tuple
import comfy.utils

# Import numpy and soundfile here, as they are needed for audio processing
import numpy as np
import soundfile as sf

@dataclass
class VideoConfig:
    width: int
    height: int
    resize_mode: str
    resize_padding_color: str
    use_gpu: bool
    clip_duration: float
    fps: float = 0.0

@dataclass
class AudioConfig:
    music_track: Optional[dict]
    trim_to_audio: bool
    path: Optional[str] = None
    duration: Optional[float] = None

@dataclass
class FileConfig:
    directory_path: str
    file_pattern: str
    sort_files: bool
    random_order: bool
    seed: int

@dataclass
class PlaylistConfig:
    video_files: List[Path]
    original_durations: List[float]
    clip_duration: float
    audio_duration: Optional[float]
    trim_to_audio: bool

@dataclass
class TransitionConfig:
    type: str
    duration: float
    fade_in_enabled: bool
    fade_in_color: str
    fade_in_duration: float
    fade_out_enabled: bool
    fade_out_color: str
    fade_out_duration: float

@dataclass
class ChunkConfig:
    prev_file: Optional[str]
    prev_duration: float
    chunk_segments: List[Tuple[Path, float]]
    is_first_chunk: bool
    is_last_chunk_no_audio_trim: bool
    playlist_length: int

@dataclass
class RenderResult:
    path: str
    duration: float

@dataclass
class RenderContext:
    video_cfg: VideoConfig
    audio_cfg: AudioConfig
    trans_cfg: TransitionConfig

@dataclass
class XFadeContext:
    i: int
    n: int
    durations: List[float]
    trans_cfg: TransitionConfig
    chunk_cfg: ChunkConfig

class VideoProcessor:
    def _parse_ffmpeg_time(self, line: str, time_re: re.Pattern) -> Optional[float]:
        m = time_re.search(line)
        if m:
            h, m_, s = m.groups()
            return int(h) * 3600 + int(m_) * 60 + float(s)
        return None

    def run_ffmpeg_with_progress(self, stream, total_duration=None):
        """
        Run ffmpeg while parsing 'time=' from stderr and updating ProgressBar.
        """
        if (total_duration is None) or (total_duration <= 0):
            stream.run(overwrite_output=True, quiet=True)
            return

        time_re = re.compile(r'time=(\d+):(\d+):(\d+\.\d+)')
        pbar = comfy.utils.ProgressBar(100)

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

                line_str = line.decode(errors="ignore") if isinstance(line, bytes) else line
                seconds = self._parse_ffmpeg_time(line_str, time_re)
                if seconds is not None:
                    percentage = int((min(seconds, total_duration) / total_duration) * 100)
                    pbar.update_absolute(percentage, 100)
            process.wait()
        finally:
            pass

        if process.returncode != 0:
            stderr_text = b"".join(
                ch if isinstance(ch, bytes) else ch.encode()
                for ch in stderr_chunks
            ).decode(errors="ignore")
            raise RuntimeError(
                f"FFmpeg process failed with code {process.returncode}:\n{stderr_text}"
            )

    def get_duration(self, video_path: str) -> float:
        try:
            probe = ffmpeg.probe(video_path)
        except (ffmpeg.Error, Exception) as e:
            msg = e.stderr.decode() if hasattr(e, "stderr") and e.stderr else str(e)
            raise RuntimeError(f"Failed to probe video '{video_path}': {msg}")

        fmt = probe.get("format", {})
        if "duration" not in fmt:
            raise RuntimeError(f"Could not determine duration for video '{video_path}'")

        try:
            return float(fmt["duration"])
        except Exception as e:
            raise RuntimeError(f"Invalid duration value for video '{video_path}': {fmt['duration']} ({e})")

    def _parse_fps(self, fps_str: Optional[str]) -> float:
        if not fps_str:
            return 0.0
        try:
            num, den = fps_str.split("/")
            return float(num) / float(den) if float(den) != 0 else 0.0
        except (ValueError, ZeroDivisionError, AttributeError):
            return 0.0

    def _get_video_stream(self, video_path: str):
        try:
            probe = ffmpeg.probe(video_path)
        except ffmpeg.Error as e:
            msg = e.stderr.decode() if getattr(e, "stderr", None) else str(e)
            raise RuntimeError(f"Failed to probe video '{video_path}': {msg}")

        vstreams = [s for s in probe.get("streams", []) if s.get("codec_type") == "video"]
        if not vstreams:
            raise RuntimeError(f"No video stream found in '{video_path}'.")
        return vstreams[0]

    def get_resolution_and_fps(self, video_path: str) -> Tuple[int, int, float]:
        v0 = self._get_video_stream(video_path)
        width, height = int(v0.get("width", 0)), int(v0.get("height", 0))
        if width <= 0 or height <= 0:
            raise RuntimeError(f"Invalid resolution for '{video_path}': {width}x{height}")

        fps_str = v0.get("r_frame_rate") or v0.get("avg_frame_rate")
        fps = self._parse_fps(fps_str)
        return width, height, fps if fps > 0 else 30.0

    def compute_durations_parallel(self, video_files: List[Path]) -> List[float]:
        durations = [0.0] * len(video_files)
        exceptions = []
        pbar = comfy.utils.ProgressBar(len(video_files))

        def worker(idx, path):
            try:
                durations[idx] = self.get_duration(str(path))
            except Exception as e:
                exceptions.append((path, e))
            finally:
                pbar.update(1)

        with ThreadPoolExecutor(max_workers=min(8, len(video_files))) as executor:
            list(executor.map(lambda p: worker(*p), enumerate(video_files)))

        if exceptions:
            path, e = exceptions[0]
            raise RuntimeError(f"Error computing duration of '{path}': {e}")

        return durations

    def sanitize_hex_color(self, color: str) -> str:
        color = color.strip().lstrip("#")
        if not color:
            return "#000000"
        color = "".join(c for c in color if c.lower() in "0123456789abcdef")
        if len(color) > 6:
            color = color[:6]
        return "#" + color.zfill(6)

    def apply_resize_filters(self, stream, config: VideoConfig):
        if config.width <= 0 or config.height <= 0:
            return stream

        pad_color = self.sanitize_hex_color(config.resize_padding_color)
        if config.resize_mode == "crop":
            stream = stream.filter("scale", config.width, config.height, force_original_aspect_ratio="increase")
            stream = stream.filter("crop", str(config.width), str(config.height), "(in_w-out_w)/2", "(in_h-out_h)/2")
        else:
            stream = stream.filter("scale", config.width, config.height, force_original_aspect_ratio="decrease")
            stream = stream.filter("pad", str(config.width), str(config.height), "(ow-iw)/2", "(oh-ih)/2", color=pad_color)
        return stream

    def build_segment_stream(self, video_path: str, video_config: VideoConfig, original_duration: float):
        if original_duration <= 0:
            raise RuntimeError(f"Invalid duration ({original_duration}) for video '{video_path}'")

        def make_segment(start: float, end: float):
            s = ffmpeg.input(video_path)
            s = s.filter("trim", start=start, end=end)
            s = s.filter("setpts", "PTS-STARTPTS")
            return s

        if video_config.clip_duration <= 0:
            seg_stream = ffmpeg.input(video_path)
            eff_dur = original_duration
        elif video_config.clip_duration <= original_duration:
            max_start = max(0.0, original_duration - video_config.clip_duration)
            start = random.uniform(0.0, max_start)
            seg_stream = make_segment(start, start + video_config.clip_duration)
            eff_dur = video_config.clip_duration
        else:
            full_copies = int(video_config.clip_duration // original_duration)
            remainder = video_config.clip_duration % original_duration
            segments = [make_segment(0.0, original_duration) for _ in range(full_copies)]
            if remainder > 0.01:
                segments.append(make_segment(0.0, remainder))
            seg_stream = ffmpeg.concat(*segments, v=1, a=0) if len(segments) > 1 else segments[0]
            eff_dur = video_config.clip_duration

        seg_stream = seg_stream.filter("fps", fps=video_config.fps).filter("settb", "AVTB")
        return self.apply_resize_filters(seg_stream, video_config), eff_dur

    def _prepare_chunk_streams(self, chunk_cfg: ChunkConfig, video_config: VideoConfig, trans_cfg: TransitionConfig):
        all_streams, all_durations = [], []

        if chunk_cfg.is_first_chunk and trans_cfg.fade_in_enabled:
            color_in = ffmpeg.input(f"color=c={trans_cfg.fade_in_color}:s={video_config.width}x{video_config.height}:r={video_config.fps}:d={trans_cfg.fade_in_duration}", f="lavfi")
            all_streams.append(color_in.filter("fps", fps=video_config.fps).filter("settb", "AVTB"))
            all_durations.append(float(trans_cfg.fade_in_duration))

        if chunk_cfg.prev_file:
            all_streams.append(ffmpeg.input(chunk_cfg.prev_file).filter("fps", fps=video_config.fps).filter("settb", "AVTB"))
            all_durations.append(float(chunk_cfg.prev_duration))

        for vf, orig_dur in chunk_cfg.chunk_segments:
            s, d = self.build_segment_stream(str(vf), video_config, orig_dur)
            all_streams.append(s)
            all_durations.append(float(d))

        if chunk_cfg.is_last_chunk_no_audio_trim and trans_cfg.fade_out_enabled:
            color_out = ffmpeg.input(f"color=c={trans_cfg.fade_out_color}:s={video_config.width}x{video_config.height}:r={video_config.fps}:d={trans_cfg.fade_out_duration}", f="lavfi")
            all_streams.append(color_out.filter("fps", fps=video_config.fps).filter("settb", "AVTB"))
            all_durations.append(float(trans_cfg.fade_out_duration))

        return all_streams, all_durations

    def _is_fade_boundary(self, ctx: XFadeContext):
        is_init = ctx.chunk_cfg.is_first_chunk and ctx.trans_cfg.fade_in_enabled and not ctx.chunk_cfg.prev_file and ctx.i == 1
        is_final = ctx.chunk_cfg.is_last_chunk_no_audio_trim and ctx.trans_cfg.fade_out_enabled and ctx.i == ctx.n - 1
        return is_init, is_final

    def _get_actual_xfade_duration(self, ctx: XFadeContext):
        is_init, is_final = self._is_fade_boundary(ctx)
        if is_init:
            return ctx.trans_cfg.fade_in_duration
        if is_final:
            return ctx.trans_cfg.fade_out_duration
        if ctx.trans_cfg.type == "fade" and ctx.chunk_cfg.playlist_length >= 2:
            return ctx.trans_cfg.duration
        return min(0.05, ctx.durations[ctx.i-1], ctx.durations[ctx.i])

    def _get_xfade_params(self, ctx: XFadeContext) -> Tuple[str, float]:
        prev_dur, cur_dur = ctx.durations[ctx.i-1], ctx.durations[ctx.i]
        actual = self._get_actual_xfade_duration(ctx)
        actual = min(actual, max(0.0, min(prev_dur, cur_dur) - 0.01))
        
        if actual <= 0:
            raise RuntimeError(f"Invalid transition duration {actual}")
        return "fade", actual

    def _apply_chunk_xfades(self, all_streams, all_durations, x_cfg: Tuple[TransitionConfig, ChunkConfig]):
        current = all_streams[0]
        offset_accum = 0.0
        n = len(all_streams)
        trans_cfg, chunk_cfg = x_cfg
        
        for i in range(1, n):
            prev_dur = all_durations[i-1]
            ctx = XFadeContext(i, n, all_durations, trans_cfg, chunk_cfg)
            trans, actual = self._get_xfade_params(ctx)

            offset = offset_accum + prev_dur - actual
            current = ffmpeg.filter([current, all_streams[i]], "xfade", transition=trans, duration=actual, offset=offset)
            offset_accum += prev_dur - actual
        return current

    def _execute_chunk_render(self, stream, video_config: VideoConfig, total_duration: float) -> RenderResult:
        tmp_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        out_kwargs = {"vcodec": "h264_nvenc"} if video_config.use_gpu else {}
        self.run_ffmpeg_with_progress(ffmpeg.output(stream, tmp_path, **out_kwargs).overwrite_output(), total_duration)
        return RenderResult(path=tmp_path, duration=self.get_duration(tmp_path))

    def render_chunk(self, chunk_cfg: ChunkConfig, video_config: VideoConfig, trans_cfg: TransitionConfig) -> RenderResult:
        all_streams, all_durations = self._prepare_chunk_streams(chunk_cfg, video_config, trans_cfg)
        current = self._apply_chunk_xfades(all_streams, all_durations, (trans_cfg, chunk_cfg))
        return self._execute_chunk_render(current, video_config, sum(all_durations))


class AudioProcessor:
    def __init__(self, video_processor: VideoProcessor):
        self.vp = video_processor

    def _normalize_waveform(self, waveform):
        if not isinstance(waveform, np.ndarray):
            waveform = waveform.cpu().numpy() if hasattr(waveform, "cpu") else np.array(waveform)
        
        if waveform.ndim > 2:
            waveform = waveform.squeeze()
        if waveform.ndim == 2 and waveform.shape[0] < waveform.shape[1]:
            waveform = waveform.T
        return waveform

    def _write_temp_audio(self, waveform, sample_rate):
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, waveform, int(sample_rate))
        return tmp.name, tmp

    def process_vhs_audio(self, audio_dict: dict) -> Tuple[Optional[str], Optional[tempfile._TemporaryFileWrapper]]:
        if not isinstance(audio_dict, dict):
            return None, None
        
        path = audio_dict.get("path") or audio_dict.get("output_path")
        if path and os.path.exists(path):
            return path, None

        waveform, sample_rate = audio_dict.get("waveform"), audio_dict.get("sample_rate")
        if waveform is None or sample_rate is None:
            return None, None

        waveform = self._normalize_waveform(waveform)
        return self._write_temp_audio(waveform, sample_rate)

    def get_duration(self, path: str) -> float:
        return self.vp.get_duration(path)

class ComfyVideoCombiner:
    """
    A ComfyUI-compatible node that:
    - Scans a directory for video files matching a pattern (e.g. *.mp4)
    - Sorts them or randomizes order
    - Optionally adds crossfade transitions between them
    - Optionally fades in from a color and/or fades out to a color
    - Optionally overlays a music track (VideoHelperSuite audio format)
    - Optionally uses GPU-accelerated encoding (NVENC) when available
    - Uses ProgressBar to show progress in the console
    """

    @classmethod
    def _video_base_params(cls):
        return {
            "directory_path": ("STRING", {"default": "", "multiline": False, "placeholder": "Path to video directory"}),
            "output_filename": ("STRING", {"default": "combined_output.mp4", "multiline": False, "placeholder": "output.mp4"}),
            "file_pattern": ("STRING", {"default": "*.mp4", "multiline": False, "placeholder": "*.mp4"}),
            "clip_duration": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 3600.0, "step": 0.1, "round": 0.1, "label": "Random segment length (s, 0=full video)"}),
            "width": ("INT", {"default": 1080, "min": 1, "max": 8192, "step": 1, "label": "Target width"}),
            "height": ("INT", {"default": 1920, "min": 1, "max": 8192, "step": 1, "label": "Target height"}),
            "resize_mode": (["crop", "padding"], {"default": "crop", "label": "Resize mode"}),
            "resize_padding_color": ("STRING", {"default": "#000000", "multiline": False, "placeholder": "#000000", "label": "Padding background color (hex)"}),
        }

    @classmethod
    def _transition_params(cls):
        return {
            "transition": (["none", "fade"], {"default": "fade"}),
            "transition_duration": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 2.0, "step": 0.1, "round": 0.1}),
        }

    @classmethod
    def _file_params(cls):
        return {
            "sort_files": ("BOOLEAN", {"default": False, "label": "Sort files alphabetically"}),
            "random_order": ("BOOLEAN", {"default": True, "label": "Randomize order (no repeats, overrides sort)"}),
            "seed": ("INT", {"default": -1, "min": -1, "max": 2**31 - 1, "step": 1, "label": "Random seed (-1 = random)"}),
        }

    @classmethod
    def _fade_params(cls):
        return {
            "fade_in_enabled": ("BOOLEAN", {"default": True, "label": "Enable fade in from color"}),
            "fade_in_color": ("STRING", {"default": "#000000", "multiline": False, "placeholder": "#000000", "label": "Fade in color (hex)"}),
            "fade_in_duration": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 10.0, "step": 0.1, "round": 0.1, "label": "Fade in duration (s)"}),
            "fade_out_enabled": ("BOOLEAN", {"default": True, "label": "Enable fade out to color"}),
            "fade_out_color": ("STRING", {"default": "#000000", "multiline": False, "placeholder": "#000000", "label": "Fade out color (hex)"}),
            "fade_out_duration": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 10.0, "step": 0.1, "round": 0.1, "label": "Fade out duration (s)"}),
        }

    @classmethod
    def INPUT_TYPES(cls):
        required = cls._video_base_params()
        required.update(cls._transition_params())
        
        optional = cls._file_params()
        optional.update({
            "music_track": ("AUDIO",),
            "trim_to_audio": ("BOOLEAN", {"default": True, "label": "Trim video to audio length"}),
        })
        optional.update(cls._fade_params())
        optional.update({
            "use_gpu": ("BOOLEAN", {"default": True, "label": "Use GPU (NVENC encoder)"}),
        })
        
        return {"required": required, "optional": optional}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "combine_videos"
    CATEGORY = "Simple Video Effects"
    OUTPUT_NODE = True

    def __init__(self):
        self.output_dir = self._get_default_output_directory()
        self.vp = VideoProcessor()
        self.ap = AudioProcessor(self.vp)

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

    def _prepare_video_files(self, cfg: FileConfig):
        directory = Path(cfg.directory_path).expanduser().resolve()
        if not directory.is_dir():
            raise ValueError(f"Directory does not exist: {directory}")

        video_files = sorted(directory.glob(cfg.file_pattern))
        if not video_files:
            raise ValueError(f"No video files found in '{directory}' matching '{cfg.file_pattern}'")

        if cfg.random_order:
            if cfg.seed != -1: random.seed(cfg.seed)
            random.shuffle(video_files)
        elif cfg.sort_files:
            video_files.sort()
        return video_files

    def _prepare_audio(self, audio_config: AudioConfig):
        temp_audio = None
        if audio_config.music_track is not None:
            audio_path, temp_audio = self.ap.process_vhs_audio(audio_config.music_track)
            audio_config.path = audio_path
            if audio_path and audio_config.trim_to_audio:
                try:
                    audio_config.duration = self.ap.get_duration(audio_path)
                except Exception as e:
                    print(f"Warning: could not probe audio duration: {e}")
        return temp_audio

    def _render_simple_video(self, video_files, ctx: RenderContext, output_path: str):
        try:
            durations = self.vp.compute_durations_parallel(video_files)
            total_dur = sum(durations)
        except Exception:
            total_dur = None

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for vf in video_files:
                path_str = str(vf.absolute()).replace("'", "'\\''")
                f.write(f"file '{path_str}'\n")
            temp_list_path = f.name

        try:
            stream = ffmpeg.input(temp_list_path, f="concat", safe=0).filter("fps", fps=ctx.video_cfg.fps)
            stream = self.vp.apply_resize_filters(stream, ctx.video_cfg)

            out_args = {"vcodec": "h264_nvenc"} if ctx.video_cfg.use_gpu else {}
            if ctx.audio_cfg.path:
                audio_stream = ffmpeg.input(ctx.audio_cfg.path)
                out_args["acodec"] = "aac"
                if ctx.audio_cfg.trim_to_audio:
                    out_args["shortest"] = None
                stream = ffmpeg.output(stream, audio_stream, output_path, **out_args)
            else:
                stream = ffmpeg.output(stream, output_path, **out_args)

            self.vp.run_ffmpeg_with_progress(stream.overwrite_output(), total_dur)
        finally:
            if os.path.exists(temp_list_path):
                os.remove(temp_list_path)

    def _estimate_playlist_duration(self, playlist, clip_duration, nom_xfade):
        count = len(playlist)
        cum_raw = sum(clip_duration if clip_duration > 0 else d for _, d in playlist)
        return cum_raw - max(0, count - 1) * nom_xfade

    def _select_random_clips(self, playlist, cfg: PlaylistConfig, nom_xfade):
        while len(playlist) < 100:
            idx = random.randrange(len(cfg.video_files))
            playlist.append((cfg.video_files[idx], cfg.original_durations[idx]))
            
            est = self._estimate_playlist_duration(playlist, cfg.clip_duration, nom_xfade)
            if est >= (cfg.audio_duration or 0):
                break
        return playlist

    def _prepare_playlist(self, cfg: PlaylistConfig, transition_config: TransitionConfig):
        orig_durs = cfg.original_durations
        nom_clip = cfg.clip_duration if cfg.clip_duration > 0 else sum(orig_durs) / max(1, len(orig_durs))
        nom_xfade = transition_config.duration if (transition_config.type == "fade" and len(cfg.video_files) >= 2) else min(0.05, nom_clip)

        should_trim = cfg.audio_duration and cfg.trim_to_audio
        if should_trim:
            return self._select_random_clips([], cfg, nom_xfade)
        
        playlist = list(zip(cfg.video_files, cfg.original_durations))
        if cfg.audio_duration and self._estimate_playlist_duration(playlist, cfg.clip_duration, nom_xfade) < cfg.audio_duration:
            return self._select_random_clips(playlist, cfg, nom_xfade)
        
        return playlist

    def _get_fade_out_filter(self, video_in, ctx: RenderContext):
        audio_ready = ctx.audio_cfg.path and ctx.audio_cfg.trim_to_audio and ctx.audio_cfg.duration
        needs_audio_fade_out = ctx.trans_cfg.fade_out_enabled and audio_ready
        
        if needs_audio_fade_out:
            fade_dur = min(ctx.trans_cfg.fade_out_duration, max(0.01, ctx.audio_cfg.duration - 0.01))
            start_time = max(0.0, ctx.audio_cfg.duration - fade_dur)
            return video_in.filter("fade", type="out", start_time=start_time, duration=fade_dur, color=ctx.trans_cfg.fade_out_color)
        return video_in

    def _apply_final_filters_and_audio(self, render_res: RenderResult, ctx: RenderContext, output_path: str):
        video_in = self._get_fade_out_filter(ffmpeg.input(render_res.path), ctx)
        out_args = {"vcodec": "h264_nvenc"} if ctx.video_cfg.use_gpu else {}
        
        if ctx.audio_cfg.path:
            out_args["acodec"] = "aac"
            if ctx.audio_cfg.trim_to_audio:
                out_args["shortest"] = None
            stream = ffmpeg.output(video_in, ffmpeg.input(ctx.audio_cfg.path), output_path, **out_args)
            total_dur = ctx.audio_cfg.duration or render_res.duration
        else:
            stream = ffmpeg.output(video_in, output_path, **out_args)
            total_dur = render_res.duration

        self.vp.run_ffmpeg_with_progress(stream.overwrite_output(), total_dur)

    def _process_chunks(self, playlist, ctx: RenderContext, pbar) -> RenderResult:
        current_res = RenderResult(path="", duration=0.0)
        CHUNK_SIZE = 12
        for idx in range(0, len(playlist), CHUNK_SIZE):
            chunk = playlist[idx: idx + CHUNK_SIZE]
            is_last = (idx + CHUNK_SIZE >= len(playlist))
            no_trim = not (ctx.audio_cfg.path and ctx.audio_cfg.trim_to_audio)
            
            chunk_cfg = ChunkConfig(
                prev_file=current_res.path if current_res.path else None,
                prev_duration=current_res.duration,
                chunk_segments=chunk,
                is_first_chunk=(not current_res.path),
                is_last_chunk_no_audio_trim=(is_last and no_trim),
                playlist_length=len(playlist)
            )
            
            new_res = self.vp.render_chunk(chunk_cfg, ctx.video_cfg, ctx.trans_cfg)
            
            if current_res.path and os.path.exists(current_res.path):
                os.remove(current_res.path)
            current_res = new_res
            pbar.update(len(chunk))
        return current_res

    def _render_advanced_video(self, video_files, ctx: RenderContext, output_path: str):
        orig_durs = self.vp.compute_durations_parallel(video_files)
        p_cfg = PlaylistConfig(video_files, orig_durs, ctx.video_cfg.clip_duration, ctx.audio_cfg.duration, ctx.audio_cfg.trim_to_audio)
        playlist = self._prepare_playlist(p_cfg, ctx.trans_cfg)
        
        pbar = comfy.utils.ProgressBar(len(playlist))
        final_res = None
        try:
            final_res = self._process_chunks(playlist, ctx, pbar)
            if not final_res or not final_res.path:
                raise RuntimeError("Advanced rendering failed: no output path")

            self._apply_final_filters_and_audio(final_res, ctx, output_path)
        finally:
            needs_cleanup = final_res and final_res.path and os.path.exists(final_res.path)
            if needs_cleanup:
                os.remove(final_res.path)

    def _get_output_path(self, filename):
        filename = (filename.strip() or "combined_output.mp4")
        if not filename.lower().endswith(".mp4"):
            filename += ".mp4"
        return self.get_unique_filename(os.path.join(self.output_dir, filename))

    def _build_configs(self, video_files, kwargs):
        f_w, f_h, base_fps = self.vp.get_resolution_and_fps(str(video_files[0]))
        v_cfg = VideoConfig(
            kwargs.get("width") or f_w, kwargs.get("height") or f_h, 
            kwargs.get("resize_mode"), kwargs.get("resize_padding_color"), 
            kwargs.get("use_gpu"), kwargs.get("clip_duration"), base_fps
        )
        t_cfg = TransitionConfig(
            kwargs.get("transition"), kwargs.get("transition_duration"), 
            kwargs.get("fade_in_enabled"), self.vp.sanitize_hex_color(kwargs.get("fade_in_color")), 
            kwargs.get("fade_in_duration"), kwargs.get("fade_out_enabled"), 
            self.vp.sanitize_hex_color(kwargs.get("fade_out_color")), kwargs.get("fade_out_duration")
        )
        return v_cfg, t_cfg

    def _is_advanced_render_needed(self, video_files, video_config: VideoConfig, transition_config: TransitionConfig) -> bool:
        has_fade = transition_config.type == "fade" and len(video_files) >= 2
        has_fade_in = transition_config.fade_in_enabled
        has_fade_out = transition_config.fade_out_enabled
        has_clip_dur = video_config.clip_duration > 0
        return has_fade or has_fade_in or has_fade_out or has_clip_dur

    def combine_videos(self, directory_path, output_filename, file_pattern, **kwargs):
        f_cfg = FileConfig(directory_path, file_pattern, kwargs.get("sort_files", True), kwargs.get("random_order", True), kwargs.get("seed", -1))
        video_files = self._prepare_video_files(f_cfg)
        
        audio_config = AudioConfig(kwargs.get("music_track"), kwargs.get("trim_to_audio", True))
        temp_audio = self._prepare_audio(audio_config)
        output_path = self._get_output_path(output_filename)
        video_config, transition_config = self._build_configs(video_files, kwargs)

        print(f"Target: {video_config.width}x{video_config.height} @ {video_config.fps:.3f} fps, GPU: {video_config.use_gpu}")

        ctx = RenderContext(video_config, audio_config, transition_config)

        try:
            if not self._is_advanced_render_needed(video_files, video_config, transition_config):
                self._render_simple_video(video_files, ctx, output_path)
            else:
                self._render_advanced_video(video_files, ctx, output_path)
        except Exception as e:
            raise RuntimeError(f"FFmpeg error: {str(e)}")
        finally:
            if temp_audio:
                temp_audio.close()
                if os.path.exists(temp_audio.name): os.remove(temp_audio.name)

        return (output_path,)


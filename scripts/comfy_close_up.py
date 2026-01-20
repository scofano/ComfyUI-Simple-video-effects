import os
import shutil
import subprocess
import tempfile
from pathlib import Path
import torch
import torch.nn.functional as F

# ComfyUI folder_paths
try:
    import folder_paths
    OUTPUT_DIR = folder_paths.get_output_directory()
except ImportError:
    OUTPUT_DIR = Path("output")

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
if not FFMPEG or not FFPROBE:
    raise RuntimeError("ffmpeg and ffprobe not found on PATH.")

def get_video_info(video_path):
    """Get video duration, fps, width, height"""
    cmd = [
        FFPROBE,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    import json
    data = json.loads(result.stdout)

    duration = float(data['format']['duration'])

    # Find video stream
    video_stream = None
    for stream in data['streams']:
        if stream['codec_type'] == 'video':
            video_stream = stream
            break

    if video_stream:
        fps_str = video_stream.get('r_frame_rate', '30/1')
        if '/' in fps_str:
            num, den = fps_str.split('/')
            fps = float(num) / float(den)
        else:
            fps = float(fps_str)
        width = int(video_stream['width'])
        height = int(video_stream['height'])
    else:
        fps = 30.0
        width = 1920
        height = 1080

    return duration, fps, width, height


def extract_frames(video_path, output_dir, fps):
    """Extract frames from video"""
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    cmd = [
        FFMPEG,
        "-i", video_path,
        "-vf", f"fps={fps}",
        str(frames_dir / "frame_%06d.png")
    ]
    subprocess.run(cmd, check=True)

    return frames_dir


def load_frames(frames_dir):
    """Load frames as torch tensor"""
    from PIL import Image
    import numpy as np

    frame_files = sorted(frames_dir.glob("frame_*.png"))
    if not frame_files:
        raise ValueError("No frames extracted")

    frames = []
    for frame_file in frame_files:
        img = Image.open(frame_file).convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        frames.append(torch.from_numpy(arr))

    if not frames:
        raise ValueError("No frames loaded")

    # Stack to (B, H, W, C)
    images = torch.stack(frames, dim=0)
    return images


def calculate_face_center(segs):
    """Calculate center point between eyes from SEGS data"""
    if not segs:
        raise ValueError("No segmentation data provided")

    # SEGS format: ((width, height), [SEG objects])
    if isinstance(segs, (list, tuple)) and len(segs) == 2:
        # Unpack the tuple
        dims, seg_list = segs
        segs = seg_list

    # Handle different SEGS formats
    valid_eyes = []

    for seg in segs:
        # Try different SEGS formats
        if hasattr(seg, 'confidence') and hasattr(seg, 'label') and hasattr(seg, 'bbox'):
            # Format: SEG objects with attributes
            confidence = seg.confidence[0] if hasattr(seg.confidence, '__len__') else seg.confidence
            if confidence > 0.4 and seg.label == 'eye':
                valid_eyes.append(seg.bbox)
        elif isinstance(seg, (list, tuple)) and len(seg) >= 3:
            # Format: tuples like (bbox, label, confidence)
            bbox, label, confidence = seg[0], seg[1], seg[2]
            confidence = confidence[0] if hasattr(confidence, '__len__') else confidence
            if confidence > 0.4 and label == 'eye':
                valid_eyes.append(bbox)
        else:
            # Unknown format, skip
            continue

    if len(valid_eyes) < 2:
        raise ValueError(f"Need at least 2 eyes with confidence > 0.4, found {len(valid_eyes)}")

    # Take first two eyes
    eye1 = valid_eyes[0]
    eye2 = valid_eyes[1]

    # Calculate eye centers
    x1 = (eye1[0] + eye1[2]) / 2
    y1 = (eye1[1] + eye1[3]) / 2

    x2 = (eye2[0] + eye2[2]) / 2
    y2 = (eye2[1] + eye2[3]) / 2

    # Face center is midpoint between eyes
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    return center_x, center_y


# ---------- NODE --------------------------------------------------------------
class CloseUpNode:
    """
    Close-up effect for video files centered on face between eyes.

    Takes a video file and SEGS data to detect eyes, calculates face center,
    and applies zoom centered on that point.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": ""}),
                "segs": ("SEGS",),
                "zoom_factor": ("FLOAT", {"default": 1.5, "min": 1.0, "step": 0.1}),
                "prefix": ("STRING", {"default": "close_up"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "apply_close_up"
    CATEGORY = "Simple Video Effects"
    OUTPUT_NODE = True

    # ---- main ---------------------------------------------------------------
    def apply_close_up(self, video_path: str, segs, zoom_factor: float, prefix: str):

        if not os.path.exists(video_path):
            raise ValueError(f"Video file not found: {video_path}")

        # Calculate face center from SEGS
        try:
            center_x, center_y = calculate_face_center(segs)
        except Exception as e:
            raise ValueError(f"Failed to calculate face center: {e}")

        # Get video info
        duration, fps, width, height = get_video_info(video_path)

        # Create output folder
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        counter = 1
        while True:
            filename = f"{prefix}_{counter:03d}.mp4"
            output_path = output_dir / filename
            if not output_path.exists():
                break
            counter += 1

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Extract frames
            frames_dir = extract_frames(video_path, tmpdir, fps)
            images = load_frames(frames_dir)

            # Apply close-up zoom
            zoomed_images = self._apply_zoom_to_center(images, zoom_factor, center_x, center_y, width, height)

            # Save processed frames
            processed_frames_dir = tmpdir / "processed_frames"
            processed_frames_dir.mkdir()

            for i, img_tensor in enumerate(zoomed_images):
                # Convert to PIL
                arr = (img_tensor.clamp(0, 1) * 255).byte().numpy()
                from PIL import Image
                img = Image.fromarray(arr)
                img.save(processed_frames_dir / f"frame_{i:06d}.png")

            # Encode to video with audio
            cmd = [
                FFMPEG,
                "-y",
                "-framerate", str(fps),
                "-i", str(processed_frames_dir / "frame_%06d.png"),
                "-i", video_path,  # for audio
                "-c:v", "libx264",
                "-c:a", "copy",  # copy audio if exists
                "-pix_fmt", "yuv420p",
                "-map", "0:v:0",  # video from frames
                "-map", "1:a?",  # audio from original (optional)
                str(output_path)
            ]
            subprocess.run(cmd, check=True)

        return (str(output_path),)

    def _apply_zoom_to_center(self, images, zoom_factor, center_x, center_y, orig_width, orig_height):
        """Apply zoom centered on the specified point"""
        if images.ndim != 4:
            return images

        B, H, W, C = images.shape
        if B <= 0:
            return images

        # Calculate crop region for zoom
        # We want to zoom by zoom_factor, centered on (center_x, center_y)
        crop_width = int(round(W / zoom_factor))
        crop_height = int(round(H / zoom_factor))

        # Ensure crop dimensions don't exceed original
        crop_width = min(crop_width, W)
        crop_height = min(crop_height, H)

        # Calculate crop position centered on face center
        crop_x = int(round(center_x - crop_width / 2))
        crop_y = int(round(center_y - crop_height / 2))

        # Clamp to valid bounds
        crop_x = max(0, min(crop_x, W - crop_width))
        crop_y = max(0, min(crop_y, H - crop_height))

        # Apply crop and resize back to original dimensions
        out_frames = []
        for i in range(B):
            frame = images[i]  # (H, W, C)

            # Crop
            cropped = frame[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width, :]

            # Resize back to original size
            cropped_tensor = cropped.permute(2, 0, 1).unsqueeze(0)  # (1, C, crop_H, crop_W)
            resized = F.interpolate(cropped_tensor, size=(H, W), mode="bicubic", align_corners=False)
            resized_frame = resized.squeeze(0).permute(1, 2, 0)  # back to (H, W, C)

            out_frames.append(resized_frame)

        return torch.stack(out_frames, dim=0)

import os
import tempfile
from pathlib import Path
import torch
import numpy as np
from PIL import Image, ImageDraw
import subprocess
import shutil

# tqdm for progress
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, *args, **kwargs):
        return x

# ComfyUI folder_paths
try:
    import folder_paths
    OUTPUT_DIR = folder_paths.get_output_directory()
except ImportError:
    OUTPUT_DIR = Path("output")

FFMPEG = shutil.which("ffmpeg")
if not FFMPEG:
    raise RuntimeError("ffmpeg not found on PATH.")

def tensor_to_pil(tensor):
    """Convert ComfyUI IMAGE tensor [H,W,C] to PIL Image"""
    arr = (tensor.clamp(0, 1) * 255).byte().numpy()
    return Image.fromarray(arr).convert("RGB")

def pil_to_tensor(pil_img):
    """Convert PIL to ComfyUI IMAGE tensor [H,W,C]"""
    arr = np.array(pil_img).astype(np.float32) / 255.0
    return torch.from_numpy(arr)

def hex_to_rgb(hex_color):
    """Convert hex string to RGB tuple"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

class ImageTransitionNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "duration": ("FLOAT", {"default": 1.0, "min": 0.1, "step": 0.1}),
                "direction": (["Vertical - Down", "Vertical - Up", "Horizontal - Left", "Horizontal - Right"], {"default": "Horizontal - Right"}),
                "line_toggle": ("BOOLEAN", {"default": False}),
                "thickness": ("INT", {"default": 2, "min": 1, "max": 50}),
                "hex_color": ("STRING", {"default": "#FFFFFF"}),
                "prefix": ("STRING", {"default": "transition"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "create_transition"
    CATEGORY = "Simple Video Effects"
    OUTPUT_NODE = True
    # OUTPUT_IS_LIST = False
    # OUTPUT_NODE = False

    OPTIONAL_OUTPUTS = ("output_path",)

    def create_transition(self, image1, image2, duration, direction, line_toggle, thickness, hex_color, prefix, output_path=None):
        # Extract single frames
        if image1.shape[0] != 1 or image2.shape[0] != 1:
            raise ValueError("Inputs must be single images, not batches.")
        img1_pil = tensor_to_pil(image1[0])
        img2_pil = tensor_to_pil(image2[0])

        # Ensure same size and even dimensions for video encoding
        width, height = img1_pil.size
        # Make dimensions even
        width = width if width % 2 == 0 else width - 1
        height = height if height % 2 == 0 else height - 1
        if img1_pil.size != (width, height):
            img1_pil = img1_pil.resize((width, height))
        if img2_pil.size != (width, height):
            img2_pil = img2_pil.resize((width, height))

        width, height = img1_pil.size
        fps = 30
        # Generate at higher frame rate for smoother animation
        gen_fps = 60
        total_frames = int(duration * gen_fps)
        if total_frames < 1:
            total_frames = 1

        # Prepare color
        try:
            line_color = hex_to_rgb(hex_color)
        except:
            line_color = (255, 255, 255)

        # Output dir
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Filename with increment
        if not prefix:
            prefix = "transition"
        counter = 1
        while True:
            filename = f"{prefix}_{counter:03d}.mp4"
            output_path = output_dir / filename
            if not output_path.exists():
                break
            counter += 1

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            frames_dir = tmpdir / "frames"
            frames_dir.mkdir()

            for i in tqdm(range(total_frames), desc="Generating frames"):
                progress = i / (total_frames - 1) if total_frames > 1 else 1.0

                # Create mask: black for img1, white for img2
                mask = Image.new("L", (width, height), 0)  # 0 = show img1
                draw_mask = ImageDraw.Draw(mask)

                if direction == "Horizontal - Right":
                    reveal_x = int(progress * width)
                    draw_mask.rectangle([0, 0, reveal_x, height], fill=255)
                elif direction == "Horizontal - Left":
                    reveal_x = int(progress * width)
                    draw_mask.rectangle([width - reveal_x, 0, width, height], fill=255)
                elif direction == "Vertical - Down":
                    reveal_y = int(progress * height)
                    draw_mask.rectangle([0, 0, width, reveal_y], fill=255)
                elif direction == "Vertical - Up":
                    reveal_y = int(progress * height)
                    draw_mask.rectangle([0, height - reveal_y, width, height], fill=255)

                # Composite images
                frame = Image.composite(img2_pil, img1_pil, mask)

                # Draw line if enabled
                if line_toggle and progress < 1.0:
                    draw = ImageDraw.Draw(frame)
                    if direction == "Horizontal - Right":
                        x = int(progress * (width - thickness))
                        draw.line([x, 0, x, height], fill=line_color, width=thickness)
                    elif direction == "Horizontal - Left":
                        x = int(width - progress * (width - thickness))
                        draw.line([x, 0, x, height], fill=line_color, width=thickness)
                    elif direction == "Vertical - Down":
                        y = int(progress * (height - thickness))
                        draw.line([0, y, width, y], fill=line_color, width=thickness)
                    elif direction == "Vertical - Up":
                        y = int(height - progress * (height - thickness))
                        draw.line([0, y, width, y], fill=line_color, width=thickness)

                # Save frame
                frame.save(frames_dir / f"frame_{i:06d}.png")

            # Encode to MP4
            cmd = [
                FFMPEG,
                "-y",
                "-framerate", str(fps),
                "-i", str(frames_dir / "frame_%06d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                str(output_path)
            ]
            subprocess.run(cmd, check=True)

        return (str(output_path),)

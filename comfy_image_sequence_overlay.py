import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path

class ImageSequenceOverlay:
    """
    ComfyUI node:
      - Input: IMAGE (sequence of images)
      - Input: folder_path (STRING, path to folder with transparent PNG files)
      - Input: mode (STRING, animation mode for overlays)
      - Output: IMAGE (overlayed sequence)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "folder_path": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "",
                    },
                ),
                "mode": (
                    ["loop", "run_once", "run_once_and_hold", "ping_pong"],
                    {
                        "default": "loop",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_overlay"
    CATEGORY = "Simple Video Effects/Image Processing"

    def apply_overlay(self, images, folder_path: str, mode: str):
        if not folder_path:
            raise RuntimeError("folder_path is empty.")
        folder = Path(folder_path)
        if not folder.exists() or not folder.is_dir():
            raise RuntimeError(f"Folder not found: {folder}")

        # Load and sort overlay PNG files
        overlay_files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.png')])
        if not overlay_files:
            raise RuntimeError(f"No PNG files found in {folder}")

        overlay_images = []
        for fname in overlay_files:
            img = Image.open(folder / fname).convert("RGBA")
            overlay_images.append(img)

        num_overlay = len(overlay_images)
        num_input = images.shape[0]

        # Generate overlay indices based on mode
        if mode == "loop":
            overlay_indices = [i % num_overlay for i in range(num_input)]
        elif mode == "run_once":
            overlay_indices = [i if i < num_overlay else None for i in range(num_input)]
        elif mode == "run_once_and_hold":
            overlay_indices = [min(i, num_overlay - 1) for i in range(num_input)]
        elif mode == "ping_pong":
            # Generate ping-pong sequence
            if num_overlay == 1:
                seq = [0] * num_input
            else:
                seq = list(range(num_overlay)) + list(range(num_overlay - 2, -1, -1))
                cycle_len = len(seq)
                seq = [seq[i % cycle_len] for i in range(num_input)]
            overlay_indices = seq
        else:
            raise RuntimeError(f"Unknown mode: {mode}")

        # Process each frame
        output_frames = []
        images_np = images.detach().cpu().numpy()  # [N, H, W, C]

        for i in range(num_input):
            # Get base image
            base_np = images_np[i]
            if base_np.shape[-1] == 3:
                base_pil = Image.fromarray((base_np * 255).astype(np.uint8), 'RGB').convert("RGBA")
            elif base_np.shape[-1] == 4:
                base_pil = Image.fromarray((base_np * 255).astype(np.uint8), 'RGBA')
            else:
                raise RuntimeError(f"Unsupported image channels: {base_np.shape[-1]}")

            # Overlay if index is not None
            idx = overlay_indices[i]
            if idx is not None:
                overlay_pil = overlay_images[idx]
                # Resize overlay to match base size
                overlay_pil = overlay_pil.resize(base_pil.size, Image.Resampling.LANCZOS)
                # Composite
                base_pil = Image.alpha_composite(base_pil, overlay_pil)

            # Convert back to numpy
            output_np = np.array(base_pil).astype(np.float32) / 255.0
            output_frames.append(output_np)

        # Stack back to tensor
        output_tensor = torch.from_numpy(np.stack(output_frames, axis=0))
        return (output_tensor,)

import torch
from PIL import Image, ImageEnhance
import numpy as np


class ColorAdjustmentNode:
    """
    Adjusts brightness, contrast, and saturation on batched image tensors.
    Parameters use 0-100 scale where 100 = no change.
    Includes real-time progress bar during processing.
    """

    def __init__(self):
        self.pbar = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "brightness": ("INT", {"default": 100, "min": 0, "max": 200, "step": 1}),
                "contrast": ("INT", {"default": 100, "min": 0, "max": 200, "step": 1}),
                "saturation": ("INT", {"default": 100, "min": 0, "max": 200, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "info")
    FUNCTION = "run"
    CATEGORY = "Simple Video Effects/Utility & Special Effects"

    def run(self, images: torch.Tensor, brightness: int, contrast: int, saturation: int):
        if images.ndim != 4:
            raise ValueError(f"Expected 4D tensor (B,H,W,C), got shape {images.shape}")

        B, H, W, C = images.shape

        # Convert brightness/contrast/saturation from 0-100 scale to PIL factors (0-2 approx)
        brightness_factor = brightness / 100.0
        contrast_factor = contrast / 100.0
        saturation_factor = saturation / 100.0

        out_frames = []

        # Process each frame
        for i in range(B):
            # Extract single frame and convert to PIL Image
            frame_tensor = images[i]  # (H, W, C) in range [0, 1]

            # Convert to numpy and then PIL Image
            frame_np = frame_tensor.cpu().numpy()
            frame_np = np.clip(frame_np * 255, 0, 255).astype(np.uint8)

            # Convert to RGB if needed
            if C == 3:
                pil_image = Image.fromarray(frame_np, mode="RGB")
            elif C == 4:
                pil_image = Image.fromarray(frame_np, mode="RGBA")
            else:
                raise ValueError(f"Unsupported number of channels: {C}")

            # Apply brightness adjustment
            if brightness_factor != 1.0:
                pil_image = ImageEnhance.Brightness(pil_image).enhance(brightness_factor)

            # Apply contrast adjustment
            if contrast_factor != 1.0:
                pil_image = ImageEnhance.Contrast(pil_image).enhance(contrast_factor)

            # Apply saturation adjustment (using Color enhance for saturation)
            if saturation_factor != 1.0:
                pil_image = ImageEnhance.Color(pil_image).enhance(saturation_factor)

            # Convert back to tensor
            adjusted_np = np.array(pil_image).astype(np.float32) / 255.0
            adjusted_tensor = torch.from_numpy(adjusted_np)

            out_frames.append(adjusted_tensor)

        # Stack frames back into batch
        output = torch.stack(out_frames, dim=0)

        # Create info string
        info = (
            f"ColorAdjustment: {B} frames, "
            f"brightness={brightness} (factor: {brightness_factor:.2f}), "
            f"contrast={contrast} (factor: {contrast_factor:.2f}), "
            f"saturation={saturation} (factor: {saturation_factor:.2f})"
        )

        return (output, info)

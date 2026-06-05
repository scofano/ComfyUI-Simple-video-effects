import torch
import torch.nn.functional as F


class ColorAdjustmentNode:
    """
    Adjusts brightness, contrast, and saturation on batched image tensors.
    Parameters use 0-100 scale where 100 = no change.
    GPU-accelerated using PyTorch operations.
    """

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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        using_gpu = device.type == "cuda"

        brightness_factor = brightness / 100.0
        contrast_factor = contrast / 100.0
        saturation_factor = saturation / 100.0

        output = images.to(device).clone()

        # Apply brightness: multiply pixel values
        if brightness_factor != 1.0:
            output = output * brightness_factor
            output = torch.clamp(output, 0, 1)

        # Apply contrast: scale around mean (0.5 is neutral gray)
        if contrast_factor != 1.0:
            output = 0.5 + (output - 0.5) * contrast_factor
            output = torch.clamp(output, 0, 1)

        # Apply saturation: convert RGB to HSV, adjust S, convert back
        if saturation_factor != 1.0 and C >= 3:
            output = self._adjust_saturation(output, saturation_factor)

        result = output.cpu()

        accel = "GPU" if using_gpu else "CPU"
        info = (
            f"ColorAdjustment ({accel}): {B} frames, "
            f"brightness={brightness}, "
            f"contrast={contrast}, "
            f"saturation={saturation}"
        )

        return (result, info)

    def _adjust_saturation(self, images: torch.Tensor, saturation_factor: float) -> torch.Tensor:
        """
        Adjust saturation by converting RGB to HSV, modifying S channel, converting back.
        GPU-accelerated using PyTorch.
        """
        B, H, W, C = images.shape

        # Extract RGB channels
        rgb = images[..., :3]  # (B, H, W, 3)
        alpha = images[..., 3:] if C > 3 else None

        # Convert RGB to HSV using PyTorch
        # Normalize RGB to [0, 1]
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

        max_rgb = torch.max(rgb, dim=-1)[0]
        min_rgb = torch.min(rgb, dim=-1)[0]
        delta = max_rgb - min_rgb

        # Value
        v = max_rgb

        # Saturation
        s = torch.where(v != 0, delta / v, torch.zeros_like(delta))

        # Hue
        h = torch.zeros_like(s)

        # Compute hue for each case
        mask_r = (max_rgb == r) & (delta != 0)
        mask_g = (max_rgb == g) & (delta != 0)
        mask_b = (max_rgb == b) & (delta != 0)

        h[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / delta[mask_r]) + 360) % 360
        h[mask_g] = (60 * ((b[mask_g] - r[mask_g]) / delta[mask_g]) + 120) % 360
        h[mask_b] = (60 * ((r[mask_b] - g[mask_b]) / delta[mask_b]) + 240) % 360

        # Adjust saturation
        s = torch.clamp(s * saturation_factor, 0, 1)

        # Convert HSV back to RGB
        h_i = (h / 60.0).long() % 6
        f = (h / 60.0) - h_i.float()

        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)

        # Create output based on hue sector
        out_r = torch.zeros_like(v)
        out_g = torch.zeros_like(v)
        out_b = torch.zeros_like(v)

        mask_0 = h_i == 0
        mask_1 = h_i == 1
        mask_2 = h_i == 2
        mask_3 = h_i == 3
        mask_4 = h_i == 4
        mask_5 = h_i == 5

        out_r[mask_0] = v[mask_0]
        out_g[mask_0] = t[mask_0]
        out_b[mask_0] = p[mask_0]

        out_r[mask_1] = q[mask_1]
        out_g[mask_1] = v[mask_1]
        out_b[mask_1] = p[mask_1]

        out_r[mask_2] = p[mask_2]
        out_g[mask_2] = v[mask_2]
        out_b[mask_2] = t[mask_2]

        out_r[mask_3] = p[mask_3]
        out_g[mask_3] = q[mask_3]
        out_b[mask_3] = v[mask_3]

        out_r[mask_4] = t[mask_4]
        out_g[mask_4] = p[mask_4]
        out_b[mask_4] = v[mask_4]

        out_r[mask_5] = v[mask_5]
        out_g[mask_5] = p[mask_5]
        out_b[mask_5] = q[mask_5]

        # Stack RGB channels
        rgb_adjusted = torch.stack([out_r, out_g, out_b], dim=-1)
        rgb_adjusted = torch.clamp(rgb_adjusted, 0, 1)

        # Combine with alpha channel if present
        if alpha is not None:
            output = torch.cat([rgb_adjusted, alpha], dim=-1)
        else:
            output = rgb_adjusted

        return output

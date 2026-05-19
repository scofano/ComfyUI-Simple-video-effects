# comfy_zoom_sequence.py
import time

import torch
import torch.nn.functional as F
from .zoom_core import (
    apply_center_zoom_subpixel,
    aspect_corrected_crop_box,
    compute_zoom_margins,
    margin_to_zoom_factor,
    resolve_direction_mode,
    resolve_ease_mode,
    resolve_seed_value,
)

# ---------- NODE --------------------------------------------------------------
class ZoomSequenceNode:
    """
    Zooms IN or OUT across a batched IMAGE (video/sequence). The canvas size stays fixed and
    the original aspect ratio is preserved for every frame.

    Parameters
    ----------
    direction: "Zoom In" or "Zoom Out"
    amount_type: "Pixels per Frame" or "Target Percentage"
    pixels_per_frame (FLOAT): speed when amount_type = Pixels per Frame.
    zoom_percentage (INT): target zoom when amount_type = Target Percentage (e.g. 110 = 110%).
    ease: "Linear", "Ease_In", "Ease_Out", "Ease_In_Out"
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "direction": (["Zoom In", "Zoom Out", "Random"], {"default": "Zoom In"}),
                "amount_type": (["Pixels per Frame", "Target Percentage"], {"default": "Pixels per Frame"}),
                "pixels_per_frame": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.1}),
                "zoom_percentage": ("INT", {"default": 110, "min": 100, "max": 10000, "step": 1}),
                "ease": (["Linear", "Ease_In", "Ease_Out", "Ease_In_Out", "Random"], {"default": "Linear"}),
                "random_seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
                "smooth_subpixel": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("images", "info",)
    FUNCTION = "run"
    CATEGORY = "Simple Video Effects/Image Processing"

    @classmethod
    def IS_CHANGED(
        cls,
        images,
        direction,
        amount_type,
        pixels_per_frame,
        zoom_percentage,
        ease,
        random_seed,
        smooth_subpixel,
    ):
        random_mode = (
            str(direction).strip().lower() == "random"
            or str(ease).strip().upper().replace("-", "_") == "RANDOM"
        )
        if random_mode and int(random_seed) == 0:
            # Force re-execution each run when using auto-seed random mode.
            # A monotonic nonce is more explicit/reliable than NaN for cache invalidation.
            return ("AUTO_RANDOM_NONCE", time.time_ns())
        return (
            str(direction),
            str(amount_type),
            float(pixels_per_frame),
            int(zoom_percentage),
            str(ease),
            int(random_seed),
            bool(smooth_subpixel),
        )

    # ---- helpers -------------------------------------------------------------
    def _resize_to(self, frame_hwc: torch.Tensor, size_hw):
        H, W = size_hw
        device = frame_hwc.device
        x = frame_hwc.permute(2, 0, 1).unsqueeze(0)  # 1,C,H,W
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        return x.squeeze(0).permute(1, 2, 0)

    # ---- main ---------------------------------------------------------------
    def run(
        self,
        images: torch.Tensor,
        direction: str,
        amount_type: str,
        pixels_per_frame: float,
        zoom_percentage: int,
        ease: str,
        random_seed: int,
        smooth_subpixel: bool,
    ):
        if images.ndim != 4:
            return (images, "Input is not a batched IMAGE (B,H,W,C).")
        B, H, W, C = images.shape
        if B <= 0:
            return (images, "Empty batch; nothing to do.")

        # GPU acceleration: keep tensors on their device (GPU if available)
        device = images.device
        images = images.to(device)

        import comfy.utils
        pbar = comfy.utils.ProgressBar(B)

        seed_i, auto_seed = resolve_seed_value(random_seed)
        effective_direction, rolled_direction = resolve_direction_mode(direction, seed_i)
        effective_ease, rolled_random = resolve_ease_mode(ease, seed_i + 1)

        small = min(W, H)
        m_smalls_f, meta = compute_zoom_margins(
            frame_count=B,
            small_dim=small,
            direction=effective_direction,
            amount_type=amount_type,
            pixels_per_frame=pixels_per_frame,
            zoom_percentage=zoom_percentage,
            ease=effective_ease,
            timeline_start=0,
            timeline_total=B,
        )

        # Apply proportional, aspect-corrected crops
        out_frames = []
        clamped = bool(meta.get("clamped", False))
        for i in range(B):
            pbar.update(1)
            if smooth_subpixel:
                zf = margin_to_zoom_factor(m_smalls_f[i], small)
                out_frames.append(apply_center_zoom_subpixel(images[i], zf, mode="bilinear"))
            else:
                # Round to integer ONLY where slicing happens
                m_small_int = int(round(m_smalls_f[i]))

                x0, y0, x1, y1, _, _ = aspect_corrected_crop_box(W, H, m_small_int)

                if x1 <= x0 or y1 <= y0:
                    out_frames.append(images[i])
                    continue

                cropped = images[i, y0:y1, x0:x1, :]
                resized = self._resize_to(cropped, (H, W))
                out_frames.append(resized)

        out = torch.stack(out_frames, dim=0)

        info_lines = []
        info_lines.append(
            f"Frames: {B}, Canvas: {W}x{H}, Direction: {effective_direction}, Amount Type: {amount_type}, Ease: {effective_ease}"
        )
        if rolled_direction:
            info_lines.append(f"Direction Random roll selected: {effective_direction}")
        if rolled_random:
            info_lines.append(f"Ease Random roll selected:      {effective_ease}")
        if rolled_direction or rolled_random:
            info_lines.append(
                f"Random seed:                   {seed_i}{' (auto from 0)' if auto_seed else ''}"
            )
        info_lines.append(
            f"Requested small-dim max margin: {meta['requested_small_margin_max_f']:.2f} px"
        )
        info_lines.append(
            f"Applied small-dim max margin:   {meta['applied_small_margin_max_f']:.2f} px "
            f"(safe limit: {meta['max_safe_small_margin']} px)"
        )
        info_lines.append(f"Effective max zoom reached:     {meta['effective_max_zoom']:.4f}x")
        info_lines.append(f"Transform mode:                 {'Subpixel (grid_sample)' if smooth_subpixel else 'Integer crop/resize'}")
        if str(amount_type).strip().lower() == "target percentage":
            info_lines.append(f"Target zoom percentage:         {int(zoom_percentage)}%")
        if smooth_subpixel:
            info_lines.append("Note: margins are converted to continuous zoom factors and applied with subpixel sampling.")
        else:
            info_lines.append("Note: margins are proportional per axis to preserve aspect; integer crop indices are used.")
        if clamped:
            info_lines.append("Warning: requested margins exceeded safe bounds and were clamped.")
        info = "\n".join(info_lines)

        return (out, info)
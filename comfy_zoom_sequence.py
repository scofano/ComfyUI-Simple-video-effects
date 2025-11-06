# comfy_zoom_sequence.py
import math
import torch
import torch.nn.functional as F

# ---- EASING HELPERS ---------------------------------------------------------
def ease_value(t: float, mode: str) -> float:
    # t in [0,1]
    if mode == "LINEAR":
        return t
    elif mode == "EASE_IN":
        # cubic in
        return t * t * t
    elif mode == "EASE_OUT":
        # cubic out
        u = 1.0 - t
        return 1.0 - (u * u * u)
    elif mode == "EASE_IN_OUT":
        # smoothstep (cubic Hermite)
        return t * t * (3 - 2 * t)
    else:
        return t

# ---- NODE -------------------------------------------------------------------
class ZoomSequenceNode:
    """
    Zooms IN or OUT across a batched IMAGE (video/sequence). The canvas size stays fixed.

    Parameters
    ----------
    mode:
        IN  -> progressively crop inward, then resize back to canvas (zooming in)
        OUT -> start cropped-in, progressively reduce crop so last frame is the original size
    pixels_per_frame:
        Margin growth (per side) in pixels/frame when fully linear. With easing, this is treated
        as the *maximum linear ramp* used to derive a total max margin.
        For IN: max_margin = clamp(pixels_per_frame * (N-1)).
        For OUT: initial_margin = same max_margin, then eased back to 0 by the final frame.
    ease:
        Easing curve applied to the progress from first to last frame.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "mode": (["IN", "OUT"], {"default": "IN"}),
                "pixels_per_frame": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.1}),
                "ease": (["LINEAR", "EASE_IN", "EASE_OUT", "EASE_IN_OUT"], {"default": "LINEAR"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("images", "info",)
    FUNCTION = "run"
    CATEGORY = "Utilities/Transforms"

    def _resize_to(self, frame_hwc: torch.Tensor, size_hw):
        """Resize HWC float[0..1] -> HWC using bilinear."""
        H, W = size_hw
        c = frame_hwc.shape[2]
        x = frame_hwc.permute(2, 0, 1).unsqueeze(0)  # 1,C,H,W
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        return x.squeeze(0).permute(1, 2, 0)  # H,W,C

    def run(self, images: torch.Tensor, mode: str, pixels_per_frame: float, ease: str):
        # Expect images shape: [B, H, W, C], values in 0..1
        if images.ndim != 4:
            return (images, "Input is not a batched IMAGE (B,H,W,C).")
        B, H, W, C = images.shape
        if B <= 0:
            return (images, "Empty batch; nothing to do.")

        min_dim = min(H, W)
        # Maximum safe per-side margin (ensure at least 1 pixel remains after crop)
        max_safe_margin = max(0, (min_dim // 2) - 1)

        # Max margin derived from pixels_per_frame and total frames
        requested_max_margin = pixels_per_frame * max(0, B - 1)
        max_margin = min(int(round(requested_max_margin)), max_safe_margin)

        # Build eased progress per frame t in [0,1]
        if B == 1:
            ts = [0.0]
        else:
            ts = [i / (B - 1) for i in range(B)]
        es = [ease_value(t, ease) for t in ts]

        # Compute margins per frame (per-side) in pixels
        margins = []
        if mode == "IN":
            # Start at 0, end at max_margin
            for e in es:
                m = int(round(max_margin * e))
                margins.append(m)
        else:  # "OUT"
            # Start at max_margin, end at 0
            for e in es:
                m = int(round(max_margin * (1.0 - e)))
                margins.append(m)

        # Apply per-frame crop + resize back to (H, W)
        out_frames = []
        clamped = False
        for i in range(B):
            m = margins[i]
            # Ensure bounds
            m = max(0, min(m, max_safe_margin))
            if m != margins[i]:
                clamped = True

            x0, y0 = m, m
            x1, y1 = W - m, H - m

            if x1 <= x0 or y1 <= y0:
                # Degenerate crop; just duplicate the frame
                out_frames.append(images[i])
                continue

            cropped = images[i, y0:y1, x0:x1, :]
            resized = self._resize_to(cropped, (H, W))
            out_frames.append(resized)

        out = torch.stack(out_frames, dim=0)

        info_lines = []
        info_lines.append(f"Frames: {B}, Canvas: {W}x{H}, Mode: {mode}, Ease: {ease}")
        info_lines.append(f"Requested max margin: {int(round(requested_max_margin))} px")
        info_lines.append(f"Applied max margin:   {max_margin} px (safe limit: {max_safe_margin} px)")
        if clamped:
            info_lines.append("Note: margin was clamped to fit the canvas.")
        info = "\n".join(info_lines)

        return (out, info)


# ---- NODE REGISTRATION -------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "ZoomSequenceNode": ZoomSequenceNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ZoomSequenceNode": "Zoom Sequence (In/Out, Easing)",
}

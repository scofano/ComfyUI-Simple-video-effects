# comfy_zoom_sequence.py
import torch
import torch.nn.functional as F

# ---------- EASING ------------------------------------------------------------
def ease_value(t: float, mode: str) -> float:
    key = mode.strip().upper().replace("-", "_")
    if key == "LINEAR":
        return t
    elif key == "EASE_IN":
        return t * t * t
    elif key == "EASE_OUT":
        u = 1.0 - t
        return 1.0 - u * u * u
    elif key == "EASE_IN_OUT":
        return t * t * (3 - 2 * t)
    return t  # fallback

def normalize_mode(mode: str) -> str:
    m = mode.strip().lower()
    if m in ("zoom in", "in"):
        return "IN"
    if m in ("zoom out", "out"):
        return "OUT"
    return "IN"

# ---------- NODE --------------------------------------------------------------
class ZoomSequenceNode:
    """
    Zooms IN or OUT across a batched IMAGE (video/sequence). The canvas size stays fixed and
    the original aspect ratio is preserved exactly for every frame.

    Parameters
    ----------
    mode: "Zoom In" or "Zoom Out"
    pixels_per_frame (INT): speed, defined on the SMALLER canvas dimension (per-side).
        We compute a scalar inset on the smaller dimension and convert it proportionally
        to per-axis margins so aspect stays constant.
    ease: "Linear", "Ease_In", "Ease_Out", "Ease_In_Out"
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "mode": (["Zoom In", "Zoom Out"], {"default": "Zoom In"}),
                "pixels_per_frame": ("INT", {"default": 1, "min": 0, "step": 1}),
                "ease": (["Linear", "Ease_In", "Ease_Out", "Ease_In_Out"], {"default": "Linear"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("images", "info",)
    FUNCTION = "run"
    CATEGORY = "Utilities/Transforms"

    # ---- helpers -------------------------------------------------------------
    def _resize_to(self, frame_hwc: torch.Tensor, size_hw):
        H, W = size_hw
        x = frame_hwc.permute(2, 0, 1).unsqueeze(0)  # 1,C,H,W
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        return x.squeeze(0).permute(1, 2, 0)

    def _aspect_corrected_crop_box(self, W, H, m_small):
        """
        Given a scalar margin on the smaller dimension (per-side),
        compute integer crop box (x0,y0,x1,y1) that:
          - preserves aspect exactly (within ±1 px due to rounding),
          - never exceeds bounds.
        """
        # Original aspect (float)
        aspect = W / H

        # Which side is smaller?
        small = min(W, H)
        big   = max(W, H)

        # Safe guard: ensure we can crop at least 1 px interior
        max_small_margin = max(0, (small // 2) - 1)
        m_small = max(0, min(m_small, max_small_margin))

        # Fractional zoom level based on smaller half-size
        half_small = small / 2.0
        frac = 0.0 if half_small <= 0 else (m_small / half_small)  # 0..~1

        # Proportional margins per axis to keep aspect (before integer correction)
        m_x_f = frac * (W / 2.0)
        m_y_f = frac * (H / 2.0)
        m_x = int(round(m_x_f))
        m_y = int(round(m_y_f))

        # Compute crop size
        cw = W - 2 * m_x
        ch = H - 2 * m_y

        # Enforce minimum interior
        cw = max(1, cw)
        ch = max(1, ch)

        # ---- Aspect correction (integer) ------------------------------------
        # target: cw / ch == W / H  =>  cw == round(ch * aspect)
        ideal_cw = int(round(ch * aspect))

        # Clamp ideal_cw to bounds [1, W - 2]
        ideal_cw = max(1, min(ideal_cw, W - 2))

        # Compare and adjust the dimension that changed least
        if ideal_cw != cw:
            # Recompute m_x from ideal_cw, keep m_y
            cw = ideal_cw
            m_x = (W - cw) // 2
            # keep symmetric crop; if parity mismatch, nudge by 1 px
            if W - 2 * m_x != cw:
                # try nudging m_x by +1
                m_x2 = max(0, min(m_x + 1, (W - 1) // 2))
                if W - 2 * m_x2 == cw:
                    m_x = m_x2
                else:
                    # Or recompute ch from cw if still off
                    ch = max(1, int(round(cw / aspect)))
                    m_y = (H - ch) // 2

        # Final clamp to safe range
        m_x = max(0, min(m_x, (W - 1) // 2))
        m_y = max(0, min(m_y, (H - 1) // 2))

        x0, y0 = m_x, m_y
        x1, y1 = W - m_x, H - m_y
        return x0, y0, x1, y1, m_x, m_y

    # ---- main ---------------------------------------------------------------
    def run(self, images: torch.Tensor, mode: str, pixels_per_frame: int, ease: str):
        if images.ndim != 4:
            return (images, "Input is not a batched IMAGE (B,H,W,C).")
        B, H, W, C = images.shape
        if B <= 0:
            return (images, "Empty batch; nothing to do.")

        # Normalize
        i_mode = normalize_mode(mode)
        i_ease = ease

        # Compute max scalar margin ON THE SMALLER DIMENSION (per-side)
        small = min(W, H)
        max_safe_small_margin = max(0, (small // 2) - 1)
        requested_small_margin_max = int(pixels_per_frame) * max(0, B - 1)
        small_margin_max = min(requested_small_margin_max, max_safe_small_margin)

        # Eased progress 0..1 across frames
        ts = [0.0] if B == 1 else [i / (B - 1) for i in range(B)]
        es = [ease_value(t, i_ease) for t in ts]

        # Per-frame scalar margin on the smaller side
        if i_mode == "IN":
            m_smalls = [int(round(small_margin_max * e)) for e in es]
        else:
            m_smalls = [int(round(small_margin_max * (1.0 - e))) for e in es]

        # Apply proportional, aspect-corrected crops
        out_frames = []
        clamped = False
        used_mx = []
        used_my = []
        for i in range(B):
            x0, y0, x1, y1, m_x, m_y = self._aspect_corrected_crop_box(W, H, m_smalls[i])

            # Detect clamping
            if m_smalls[i] != min(m_smalls[i], max(0, (min(W, H) // 2) - 1)):
                clamped = True

            # Guard degenerate
            if x1 <= x0 or y1 <= y0:
                out_frames.append(images[i])
                continue

            cropped = images[i, y0:y1, x0:x1, :]
            resized = self._resize_to(cropped, (H, W))
            out_frames.append(resized)
            used_mx.append(m_x)
            used_my.append(m_y)

        out = torch.stack(out_frames, dim=0)

        info = (
            f"Frames: {B}, Canvas: {W}x{H}, Mode: {mode}, Ease: {ease}\n"
            f"Requested small-dim max margin: {requested_small_margin_max} px\n"
            f"Applied small-dim max margin:   {small_margin_max} px (safe limit: {max_safe_small_margin} px)\n"
            f"Note: margins are proportional per axis to preserve aspect; minor ±1px integer adjustments may apply."
        )
        if clamped:
            info += "\nWarning: requested margins exceeded safe bounds and were clamped."

        return (out, info)

# ---------- REGISTRATION ------------------------------------------------------
NODE_CLASS_MAPPINGS = {"ZoomSequenceNode": ZoomSequenceNode}
NODE_DISPLAY_NAME_MAPPINGS = {"ZoomSequenceNode": "Zoom Sequence (In/Out, Easing)"}

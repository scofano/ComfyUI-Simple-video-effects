# comfy_zoom_sequence_batched.py
import os
import json
import tempfile

import torch
import torch.nn.functional as F

STATE_FILE = os.path.join(tempfile.gettempdir(), "zoom_sequence_state.json")

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
        return t * t * (3 - 2 * t)  # smoothstep
    return t

def normalize_mode(mode: str) -> str:
    m = mode.strip().lower()
    if m in ("zoom in", "in"):
        return "IN"
    if m in ("zoom out", "out"):
        return "OUT"
    return "IN"

# ---------- STATE HELPERS -----------------------------------------------------
def _load_state():
    if not os.path.exists(STATE_FILE):
        return None
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _save_state(state: dict):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f)
    except Exception:
        pass

def _clear_state_file():
    if os.path.exists(STATE_FILE):
        try:
            os.remove(STATE_FILE)
        except Exception:
            pass

# ---------- NODE --------------------------------------------------------------
class ZoomSequenceNode:
    """
    Batched zoom with persistent state across batches.

    New inputs:
      - source_frame_count: total frames in full video

    The temp state is automatically cleared when the processed frames
    reach the end (last processed frame index == source_frame_count - 1).
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "source_frame_count": ("INT", {"default": 1, "min": 1, "max": 100000}),
                "mode": (["Zoom In", "Zoom Out"], {"default": "Zoom In"}),
                "pixels_per_frame": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.05}),
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

    def _aspect_corrected_crop_box(self, W: int, H: int, m_small_int: int):
        aspect = W / H if H > 0 else 1.0

        small = min(W, H)
        if small <= 2:
            return 0, 0, W, H, 0, 0

        max_small_margin = max(0, (small // 2) - 1)
        m_small = max(0, min(int(m_small_int), max_small_margin))

        half_small = small / 2.0
        frac = 0.0 if half_small <= 0 else (m_small / half_small)

        m_x = int(round(frac * (W / 2.0)))
        m_y = int(round(frac * (H / 2.0)))

        cw = W - 2 * m_x
        ch = H - 2 * m_y

        cw = max(1, cw)
        ch = max(1, ch)

        ideal_cw = int(round(ch * aspect))
        ideal_cw = max(1, min(ideal_cw, W - 2))

        if ideal_cw != cw:
            cw = ideal_cw
            m_x = (W - cw) // 2
            if W - 2 * m_x != cw:
                m_x2 = max(0, min(m_x + 1, (W - 1) // 2))
                if W - 2 * m_x2 == cw:
                    m_x = m_x2
                else:
                    ch = max(1, int(round(cw / aspect)))
                    m_y = (H - ch) // 2

        m_x = max(0, min(m_x, (W - 1) // 2))
        m_y = max(0, min(m_y, (H - 1) // 2))

        x0, y0 = m_x, m_y
        x1, y1 = W - m_x, H - m_y
        return x0, y0, x1, y1, m_x, m_y

    # ---- main ---------------------------------------------------------------
    def run(
        self,
        images: torch.Tensor,
        source_frame_count: int,
        mode: str,
        pixels_per_frame: float,
        ease: str,
    ):
        if images.ndim != 4:
            return (images, "Input is not a batched IMAGE (B,H,W,C).")
        B, H, W, C = images.shape
        if B <= 0:
            return (images, "Empty batch; nothing to do.")

        i_mode = normalize_mode(mode)
        i_ease = ease

        small = min(W, H)
        max_safe_small_margin = max(0, (small // 2) - 1)

        prev_state = _load_state()
        state_valid = False
        if prev_state is not None:
            try:
                state_valid = (
                    prev_state.get("W") == W
                    and prev_state.get("H") == H
                    and prev_state.get("mode") == i_mode
                    and prev_state.get("ease") == i_ease
                    and prev_state.get("source_frame_count") == int(source_frame_count)
                    and float(prev_state.get("pixels_per_frame", pixels_per_frame)) == float(pixels_per_frame)
                )
            except Exception:
                state_valid = False

        if not state_valid:
            prev_state = {
                "last_frame": -1,
                "max_margin": None,
                "W": W,
                "H": H,
                "mode": i_mode,
                "ease": i_ease,
                "source_frame_count": int(source_frame_count),
                "pixels_per_frame": float(pixels_per_frame),
            }

        last_frame = int(prev_state.get("last_frame", -1))

        if prev_state.get("max_margin") is not None:
            small_margin_max_f = float(prev_state["max_margin"])
        else:
            requested_small_margin_max_f = pixels_per_frame * max(0, source_frame_count - 1)
            small_margin_max_f = requested_small_margin_max_f

        small_margin_max_f = min(max(0.0, small_margin_max_f), float(max_safe_small_margin))
        prev_state["max_margin"] = small_margin_max_f

        start_global = max(0, last_frame + 1)
        denom = max(1, source_frame_count - 1)

        out_frames = []
        clamped = False

        for i in range(B):
            g = start_global + i
            if g >= source_frame_count:
                g = source_frame_count - 1

            t = g / denom
            e = ease_value(t, i_ease)

            if i_mode == "IN":
                m_small_f = small_margin_max_f * e
            else:
                m_small_f = small_margin_max_f * (1.0 - e)

            m_small_int = int(round(m_small_f))
            if m_small_int != max(0, min(m_small_int, (small // 2) - 1)):
                clamped = True

            x0, y0, x1, y1, m_x, m_y = self._aspect_corrected_crop_box(W, H, m_small_int)

            if x1 <= x0 or y1 <= y0:
                out_frames.append(images[i])
                continue

            cropped = images[i, y0:y1, x0:x1, :]
            resized = self._resize_to(cropped, (H, W))
            out_frames.append(resized)

        out = torch.stack(out_frames, dim=0)

        new_last_frame = min(start_global + B - 1, source_frame_count - 1)

        # If we've reached the end of the sequence, clear the temp state.
        if new_last_frame >= source_frame_count - 1:
            _clear_state_file()
        else:
            prev_state["last_frame"] = new_last_frame
            _save_state(prev_state)

        info_lines = [
            f"Batch frames: {B}, Canvas: {W}x{H}, Mode: {mode}, Ease: {ease}",
            f"Source frame count: {source_frame_count}",
            f"Global frames processed in this call: {start_global}..{new_last_frame}",
            f"Requested conceptual max margin: ~{pixels_per_frame * max(0, source_frame_count - 1):.2f} px",
            f"Applied small-dim max margin:     {small_margin_max_f:.2f} px (safe limit: {max_safe_small_margin} px)",
            "Note: zoom continuity is preserved across batches via a temp state file.",
            f"State file: {STATE_FILE}",
        ]
        if new_last_frame >= source_frame_count - 1:
            info_lines.append("Info: reached final frame; zoom state has been reset.")
        if clamped:
            info_lines.append("Warning: requested margins exceeded safe bounds and were clamped.")
        info = "\n".join(info_lines)

        return (out, info)
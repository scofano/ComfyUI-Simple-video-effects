# comfy_zoom_sequence_batched.py
import os
import json
import tempfile
import time

import torch
import torch.nn.functional as F

from .zoom_core import (
    apply_center_zoom_subpixel,
    aspect_corrected_crop_box,
    compute_zoom_margins,
    margin_to_zoom_factor,
    normalize_amount_type,
    normalize_direction,
    resolve_direction_mode,
    resolve_ease_mode,
    resolve_seed_value,
)

STATE_FILE = os.path.join(tempfile.gettempdir(), "zoom_sequence_state.json")


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

    source_frame_count controls the global timeline (full sequence length).
    State is automatically cleared when reaching source_frame_count - 1.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "source_frame_count": ("INT", {"default": 1, "min": 1, "max": 100000}),
                "direction": (["Zoom In", "Zoom Out", "Random"], {"default": "Zoom In"}),
                "amount_type": (["Pixels per Frame", "Target Percentage"], {"default": "Pixels per Frame"}),
                "pixels_per_frame": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.05}),
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
        source_frame_count,
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
            int(source_frame_count),
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
        x = frame_hwc.permute(2, 0, 1).unsqueeze(0)  # 1,C,H,W
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        return x.squeeze(0).permute(1, 2, 0)

    # ---- main ----------------------------------------------------------------
    def run(
        self,
        images: torch.Tensor,
        source_frame_count: int,
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

        requested_direction = (direction or "Zoom In").strip()
        i_amount_type = normalize_amount_type(amount_type)
        requested_ease = (ease or "Linear").strip()
        input_seed = int(random_seed)
        random_mode = (
            requested_direction.strip().lower() == "random"
            or requested_ease.strip().upper().replace("-", "_") == "RANDOM"
        )
        use_auto_seed = random_mode and input_seed == 0

        prev_state = _load_state()
        state_valid = False
        if prev_state is not None:
            try:
                state_valid = (
                    prev_state.get("W") == W
                    and prev_state.get("H") == H
                    and prev_state.get("requested_direction") == requested_direction
                    and prev_state.get("amount_type") == i_amount_type
                    and prev_state.get("requested_ease") == requested_ease
                    and int(prev_state.get("random_seed_input", random_seed)) == int(input_seed)
                    and prev_state.get("source_frame_count") == int(source_frame_count)
                    and float(prev_state.get("pixels_per_frame", pixels_per_frame)) == float(pixels_per_frame)
                    and int(prev_state.get("zoom_percentage", zoom_percentage)) == int(zoom_percentage)
                    and bool(prev_state.get("smooth_subpixel", smooth_subpixel)) == bool(smooth_subpixel)
                )
            except Exception:
                state_valid = False

        if not state_valid:
            seed_i, auto_seed = resolve_seed_value(input_seed)
            effective_direction, rolled_direction = resolve_direction_mode(requested_direction, seed_i)
            rolled_ease, rolled_random = resolve_ease_mode(requested_ease, seed_i + 1)
            i_direction = normalize_direction(effective_direction)
            prev_state = {
                "last_frame": -1,
                "W": W,
                "H": H,
                "requested_direction": requested_direction,
                "effective_direction": effective_direction,
                "amount_type": i_amount_type,
                "requested_ease": requested_ease,
                "effective_ease": rolled_ease,
                "random_seed_input": int(input_seed),
                "effective_random_seed": int(seed_i),
                "auto_seed": bool(auto_seed),
                "source_frame_count": int(source_frame_count),
                "pixels_per_frame": float(pixels_per_frame),
                "zoom_percentage": int(zoom_percentage),
                "smooth_subpixel": bool(smooth_subpixel),
            }
        else:
            effective_direction = str(prev_state.get("effective_direction", "Zoom In"))
            rolled_ease = str(prev_state.get("effective_ease", "Linear"))
            seed_i = int(prev_state.get("effective_random_seed", input_seed))
            auto_seed = bool(prev_state.get("auto_seed", False))
            i_direction = normalize_direction(effective_direction)
            rolled_direction = requested_direction.strip().lower() == "random"
            rolled_random = requested_ease.strip().upper().replace("-", "_") == "RANDOM"

        last_frame = int(prev_state.get("last_frame", -1))
        start_global = max(0, last_frame + 1)

        small = min(W, H)
        m_smalls_f, meta = compute_zoom_margins(
            frame_count=B,
            small_dim=small,
            direction=i_direction,
            amount_type=i_amount_type,
            pixels_per_frame=pixels_per_frame,
            zoom_percentage=zoom_percentage,
            ease=rolled_ease,
            timeline_start=start_global,
            timeline_total=source_frame_count,
        )

        out_frames = []
        for i in range(B):
            if smooth_subpixel:
                zf = margin_to_zoom_factor(m_smalls_f[i], small)
                out_frames.append(apply_center_zoom_subpixel(images[i], zf, mode="bilinear"))
            else:
                m_small_int = int(round(m_smalls_f[i]))
                x0, y0, x1, y1, _, _ = aspect_corrected_crop_box(W, H, m_small_int)

                if x1 <= x0 or y1 <= y0:
                    out_frames.append(images[i])
                    continue

                cropped = images[i, y0:y1, x0:x1, :]
                resized = self._resize_to(cropped, (H, W))
                out_frames.append(resized)

        out = torch.stack(out_frames, dim=0)

        new_last_frame = min(start_global + B - 1, source_frame_count - 1)
        if new_last_frame >= source_frame_count - 1:
            _clear_state_file()
        else:
            prev_state["last_frame"] = new_last_frame
            _save_state(prev_state)

        info_lines = [
            f"Batch frames: {B}, Canvas: {W}x{H}, Direction: {effective_direction}, Amount Type: {amount_type}, Ease: {rolled_ease}",
            f"Source frame count: {source_frame_count}",
            f"Global frames processed in this call: {start_global}..{new_last_frame}",
            f"Requested conceptual max margin: {meta['requested_small_margin_max_f']:.2f} px",
            f"Applied small-dim max margin:   {meta['applied_small_margin_max_f']:.2f} px (safe limit: {meta['max_safe_small_margin']} px)",
            f"Effective max zoom reached:     {meta['effective_max_zoom']:.4f}x",
            f"Transform mode:                 {'Subpixel (grid_sample)' if smooth_subpixel else 'Integer crop/resize'}",
            (
                "Note: continuous zoom factors are applied with subpixel sampling; continuity is preserved across batches via a temp state file."
                if smooth_subpixel
                else "Note: integer crop indices are used; continuity is preserved across batches via a temp state file."
            ),
            f"State file: {STATE_FILE}",
        ]
        if rolled_direction:
            info_lines.append(f"Direction Random roll selected: {effective_direction}")
        if rolled_random:
            info_lines.append(f"Ease Random roll selected:      {rolled_ease}")
        if rolled_direction or rolled_random:
            info_lines.append(
                f"Random seed:                   {seed_i}{' (auto from 0)' if auto_seed else ''}"
            )
        if i_amount_type == "PERCENTAGE":
            info_lines.append(f"Target zoom percentage:         {int(zoom_percentage)}%")
        if new_last_frame >= source_frame_count - 1:
            info_lines.append("Info: reached final frame; zoom state has been reset.")
        if bool(meta.get("clamped", False)):
            info_lines.append("Warning: requested margins exceeded safe bounds and were clamped.")

        return (out, "\n".join(info_lines))

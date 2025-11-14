import torch
import torch.nn.functional as F


# ---------- EASING ------------------------------------------------------------
def ease_value(t: float, mode: str) -> float:
    """
    Cubic-based ease modes:
      - Linear
      - Ease_In
      - Ease_Out
      - Ease_In_Out
    """
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
    return t  # fallback


# ---------- NODE --------------------------------------------------------------
class CameraMoveNode:
    """
    Camera move / directional pan for a batched IMAGE (video/sequence).

    Canvas size stays fixed. To avoid black borders, the input is uniformly
    scaled up (keeping aspect ratio) so that along the movement axis there is
    at least `distance_px` extra pixels, and each frame is a crop that slides
    across this larger image.

    Directions
    ----------
    Top:
      - Scale image so height = H + distance_px (width scaled proportionally).
      - Crop at top at t=0, move the crop downward over time.
    Bottom:
      - Same scaling, but crop moves upward over time.
    Left / Right:
      - Similar, but scale based on width and move horizontally.
    Random:
      - On each run, randomly picks one of Top/Bottom/Left/Right and uses it
        for the whole sequence.

    Parameters
    ----------
    distance_px (FLOAT):
        Total camera travel in pixels over the whole sequence.
    duration_s (FLOAT):
        Duration in seconds that this camera move represents (for info only).
    ease: "Linear", "Ease_In", "Ease_Out", "Ease_In_Out"
        Easing of the movement over the frames.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "direction": (
                    ["Left", "Right", "Top", "Bottom", "Random"],
                    {"default": "Top"},
                ),
                "distance_px": ("FLOAT", {"default": 100.0, "min": 0.0, "step": 1.0}),
                "duration_s": ("FLOAT", {"default": 5.0, "min": 0.0, "step": 0.1}),
                "ease": (
                    ["Linear", "Ease_In", "Ease_Out", "Ease_In_Out"],
                    {"default": "Ease_Out"},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("images", "info",)
    FUNCTION = "run"
    CATEGORY = "Utilities/Transforms"

    # ---- main ---------------------------------------------------------------
    def run(self, images: torch.Tensor, direction: str, distance_px: float,
            duration_s: float, ease: str):

        if images.ndim != 4:
            return (images, "Input is not a batched IMAGE (B,H,W,C).")

        B, H, W, C = images.shape
        if B <= 0:
            return (images, "Empty batch; nothing to do.")

        # Normalize parameters
        dir_key = direction.strip().lower()
        dist = max(0.0, float(distance_px))
        dist_int = int(round(dist))

        # Resolve Random direction once per run
        effective_dir = dir_key
        if dir_key == "random":
            # 0: left, 1: right, 2: top, 3: bottom
            idx = int(torch.randint(0, 4, (1,)).item())
            effective_dir = ["left", "right", "top", "bottom"][idx]

        if dist_int == 0:
            # No movement requested, just return original
            info = (
                f"Frames: {B}, Canvas: {W}x{H}\n"
                f"Direction: {direction} (resolved: {effective_dir.title()}), "
                f"Distance: 0px, Ease: {ease}\n"
                f"Duration: {duration_s:.2f}s\n"
                "No movement applied (distance_px = 0)."
            )
            return (images, info)

        # Decide which axis we move along and compute target size
        move_axis = None  # "x" or "y"
        if effective_dir in ("left", "right"):
            move_axis = "x"
            target_w = W + dist_int
            scale = target_w / float(W)
            target_h = int(round(H * scale))
        elif effective_dir in ("top", "bottom"):
            move_axis = "y"
            target_h = H + dist_int
            scale = target_h / float(H)
            target_w = int(round(W * scale))
        else:
            # Fallback: no movement axis, center crop only
            move_axis = None
            target_h, target_w = H, W

        newH, newW = target_h, target_w

        # Convert to BCHW for interpolation
        imgs_nchw = images.permute(0, 3, 1, 2)

        # Uniform scale up (keeps proportions)
        imgs_big = torch.nn.functional.interpolate(
            imgs_nchw,
            size=(newH, newW),
            mode="bicubic",
            align_corners=False,
        )

        # Eased progress 0..1 across frames
        ts = [0.0] if B == 1 else [i / (B - 1) for i in range(B)]
        es = [ease_value(t, ease) for t in ts]

        crops = []

        for i in range(B):
            e = es[i]
            frame = imgs_big[i]  # (C, newH, newW)

            if move_axis == "y":
                # vertical move: crop slides along height
                max_off_y = newH - H  # should be == dist_int
                if effective_dir == "top":
                    start_off = 0
                    end_off = max_off_y
                else:  # "bottom"
                    start_off = max_off_y
                    end_off = 0

                offset = start_off + (end_off - start_off) * e
                y0 = int(round(offset))
                y0 = max(0, min(max_off_y, y0))
                y1 = y0 + H

                # horizontally center the crop
                max_off_x = newW - W
                x0 = max(0, max_off_x // 2)
                x1 = x0 + W

                crop = frame[:, y0:y1, x0:x1]

            elif move_axis == "x":
                # horizontal move: crop slides along width
                max_off_x = newW - W  # should be == dist_int
                if effective_dir == "left":
                    start_off = 0
                    end_off = max_off_x
                else:  # "right"
                    start_off = max_off_x
                    end_off = 0

                offset = start_off + (end_off - start_off) * e
                x0 = int(round(offset))
                x0 = max(0, min(max_off_x, x0))
                x1 = x0 + W

                # vertically center the crop
                max_off_y = newH - H
                y0 = max(0, max_off_y // 2)
                y1 = y0 + H

                crop = frame[:, y0:y1, x0:x1]

            else:
                # fallback: center crop (no move)
                y0 = max(0, (newH - H) // 2)
                x0 = max(0, (newW - W) // 2)
                crop = frame[:, y0:y0+H, x0:x0+W]

            # Safety: ensure crop size matches original canvas
            if crop.shape[1] != H or crop.shape[2] != W:
                crop = torch.nn.functional.interpolate(
                    crop.unsqueeze(0),
                    size=(H, W),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze(0)

            crops.append(crop)

        out_nchw = torch.stack(crops, dim=0)  # (B, C, H, W)
        out_bhwc = out_nchw.permute(0, 2, 3, 1)  # back to (B, H, W, C)

        # Info / diagnostics
        fps = None
        if duration_s > 0.0 and B > 1:
            fps = (B - 1) / duration_s

        info_lines = []
        info_lines.append(f"Frames: {B}, Canvas: {W}x{H}")
        info_lines.append(
            f"Direction: {direction} (resolved: {effective_dir.title()}), "
            f"Distance: {dist_int}px, Ease: {ease}"
        )
        info_lines.append(f"Duration: {duration_s:.2f}s")
        if fps is not None:
            info_lines.append(f"Implied FPS (based on duration): {fps:.3f}")
        info_lines.append(
            f"Image uniformly scaled to {newW}x{newH} then animated via sliding crop "
            "(no black borders, aspect ratio preserved)."
        )
        info = "\n".join(info_lines)

        return (out_bhwc, info)
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
    scaled up (keeping aspect ratio) so that along each movement axis there is
    at least `distance_px` extra pixels. Each frame is then a crop that slides
    across this larger image.

    Directions
    ----------
    Horizontal:
      - None  : no horizontal move
      - Left  : crop moves from left to right
      - Right : crop moves from right to left
      - Random: randomly chooses Left or Right once per run

    Vertical:
      - None  : no vertical move
      - Top   : crop moves from top to bottom
      - Bottom: crop moves from bottom to top
      - Random: randomly chooses Top or Bottom once per run

    You can combine them (e.g. Left + Top) to get diagonal movement.

    Parameters
    ----------
    distance_px (FLOAT):
        Total camera travel in pixels over the whole sequence, per active axis.
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
                "horizontal_direction": (
                    ["None", "Left", "Right", "Random"],
                    {"default": "None"},
                ),
                "vertical_direction": (
                    ["None", "Top", "Bottom", "Random"],
                    {"default": "None"},
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
    CATEGORY = "Simple Video Effects"

    # Tell Comfy when to re-run this node instead of using cache
    @classmethod
    def IS_CHANGED(cls,
                   images,
                   horizontal_direction,
                   vertical_direction,
                   distance_px,
                   duration_s,
                   ease):
        # If any direction is Random, force a change every execution
        if horizontal_direction == "Random" or vertical_direction == "Random":
            # NaN is never equal to itself, so cache can't match -> always recompute
            return float("nan")
        # Normal caching behavior otherwise
        return None

    # ---- main ---------------------------------------------------------------
    def run(self,
            images: torch.Tensor,
            horizontal_direction: str,
            vertical_direction: str,
            distance_px: float,
            duration_s: float,
            ease: str):

        if images.ndim != 4:
            return (images, "Input is not a batched IMAGE (B,H,W,C).")

        B, H, W, C = images.shape
        if B <= 0:
            return (images, "Empty batch; nothing to do.")

        dist = max(0.0, float(distance_px))
        dist_int = int(round(dist))

        # Resolve directions / randoms
        hdir_key = horizontal_direction.strip().lower()
        vdir_key = vertical_direction.strip().lower()

        # Resolve random horizontally
        if hdir_key == "random":
            # choose left or right once per run
            idx = int(torch.randint(0, 2, (1,)).item())
            hdir_eff = ["left", "right"][idx]
        else:
            hdir_eff = hdir_key

        # Resolve random vertically
        if vdir_key == "random":
            idx = int(torch.randint(0, 2, (1,)).item())
            vdir_eff = ["top", "bottom"][idx]
        else:
            vdir_eff = vdir_key

        move_x = hdir_eff in ("left", "right")
        move_y = vdir_eff in ("top", "bottom")

        # If no movement or zero distance, just return original
        if dist_int == 0 or (not move_x and not move_y):
            info = (
                f"Frames: {B}, Canvas: {W}x{H}\n"
                f"Horizontal: {horizontal_direction} (resolved: {hdir_eff.title() if hdir_eff != 'none' else 'None'}), "
                f"Vertical: {vertical_direction} (resolved: {vdir_eff.title() if vdir_eff != 'none' else 'None'}), "
                f"Distance: {dist_int}px, Ease: {ease}\n"
                f"Duration: {duration_s:.2f}s\n"
                "No movement applied (either distance_px = 0 or both directions are None)."
            )
            return (images, info)

        # Desired overscan in each axis
        target_w = W + dist_int if move_x else W
        target_h = H + dist_int if move_y else H

        # Uniform scale to cover both overscans (keep aspect ratio)
        scale = max(target_w / float(W), target_h / float(H))
        newW = int(round(W * scale))
        newH = int(round(H * scale))

        # Convert to BCHW for interpolation
        imgs_nchw = images.permute(0, 3, 1, 2)

        # Uniform scale up
        imgs_big = F.interpolate(
            imgs_nchw,
            size=(newH, newW),
            mode="bicubic",
            align_corners=False,
        )

        # Eased progress 0..1 across frames
        ts = [0.0] if B == 1 else [i / (B - 1) for i in range(B)]
        es = [ease_value(t, ease) for t in ts]

        max_off_x = max(0, newW - W)
        max_off_y = max(0, newH - H)

        crops = []

        for i in range(B):
            e = es[i]
            frame = imgs_big[i]  # (C, newH, newW)

            # Horizontal offset
            if move_x and max_off_x > 0:
                if hdir_eff == "left":
                    start_x = 0
                    end_x = max_off_x
                else:  # "right"
                    start_x = max_off_x
                    end_x = 0
                offset_x = start_x + (end_x - start_x) * e
                x0 = int(round(offset_x))
                x0 = max(0, min(max_off_x, x0))
            else:
                # center if no movement on X
                x0 = max_off_x // 2

            x1 = x0 + W
            if x1 > newW:
                x1 = newW
                x0 = newW - W

            # Vertical offset
            if move_y and max_off_y > 0:
                if vdir_eff == "top":
                    start_y = 0
                    end_y = max_off_y
                else:  # "bottom"
                    start_y = max_off_y
                    end_y = 0
                offset_y = start_y + (end_y - start_y) * e
                y0 = int(round(offset_y))
                y0 = max(0, min(max_off_y, y0))
            else:
                # center if no movement on Y
                y0 = max_off_y // 2

            y1 = y0 + H
            if y1 > newH:
                y1 = newH
                y0 = newH - H

            crop = frame[:, y0:y1, x0:x1]

            # Safety: ensure crop size matches original canvas
            if crop.shape[1] != H or crop.shape[2] != W:
                crop = F.interpolate(
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
            "Horizontal: "
            f"{horizontal_direction} (resolved: {hdir_eff.title() if hdir_eff != 'none' else 'None'}), "
            "Vertical: "
            f"{vertical_direction} (resolved: {vdir_eff.title() if vdir_eff != 'none' else 'None'})"
        )
        info_lines.append(f"Distance per axis: {dist_int}px, Ease: {ease}")
        info_lines.append(f"Duration: {duration_s:.2f}s")
        if fps is not None:
            info_lines.append(f"Implied FPS (based on duration): {fps:.3f}")
        info_lines.append(
            f"Image uniformly scaled to {newW}x{newH} then animated via sliding crop "
            "(no black borders, aspect ratio preserved)."
        )
        info = "\n".join(info_lines)

        return (out_bhwc, info)
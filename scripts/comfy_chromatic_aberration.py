import torch

try:
    from comfy.utils import ProgressBar
except ImportError:
    class ProgressBar:
        def __init__(self, total): pass
        def update(self, v): pass


class ChromaticAberrationNode:
    """
    Shifts the R and B channels in opposite directions, leaving G centered.
    Works on a single image or a full batch (video sequence).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "shift": ("INT", {
                    "default": 8, "min": 0, "max": 300, "step": 1,
                    "tooltip": "Pixel distance each colour channel is offset from centre"
                }),
                "direction": (
                    ["Horizontal", "Vertical", "Diagonal"],
                    {"default": "Horizontal"}
                ),
                "red_leads": (
                    ["Red right / Blue left", "Red left / Blue right",
                     "Red down / Blue up", "Red up / Blue down"],
                    {
                        "default": "Red right / Blue left",
                        "tooltip": "Which side each channel shifts toward (Diagonal uses the first horizontal + vertical pair)"
                    }
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "apply"
    CATEGORY = "Simple Video Effects/Image Processing"

    # ------------------------------------------------------------------
    def _shift(self, ch: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
        """Shift a (B, H, W) channel tensor by (dx, dy) pixels, filling edges with 0."""
        B, H, W = ch.shape
        out = torch.zeros_like(ch)

        # horizontal
        if dx > 0 and dx < W:          # content moves right
            out[:, :, dx:] = ch[:, :, :W - dx]
        elif dx < 0 and -dx < W:       # content moves left
            out[:, :, :W + dx] = ch[:, :, -dx:]
        elif dx == 0:
            out = ch.clone()
        # if |dx| >= W, channel stays zero (full blackout — edge case)

        # vertical (operate on out so both axes stack correctly)
        if dy != 0:
            tmp = out.clone()
            out = torch.zeros_like(ch)
            if dy > 0 and dy < H:      # content moves down
                out[:, dy:, :] = tmp[:, :H - dy, :]
            elif dy < 0 and -dy < H:   # content moves up
                out[:, :H + dy, :] = tmp[:, -dy:, :]
            # if |dy| >= H, channel is zero

        return out

    def apply(self, images: torch.Tensor, shift: int, direction: str, red_leads: str):
        if images.ndim != 4:
            return (images,)
        if shift == 0:
            return (images,)

        B, H, W, C = images.shape

        # ---- resolve per-channel vectors from direction + red_leads ----
        if direction == "Horizontal":
            if red_leads == "Red right / Blue left":
                r_dx, r_dy, b_dx, b_dy = shift, 0, -shift, 0
            else:  # Red left / Blue right
                r_dx, r_dy, b_dx, b_dy = -shift, 0, shift, 0
        elif direction == "Vertical":
            if red_leads == "Red down / Blue up":
                r_dx, r_dy, b_dx, b_dy = 0, shift, 0, -shift
            else:  # Red up / Blue down
                r_dx, r_dy, b_dx, b_dy = 0, -shift, 0, shift
        else:  # Diagonal — combine horizontal + vertical choices
            if red_leads in ("Red right / Blue left", "Red left / Blue right"):
                h_r = shift if red_leads == "Red right / Blue left" else -shift
                r_dx, r_dy = h_r, -shift
                b_dx, b_dy = -h_r, shift
            else:
                v_r = shift if red_leads == "Red down / Blue up" else -shift
                r_dx, r_dy = shift, v_r
                b_dx, b_dy = -shift, -v_r

        pbar = ProgressBar(B)

        # channels: (B, H, W)
        r = images[:, :, :, 0]
        g = images[:, :, :, 1]
        b = images[:, :, :, 2]

        r_out = self._shift(r, r_dx, r_dy)
        b_out = self._shift(b, b_dx, b_dy)

        pbar.update(B)

        channels = [r_out, g, b_out]
        if C == 4:
            channels.append(images[:, :, :, 3])

        out = torch.stack(channels, dim=-1).clamp(0.0, 1.0)
        return (out,)

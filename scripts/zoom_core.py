import random

import torch
import torch.nn.functional as F


def ease_value(t: float, mode: str) -> float:
    key = mode.strip().upper().replace("-", "_")
    if key == "LINEAR":
        return t
    if key == "EASE_IN":
        return t * t * t
    if key == "EASE_OUT":
        u = 1.0 - t
        return 1.0 - u * u * u
    if key == "EASE_IN_OUT":
        return t * t * (3 - 2 * t)  # smoothstep
    return t  # fallback


def resolve_ease_mode(ease: str, seed: int = None):
    """
    Resolve ease mode, supporting 'Random'.
    Returns: (effective_ease: str, rolled_random: bool)
    """
    raw = (ease or "Linear").strip()
    key = raw.upper().replace("-", "_")
    if key == "RANDOM":
        choices = ["Linear", "Ease_In", "Ease_Out", "Ease_In_Out"]
        if seed is None:
            return random.choice(choices), True
        rng = random.Random(int(seed))
        return rng.choice(choices), True
    return raw, False


def resolve_direction_mode(direction: str, seed: int = None):
    """
    Resolve direction mode, supporting 'Random'.
    Returns: (effective_direction: str, rolled_random: bool)
    """
    raw = (direction or "Zoom In").strip()
    key = raw.lower()
    if key == "random":
        choices = ["Zoom In", "Zoom Out"]
        if seed is None:
            return random.choice(choices), True
        rng = random.Random(int(seed))
        return rng.choice(choices), True
    return raw, False


def resolve_seed_value(seed: int):
    """
    Resolve user-provided seed.
    - seed == 0: auto-random seed (new value per execution)
    - otherwise: fixed deterministic seed
    Returns: (effective_seed: int, auto_seed: bool)
    """
    s = int(seed)
    if s == 0:
        return random.SystemRandom().randint(1, 2147483647), True
    return s, False


def normalize_direction(direction: str) -> str:
    d = (direction or "").strip().lower()
    if d in ("zoom in", "in"):
        return "IN"
    if d in ("zoom out", "out"):
        return "OUT"
    return "IN"


def normalize_amount_type(amount_type: str) -> str:
    a = (amount_type or "").strip().lower()
    if a in ("pixels per frame", "pixels", "pixel", "px", "px/frame"):
        return "PIXELS"
    if a in ("target percentage", "percentage", "percent", "%"):
        return "PERCENTAGE"
    return "PIXELS"


def aspect_corrected_crop_box(W: int, H: int, m_small_int: int):
    """
    Given an integer scalar margin on the smaller dimension (per-side),
    compute integer crop box (x0,y0,x1,y1) that:
      - preserves aspect (within ±1 px due to rounding),
      - never exceeds bounds.
    """
    aspect = W / H if H > 0 else 1.0

    # Which side is smaller?
    small = min(W, H)
    if small <= 2:
        return 0, 0, W, H, 0, 0

    # Safe guard: ensure at least 1 px interior
    max_small_margin = max(0, (small // 2) - 1)
    m_small = max(0, min(int(m_small_int), max_small_margin))

    # Fractional zoom level based on smaller half-size
    half_small = small / 2.0
    frac = 0.0 if half_small <= 0 else (m_small / half_small)  # 0..~1

    # Proportional margins per axis to keep aspect (before integer correction)
    m_x = int(round(frac * (W / 2.0)))
    m_y = int(round(frac * (H / 2.0)))

    # Compute crop size
    cw = W - 2 * m_x
    ch = H - 2 * m_y

    # Enforce minimum interior
    cw = max(1, cw)
    ch = max(1, ch)

    # ---- Aspect correction (integer) ------------------------------------
    # target: cw / ch == W / H  =>  cw == round(ch * aspect)
    ideal_cw = int(round(ch * aspect))
    ideal_cw = max(1, min(ideal_cw, W - 2))  # keep some border to be safe in edge cases

    if ideal_cw != cw:
        # Recompute m_x from ideal_cw, keep m_y if possible
        cw = ideal_cw
        m_x = (W - cw) // 2
        # ensure symmetric crop; parity fix if needed
        if W - 2 * m_x != cw:
            m_x2 = max(0, min(m_x + 1, (W - 1) // 2))
            if W - 2 * m_x2 == cw:
                m_x = m_x2
            else:
                # else recompute ch from cw to restore symmetry
                ch = max(1, int(round(cw / aspect)))
                m_y = (H - ch) // 2

    # Final clamp to safe range
    m_x = max(0, min(m_x, (W - 1) // 2))
    m_y = max(0, min(m_y, (H - 1) // 2))

    x0, y0 = m_x, m_y
    x1, y1 = W - m_x, H - m_y
    return x0, y0, x1, y1, m_x, m_y


def _timeline_values(frame_count: int, timeline_start: int, timeline_total: int, ease: str):
    total = max(1, int(timeline_total))
    start = max(0, int(timeline_start))
    denom = max(1, total - 1)

    values = []
    for i in range(max(0, int(frame_count))):
        g = start + i
        if g >= total:
            g = total - 1
        t = g / denom
        values.append(ease_value(t, ease))
    return values


def compute_zoom_margins(
    frame_count: int,
    small_dim: int,
    direction: str,
    amount_type: str,
    pixels_per_frame: float,
    zoom_percentage: int,
    ease: str,
    timeline_start: int = 0,
    timeline_total: int = None,
):
    """
    Returns:
      margins_f: list[float] on smaller dimension (per-side)
      meta: dict with diagnostics values
    """
    if timeline_total is None:
        timeline_total = frame_count

    direction_key = normalize_direction(direction)
    amount_key = normalize_amount_type(amount_type)

    small = max(0, int(small_dim))
    half_small = small / 2.0 if small > 0 else 1.0
    max_safe_small_margin = max(0, (small // 2) - 1)

    es = _timeline_values(frame_count, timeline_start, timeline_total, ease)

    clamped = False
    requested_small_margin_max_f = 0.0

    if amount_key == "PERCENTAGE":
        zp = max(100, int(zoom_percentage))
        target_zoom = max(1.0, zp / 100.0)

        # Interpolate zoom factor over timeline.
        if direction_key == "IN":
            zoom_factors = [1.0 + (target_zoom - 1.0) * e for e in es]
        else:  # OUT
            zoom_factors = [1.0 + (target_zoom - 1.0) * (1.0 - e) for e in es]

        # For this crop model: zoom = 1 / (1 - frac), frac = margin / half_small.
        margins_raw = [half_small * (1.0 - (1.0 / max(1e-8, z))) for z in zoom_factors]
        requested_small_margin_max_f = max(margins_raw) if margins_raw else 0.0

        margins_f = [min(max(0.0, m), float(max_safe_small_margin)) for m in margins_raw]
        if requested_small_margin_max_f > float(max_safe_small_margin):
            clamped = True
    else:
        ppf = max(0.0, float(pixels_per_frame))
        requested_small_margin_max_f = ppf * max(0, int(timeline_total) - 1)
        small_margin_max_f = min(requested_small_margin_max_f, float(max_safe_small_margin))
        if requested_small_margin_max_f > float(max_safe_small_margin):
            clamped = True

        if direction_key == "IN":
            margins_f = [small_margin_max_f * e for e in es]
        else:  # OUT
            margins_f = [small_margin_max_f * (1.0 - e) for e in es]

    applied_small_margin_max_f = max(margins_f) if margins_f else 0.0

    # Effective maximum zoom (derived from applied max margin).
    frac_max = 0.0 if half_small <= 0 else (applied_small_margin_max_f / half_small)
    frac_max = min(max(0.0, frac_max), 0.999999)
    effective_max_zoom = 1.0 / max(1e-8, (1.0 - frac_max))

    meta = {
        "direction": direction_key,
        "amount_type": amount_key,
        "max_safe_small_margin": max_safe_small_margin,
        "requested_small_margin_max_f": requested_small_margin_max_f,
        "applied_small_margin_max_f": applied_small_margin_max_f,
        "effective_max_zoom": effective_max_zoom,
        "clamped": clamped,
    }
    return margins_f, meta


def margin_to_zoom_factor(m_small_f: float, small_dim: int) -> float:
    """Convert small-dimension per-side margin to an equivalent zoom factor."""
    small = max(1.0, float(small_dim))
    half_small = small / 2.0
    frac = max(0.0, min(float(m_small_f) / half_small, 0.999999))
    return 1.0 / max(1e-8, (1.0 - frac))


def apply_center_zoom_subpixel(
    frame_hwc: torch.Tensor,
    zoom_factor: float,
    mode: str = "bilinear",
) -> torch.Tensor:
    """
    Apply center-based subpixel zoom with grid_sample.
    Input/output shape: (H, W, C)
    """
    if frame_hwc.ndim != 3:
        return frame_hwc

    z = max(1.0, float(zoom_factor))
    if abs(z - 1.0) < 1e-8:
        return frame_hwc

    x = frame_hwc.permute(2, 0, 1).unsqueeze(0)  # 1,C,H,W
    dtype = x.dtype
    device = x.device

    s = 1.0 / z
    theta = torch.tensor(
        [[[s, 0.0, 0.0], [0.0, s, 0.0]]],
        dtype=dtype,
        device=device,
    )

    grid = F.affine_grid(theta, x.size(), align_corners=False)
    y = F.grid_sample(
        x,
        grid,
        mode=mode,
        padding_mode="border",
        align_corners=False,
    )

    return y.squeeze(0).permute(1, 2, 0)

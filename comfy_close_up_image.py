import torch
import torch.nn.functional as F
import random

def calculate_face_center(segs):
    """Calculate center point between eyes from SEGS data"""
    if not segs:
        raise ValueError("No segmentation data provided")

    # SEGS format: ((width, height), [SEG objects])
    if isinstance(segs, (list, tuple)) and len(segs) == 2:
        # Unpack the tuple
        dims, seg_list = segs
        segs = seg_list

    # Handle different SEGS formats
    valid_eyes = []

    for seg in segs:
        # Try different SEGS formats
        if hasattr(seg, 'confidence') and hasattr(seg, 'label') and hasattr(seg, 'bbox'):
            # Format: SEG objects with attributes
            confidence = seg.confidence[0] if hasattr(seg.confidence, '__len__') else seg.confidence
            if confidence > 0.4 and seg.label == 'eye':
                valid_eyes.append(seg.bbox)
        elif isinstance(seg, (list, tuple)) and len(seg) >= 3:
            # Format: tuples like (bbox, label, confidence)
            bbox, label, confidence = seg[0], seg[1], seg[2]
            confidence = confidence[0] if hasattr(confidence, '__len__') else confidence
            if confidence > 0.4 and label == 'eye':
                valid_eyes.append(bbox)
        else:
            # Unknown format, skip
            continue

    if len(valid_eyes) < 2:
        raise ValueError(f"Need at least 2 eyes with confidence > 0.4, found {len(valid_eyes)}")

    # Take first two eyes
    eye1 = valid_eyes[0]
    eye2 = valid_eyes[1]

    # Calculate eye centers
    x1 = (eye1[0] + eye1[2]) / 2
    y1 = (eye1[1] + eye1[3]) / 2

    x2 = (eye2[0] + eye2[2]) / 2
    y2 = (eye2[1] + eye2[3]) / 2

    # Face center is midpoint between eyes
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    return center_x, center_y


# ---------- NODE --------------------------------------------------------------
class CloseUpImageNode:
    """
    Close-up effect for images centered on face between eyes.

    Takes an image and SEGS data to detect eyes, calculates face center,
    and applies zoom centered on that point.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "segs": ("SEGS",),
                "zoom_factor": ("FLOAT", {"default": 1.5, "min": 1.0, "step": 0.1}),
                "random_zoom": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "zoom_factor_min": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.1}),
                "steps": ("FLOAT", {"default": 0.5, "min": 0.1, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("output_image",)
    FUNCTION = "apply_close_up_image"
    CATEGORY = "Simple Video Effects"

    # ---- main ---------------------------------------------------------------
    def apply_close_up_image(self, image, segs, zoom_factor: float, random_zoom: bool, seed: int, zoom_factor_min: float = None, steps: float = None):
        # Calculate face center from SEGS
        try:
            center_x, center_y = calculate_face_center(segs)
        except Exception as e:
            raise ValueError(f"Failed to calculate face center: {e}")

        # Get image dimensions
        B, H, W, C = image.shape
        width, height = W, H

        # Handle random zoom factor
        if random_zoom:
            random.seed(seed)
            if zoom_factor_min is None:
                zoom_factor_min = 1.0
            if steps is None:
                steps = 0.5
            # Generate possible zoom factors in steps
            possible_zooms = []
            current = zoom_factor_min
            while current <= zoom_factor:
                possible_zooms.append(current)
                current += steps
            zoom_factor = random.choice(possible_zooms)
            print(f"CloseUpImageNode: Using random zoom factor {zoom_factor}")
        else:
            print(f"CloseUpImageNode: Using zoom factor {zoom_factor}")

        # Apply close-up zoom
        zoomed_images = self._apply_zoom_to_center(image, zoom_factor, center_x, center_y, width, height)

        return (zoomed_images,)

    def _apply_zoom_to_center(self, images, zoom_factor, center_x, center_y, orig_width, orig_height):
        """Apply zoom centered on the specified point"""
        if images.ndim != 4:
            return images

        B, H, W, C = images.shape
        if B <= 0:
            return images

        # Calculate crop region for zoom
        # We want to zoom by zoom_factor, centered on (center_x, center_y)
        crop_width = int(round(W / zoom_factor))
        crop_height = int(round(H / zoom_factor))

        # Ensure crop dimensions don't exceed original
        crop_width = min(crop_width, W)
        crop_height = min(crop_height, H)

        # Calculate crop position centered on face center
        crop_x = int(round(center_x - crop_width / 2))
        crop_y = int(round(center_y - crop_height / 2))

        # Clamp to valid bounds
        crop_x = max(0, min(crop_x, W - crop_width))
        crop_y = max(0, min(crop_y, H - crop_height))

        # Apply crop and resize back to original dimensions
        out_frames = []
        for i in range(B):
            frame = images[i]  # (H, W, C)

            # Crop
            cropped = frame[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width, :]

            # Resize back to original size
            cropped_tensor = cropped.permute(2, 0, 1).unsqueeze(0)  # (1, C, crop_H, crop_W)
            resized = F.interpolate(cropped_tensor, size=(H, W), mode="bicubic", align_corners=False)
            resized_frame = resized.squeeze(0).permute(1, 2, 0)  # back to (H, W, C)

            out_frames.append(resized_frame)

        return torch.stack(out_frames, dim=0)

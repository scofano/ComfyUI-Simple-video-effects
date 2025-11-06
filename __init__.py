# __init__.py
from .comfy_zoom_sequence import ZoomSequenceNode

NODE_CLASS_MAPPINGS = {
    "ZoomSequenceNode": ZoomSequenceNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZoomSequenceNode": "Zoom Sequence (In/Out, Easing)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
from .comfy_zoom_sequence import ZoomSequenceNode as ZoomSequenceNodeSingle
from .batch_comfy_zoom_sequence import ZoomSequenceNode as ZoomSequenceNodeBatch

NODE_CLASS_MAPPINGS = {
    "ZoomSequenceNode": ZoomSequenceNodeSingle,
    "ZoomSequenceBatchNode": ZoomSequenceNodeBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZoomSequenceNode": "Zoom Sequence (In/Out, Easing)",
    "ZoomSequenceBatchNode": "Batch Zoom Sequence (In/Out, Easing)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
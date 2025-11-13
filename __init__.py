from .comfy_zoom_sequence import ZoomSequenceNode as ZoomSequenceNodeSingle
from .batch_comfy_zoom_sequence import ZoomSequenceNode as ZoomSequenceNodeBatch

NODE_CLASS_MAPPINGS = {
    "ZoomSequenceNode": ZoomSequenceNodeSingle,
    "ZoomSequenceBatchNode": ZoomSequenceNodeBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZoomSequenceNode": "Zoom Sequence (Single Batch)",
    "ZoomSequenceBatchNode": "Zoom Sequence (Batched)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
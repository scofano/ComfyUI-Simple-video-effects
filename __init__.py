from .comfy_zoom_sequence import ZoomSequenceNode as ZoomSequenceNodeSingle
from .batch_comfy_zoom_sequence import ZoomSequenceNode as ZoomSequenceNodeBatch
from .comfy_camera_shake import CameraShakeNode
from .comfy_video_overlay import VideoOverlay

NODE_CLASS_MAPPINGS = {
    "ZoomSequenceNode": ZoomSequenceNodeSingle,
    "ZoomSequenceBatchNode": ZoomSequenceNodeBatch,
    "CameraShakeNode": CameraShakeNode,
    "VideoOverlay": VideoOverlay,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZoomSequenceNode": "Zoom (In/Out, Easing)",
    "ZoomSequenceBatchNode": "Batch Zoom (In/Out, Easing)",
    "CameraShakeNode": "Camera Shake",
    "VideoOverlay": "Video Overlay",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

from .comfy_zoom_sequence import ZoomSequenceNode as ZoomSequenceNodeSingle
from .batch_comfy_zoom_sequence import ZoomSequenceNode as ZoomSequenceNodeBatch
from .comfy_camera_shake import CameraShakeNode
from .comfy_video_overlay import VideoOverlay
from .comfy_camera_move import CameraMoveNode
from .comfy_video_combiner import ComfyVideoCombiner
from .comfy_video_image_overlay import VideoImageOverlay


NODE_CLASS_MAPPINGS = {
    "ZoomSequenceNode": ZoomSequenceNodeSingle,
    "ZoomSequenceBatchNode": ZoomSequenceNodeBatch,
    "CameraShakeNode": CameraShakeNode,
    "VideoOverlay": VideoOverlay,
    "CameraMoveNode": CameraMoveNode,
    "AdvancedFolderVideoCombiner": ComfyVideoCombiner,
    "VideoImageOverlay": VideoImageOverlay,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZoomSequenceNode": "Zoom (In/Out, Easing)",
    "ZoomSequenceBatchNode": "Batch Zoom (In/Out, Easing)",
    "CameraShakeNode": "Camera Shake",
    "VideoOverlay": "Video Overlay",
    "CameraMoveNode": "Camera Move",
    "AdvancedFolderVideoCombiner": "Advanced Folder Video Combiner",
    "VideoImageOverlay": "Video Image Overlay",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

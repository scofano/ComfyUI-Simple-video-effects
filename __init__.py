from .comfy_zoom_sequence import ZoomSequenceNode as ZoomSequenceNodeSingle
from .batch_comfy_zoom_sequence import ZoomSequenceNode as ZoomSequenceNodeBatch
from .comfy_camera_shake import CameraShakeNode
from .comfy_video_overlay import VideoOverlay
from .comfy_camera_move import CameraMoveNode
from .comfy_video_combiner import ComfyVideoCombiner
from .comfy_video_image_overlay import VideoImageOverlay
from .comfy_video_overlay_batch import VideoOverlayBatch
from .comfy_audio_video_merger import MergeVideoAudioNode
from .comfy_image_transition import ImageTransitionNode
from .comfy_video_splitter import VideoSplitterNode


NODE_CLASS_MAPPINGS = {
    "ZoomSequenceNode": ZoomSequenceNodeSingle,
    "ZoomSequenceBatchNode": ZoomSequenceNodeBatch,
    "CameraShakeNode": CameraShakeNode,
    "VideoOverlay": VideoOverlay,
    "CameraMoveNode": CameraMoveNode,
    "AdvancedFolderVideoCombiner": ComfyVideoCombiner,
    "VideoImageOverlay": VideoImageOverlay,
    "VideoOverlayBatch": VideoOverlayBatch,
    "MergeVideoAudioNode": MergeVideoAudioNode,
    "ImageTransitionNode": ImageTransitionNode,
    "VideoSplitterNode": VideoSplitterNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZoomSequenceNode": "Zoom (In/Out, Easing)",
    "ZoomSequenceBatchNode": "Batch Zoom (In/Out, Easing)",
    "CameraShakeNode": "Camera Shake",
    "VideoOverlay": "Video Overlay",
    "CameraMoveNode": "Camera Move",
    "AdvancedFolderVideoCombiner": "Advanced Folder Video Combiner",
    "VideoImageOverlay": "Video Image Overlay",
    "VideoOverlayBatch": "Video Overlay (Video Path)",
    "MergeVideoAudioNode": "Merge Video + Audio (ffmpeg)",
    "ImageTransitionNode": "Image Transition",
    "VideoSplitterNode": "Video Splitter (ASS Subtitles)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

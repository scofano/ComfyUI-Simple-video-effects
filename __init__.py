from .scripts.comfy_zoom_sequence import ZoomSequenceNode as ZoomSequenceNodeSingle
from .scripts.batch_comfy_zoom_sequence import ZoomSequenceNode as ZoomSequenceNodeBatch
from .scripts.comfy_camera_shake import CameraShakeNode
from .scripts.comfy_video_overlay import VideoOverlay
from .scripts.comfy_camera_move import CameraMoveNode
from .scripts.comfy_video_combiner import ComfyVideoCombiner
from .scripts.comfy_simple_video_combiner import ComfySimpleVideoCombiner
from .scripts.comfy_video_image_overlay import VideoImageOverlay
from .scripts.comfy_video_overlay_batch import VideoOverlayBatch
from .scripts.comfy_audio_video_merger import MergeVideoAudioNode
from .scripts.comfy_image_transition import ImageTransitionNode
from .scripts.comfy_video_splitter import VideoSplitterNode
from .scripts.comfy_camera_move_video import CameraMoveVideoNode
from .scripts.comfy_camera_shake_video import CameraShakeVideoNode
from .scripts.comfy_zoom_sequence_video import ZoomSequenceVideoNode
from .scripts.comfy_close_up import CloseUpNode
from .scripts.comfy_close_up_image import CloseUpImageNode
from .scripts.comfy_video_loop_extender import VideoLoopExtenderNode
from .scripts.comfy_image_sequence_overlay import ImageSequenceOverlay
from .scripts.comfy_video_overlay_from_file import VideoOverlayFromFile
from .scripts.comfy_add_soundtrack import ComfyAddSoundtrack
from .scripts.comfy_image_audio_csv import ComfyImageAudioCSV


NODE_CLASS_MAPPINGS = {
    "ZoomSequenceNode": ZoomSequenceNodeSingle,
    "ZoomSequenceBatchNode": ZoomSequenceNodeBatch,
    "CameraShakeNode": CameraShakeNode,
    "VideoOverlay": VideoOverlay,
    "CameraMoveNode": CameraMoveNode,
    "AdvancedFolderVideoCombiner": ComfyVideoCombiner,
    "SimpleFolderVideoCombiner": ComfySimpleVideoCombiner,
    "VideoImageOverlay": VideoImageOverlay,
    "VideoOverlayBatch": VideoOverlayBatch,
    "MergeVideoAudioNode": MergeVideoAudioNode,
    "ImageTransitionNode": ImageTransitionNode,
    "VideoSplitterNode": VideoSplitterNode,
    "CameraMoveVideoNode": CameraMoveVideoNode,
    "CameraShakeVideoNode": CameraShakeVideoNode,
    "ZoomSequenceVideoNode": ZoomSequenceVideoNode,
    "CloseUpNode": CloseUpNode,
    "CloseUpImageNode": CloseUpImageNode,
    "VideoLoopExtenderNode": VideoLoopExtenderNode,
    "ImageSequenceOverlay": ImageSequenceOverlay,
    "VideoOverlayFromFile": VideoOverlayFromFile,
    "ComfyAddSoundtrack": ComfyAddSoundtrack,
    "ComfyImageAudioCSV": ComfyImageAudioCSV,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZoomSequenceNode": "Zoom (In/Out, Easing)",
    "ZoomSequenceBatchNode": "Batch Zoom (In/Out, Easing)",
    "CameraShakeNode": "Camera Shake",
    "VideoOverlay": "Video Overlay",
    "CameraMoveNode": "Camera Move",
    "AdvancedFolderVideoCombiner": "Advanced Folder Video Combiner",
    "SimpleFolderVideoCombiner": "Simple Folder Video Combiner",
    "VideoImageOverlay": "Video Image Overlay",
    "VideoOverlayBatch": "Video Overlay (Video Path)",
    "MergeVideoAudioNode": "Merge Video + Audio (ffmpeg)",
    "ImageTransitionNode": "Image Transition",
    "VideoSplitterNode": "Video Splitter (ASS Subtitles)",
    "CameraMoveVideoNode": "Camera Move (Video File)",
    "CameraShakeVideoNode": "Camera Shake (Video File)",
    "ZoomSequenceVideoNode": "Zoom Sequence (Video File)",
    "CloseUpNode": "Video - Close Up (Face Centered)",
    "CloseUpImageNode": "Image - Close Up (Face Centered)",
    "VideoLoopExtenderNode": "Video Loop Extender",
    "ImageSequenceOverlay": "Image Sequence Overlay",
    "VideoOverlayFromFile": "Video Overlay (File Input)",
    "ComfyAddSoundtrack": "Add Soundtrack",
    "ComfyImageAudioCSV": "Image Audio CSV Generator",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

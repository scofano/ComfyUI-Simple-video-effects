import os
import re
import shutil
import subprocess
from pathlib import Path

# ComfyUI folder_paths
try:
    import folder_paths
    OUTPUT_DIR = folder_paths.get_output_directory()
except ImportError:
    OUTPUT_DIR = Path("output")

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
if not FFMPEG or not FFPROBE:
    raise RuntimeError("ffmpeg and ffprobe not found on PATH.")

def parse_ass_file(ass_path):
    """Parse ASS file and extract dialogues with start, end, text"""
    dialogues = []
    in_events = False
    
    print(f"Parsing ASS file: {ass_path}")
    
    with open(ass_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Detect Start of Events section
            if line == '[Events]':
                in_events = True
                continue
            
            # Flexible Dialogue detection
            if in_events and line.startswith('Dialogue:'):
                parts = line.split(',', 9)
                if len(parts) >= 10:
                    start = parts[1]
                    end = parts[2]
                    text = parts[9]
                    
                    # Clean text: remove ASS tags and replace ellipsis
                    # Regex explanation: match { followed by anything not } followed by }
                    text = re.sub(r'\{[^}]*\}', '', text)
                    text = text.replace('…', '.')  # treat ellipsis char as period
                    
                    clean_text = text.strip()
                    # Debug print to see what the script sees
                    # print(f"Found Line: {clean_text} | End Time: {end}")
                    
                    dialogues.append({
                        'start': start,
                        'end': end,
                        'text': clean_text
                    })
    return dialogues

def time_to_seconds(time_str):
    """Convert ASS time format (0:00:00.00) to seconds"""
    try:
        h, m, s = time_str.split(':')
        return int(h) * 3600 + int(m) * 60 + float(s)
    except ValueError:
        print(f"Error parsing time: {time_str}")
        return 0.0

def get_video_duration(video_path):
    """Get video duration in seconds using ffprobe"""
    cmd = [
        FFPROBE,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    import json
    data = json.loads(result.stdout)
    return float(data['format']['duration'])

def get_video_fps(video_path):
    """Get video fps using ffprobe"""
    cmd = [
        FFPROBE,
        "-v", "quiet",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "csv=p=0",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    fps_str = result.stdout.strip()
    if '/' in fps_str:
        num, den = fps_str.split('/')
        return float(num) / float(den)
    return float(fps_str)

class VideoSplitterNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": ""}),
                "ass_path": ("STRING", {"default": ""}),
                "divider_chars": ("STRING", {"default": ".!?"}),
                "folder_prefix": ("STRING", {"default": "split_video"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_folder",)
    FUNCTION = "split_video"
    CATEGORY = "Simple Video Effects"
    OUTPUT_NODE = True

    def split_video(self, video_path, ass_path, divider_chars, folder_prefix):
        if not os.path.exists(video_path):
            raise ValueError(f"Video file not found: {video_path}")
        if not os.path.exists(ass_path):
            raise ValueError(f"ASS file not found: {ass_path}")

        # Parse ASS
        dialogues = parse_ass_file(ass_path)
        print(f"Total dialogues parsed: {len(dialogues)}")

        split_points = []
        for dialogue in dialogues:
            text = dialogue['text']
            
            # Logic to identify split points
            if text and text[-1] in divider_chars:
                # Don't split if ends with .. (multiple dots)
                # Check for ".." at the end to avoid splitting on "..." or ".."
                if text[-1] == '.' and len(text) > 1 and text[-2] == '.':
                    continue
                
                end_sec = time_to_seconds(dialogue['end'])
                split_points.append(end_sec)
                print(f"Split point found at {end_sec}s (Text: {text})")

        # FIX: Deduplicate FIRST, then SORT. 
        # Previous code sorted then set(), which destroys order.
        split_points = sorted(list(set(split_points)))

        # Get video info
        duration = get_video_duration(video_path)
        
        # Determine times [0, split1, split2, ..., duration]
        times = [0.0] + split_points + [duration]
        
        # Sanitize list: ensure monotonic increase and unique
        times = sorted(list(set(times)))
        
        print(f"Cut points: {times}")

        # Create output folder
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_folder = output_dir / folder_prefix
        if output_folder.exists():
            counter = 1
            while True:
                folder_name = f"{folder_prefix}_{counter}"
                output_folder = output_dir / folder_name
                if not output_folder.exists():
                    break
                counter += 1
        output_folder.mkdir()

        video_name = Path(video_path).stem
        file_counter = 0
        
        for i in range(len(times) - 1):
            start_time = times[i]
            end_time = times[i + 1]

            # Avoid tiny segments or negative duration
            if end_time - start_time < 0.1:
                continue

            output_file = output_folder / f"{video_name}_[{file_counter:03d}].mp4"
            duration_segment = end_time - start_time
            
            print(f"Exporting segment {file_counter}: {start_time} to {end_time}")

            cmd = [
                FFMPEG,
                "-y",
                "-ss", str(start_time),
                "-i", video_path,
                "-t", str(duration_segment),
                "-c:v", "libx264",
                "-c:a", "aac",
                "-preset", "ultrafast",
                "-avoid_negative_ts", "make_zero",
                str(output_file)
            ]
            subprocess.run(cmd, check=True)
            file_counter += 1

        return (str(output_folder),)
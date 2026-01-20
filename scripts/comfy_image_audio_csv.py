import os
import csv

# extensões comuns (pode adicionar mais se quiser)
IMAGE_EXTENSIONS = {
    '.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif', '.webp'
}

AUDIO_EXTENSIONS = {
    '.mp3', '.wav', '.ogg', '.flac', '.aac', '.m4a', '.wma'
}


class ComfyImageAudioCSV:
    """
    A ComfyUI-compatible node that:
    - Scans directories for image and audio files
    - Sorts them ascending
    - Pairs images with audios and saves to CSV with ';' delimiter
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_directory_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Path to image directory"
                }),
                "audio_directory_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Path to audio directory"
                }),
                "output_csv_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Full path for output CSV file (e.g., C:/output/data.csv)"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("csv_path",)
    FUNCTION = "create_csv"
    CATEGORY = "Simple Video Effects"
    OUTPUT_NODE = True

    def create_csv(
        self,
        image_directory_path: str,
        audio_directory_path: str,
        output_csv_path: str,
    ) -> tuple:

        # Resolve and validate paths
        image_dir = os.path.abspath(image_directory_path)
        audio_dir = os.path.abspath(audio_directory_path)
        output_csv = os.path.abspath(output_csv_path)

        if not os.path.isdir(image_dir):
            raise ValueError(f"Image directory does not exist: {image_dir}")
        if not os.path.isdir(audio_dir):
            raise ValueError(f"Audio directory does not exist: {audio_dir}")

        # Ensure output directory exists
        output_dir = os.path.dirname(output_csv)
        os.makedirs(output_dir, exist_ok=True)

        images = []
        audios = []

        # Collect image files
        for file in os.listdir(image_dir):
            full_path = os.path.join(image_dir, file)
            if not os.path.isfile(full_path):
                continue
            ext = os.path.splitext(file)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                images.append(full_path)

        # Collect audio files
        for file in os.listdir(audio_dir):
            full_path = os.path.join(audio_dir, file)
            if not os.path.isfile(full_path):
                continue
            ext = os.path.splitext(file)[1].lower()
            if ext in AUDIO_EXTENSIONS:
                audios.append(full_path)

        # Sort ascending
        images.sort()
        audios.sort()

        # Write CSV
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=';')
            for img, aud in zip(images, audios):
                writer.writerow([img, aud])

        print(f'CSV generated successfully: {output_csv}')
        return (output_csv,)

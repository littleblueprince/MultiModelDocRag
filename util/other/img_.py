# -*- coding: utf-8 -*-
# @Time    : 2024/12/12 23:56
# @Author  : blue
# @Description : 
import os

os.environ["HF_HOME"] = "/data1/zch/tmp"
os.environ["TMPDIR"] = "/data1/zch/tmp"

import os
from PIL import Image

def normalize_images(input_folder, output_folder, target_size=(256, 256), target_format="png"):
    """
    Normalize images in a folder by resizing and converting format.

    :param input_folder: Path to the folder containing input images.
    :param output_folder: Path to the folder to save normalized images.
    :param target_size: Tuple indicating target resolution (width, height).
    :param target_format: Target image format (e.g., 'png', 'jpeg').
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        filepath = os.path.join(input_folder, filename)

        # Skip non-image files
        if not os.path.isfile(filepath):
            continue

        try:
            with Image.open(filepath) as img:
                # Resize the image
                img = img.resize(target_size, Image.ANTIALIAS)

                # Convert to RGB if not already in RGB mode
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Save the image in the target format
                base_name, _ = os.path.splitext(filename)
                output_path = os.path.join(output_folder, f"{base_name}.{target_format}")
                img.save(output_path, format=target_format.upper())

                print(f"Processed: {filename} -> {output_path}")
        except Exception as e:
            print(f"Failed to process {filename}: {e}")

# Example usage
input_folder = "/mnt/sdb/zch/LLaMA-Factory/data/qwen2_vl_ft/mllm_data"
output_folder = "/mnt/sdb/zch/LLaMA-Factory/data/qwen2_vl_ft/mllm_data_normalized"
normalize_images(input_folder, output_folder, target_size=(256, 256), target_format="png")

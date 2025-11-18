import hashlib
import logging
import os
from datetime import datetime

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def get_image_hash(image_array: np.ndarray, algorithm: str = 'sha256') -> str:
    """
    Computes a hash for a NumPy image array to verify its integrity.

    Args:
        image_array: The NumPy array of the image.
        algorithm: The hashing algorithm to use (e.g., 'sha256', 'md5').

    Returns:
        The hexadecimal hash string.
    """
    hasher = hashlib.new(algorithm)
    hasher.update(image_array.tobytes())
    return hasher.hexdigest()


def save_captured_images(color_bgr_image: np.ndarray, depth_image: np.ndarray, save_dir: str = "src/image_captured"):
    """
    Saves the raw color and depth images to a specified directory with timestamps.

    Args:
        color_bgr_image: The BGR color image as a NumPy array.
        depth_image: The depth image as a NumPy array.
        save_dir: The directory to save the images in.
    """
    try:
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        color_filename = os.path.join(save_dir, f"color_{timestamp}.png")
        depth_filename = os.path.join(save_dir, f"depth_{timestamp}.npy")
        cv2.imwrite(color_filename, color_bgr_image)
        np.save(depth_filename, depth_image)
        logger.info(f"Saved captured images: {color_filename}, {depth_filename}")
    except Exception as e:
        logger.error(f"Failed to save captured images: {e}", exc_info=True)

import hashlib

import numpy as np

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

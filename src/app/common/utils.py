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

def format_3d_coordinates(point_3d: list) -> dict:
    """Formats 3D coordinates as a dictionary."""
    if not point_3d or len(point_3d) != 3:
        return {}
    
    coord_text = {"x": round(point_3d[0], 5),  
                  "y": round(point_3d[1], 5), 
                  "z": round(point_3d[2], 5)}
    
    return coord_text

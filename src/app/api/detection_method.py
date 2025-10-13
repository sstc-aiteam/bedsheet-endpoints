from enum import Enum


class DetectionMethod(str, Enum):
    """Enum for selecting the keypoint detection method."""
    RGB = "rgb"
    DEPTH = "depth"
    METACLIP = "metaclip"
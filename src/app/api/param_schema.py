from enum import Enum

class DetectionMethod(str, Enum):
    """Enum for selecting the keypoint detection method."""
    RGB = "rgb"
    DEPTH = "depth"
    METACLIP = "metaclip"

class ModelType(str, Enum):
    """Enum for selecting the MetaCLIP model type."""
    BEDSHEET = "bedsheet"
    MATTRESS = "mattress"
    FITTED_SHEET = "fitted_sheet"


from pydantic import BaseModel
from typing import List, Optional

class Keypoint(BaseModel):
    x: int
    y: int
    depth_m: float


class ProcessedImagePayload(BaseModel):
    keypoints: Optional[List[Keypoint]] = None
    processed_image: str
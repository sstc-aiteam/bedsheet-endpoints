from enum import Enum

class DetectionMethod(str, Enum):
    """Enum for selecting the keypoint detection method."""
    METACLIP = "metaclip"
    RGB = "rgb"
    DEPTH = "depth"

class ModelType(str, Enum):
    """Enum for selecting the MetaCLIP model type."""
    MATTRESS = "mattress"
    FITTED_SHEET = "fitted_sheet"
    BEDSHEET = "bedsheet"


from pydantic import BaseModel
from typing import List, Optional

class Keypoint(BaseModel):
    x: int
    y: int
    depth_m: float


class ProcessedImagePayload(BaseModel):
    keypoints: Optional[List[Keypoint]] = None
    processed_image: str
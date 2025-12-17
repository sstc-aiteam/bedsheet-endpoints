from enum import Enum


class DetectionMethod(str, Enum):
    """Enum for selecting the keypoint detection method."""
    METACLIP = "metaclip"
    RGB = "rgb"
    DEPTH = "depth"
    QUAD_D = "quad_d"
    QUAD_YC = "quad_yc"
    QUAD_YC_SB = "quad_yc_sb"

class ModelType(str, Enum):
    """Enum for selecting the MetaCLIP model type."""
    MATTRESS = "mattress"
    FITTED_SHEET = "fitted_sheet"
    FITTED_SHEET_INVERSE = "fitted_sheet_inverse"
    BEDSHEET = "bedsheet"


from pydantic import BaseModel, Field
from typing import List, Optional, Tuple


class RGBDepth(BaseModel):
    x: int
    y: int
    depth_m: float

class Keypoint3D(BaseModel):
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None

class Keypoint(BaseModel):
    rgbd: RGBDepth
    point3d_m: Keypoint3D

class ProcessedImagePayload(BaseModel):
    keypoints: Optional[List[Keypoint]] = None
    processed_image: str

class DetectionParams(BaseModel):
    ntile_divisor: float = 4.0

class BoundingBox(BaseModel):
    x_min: int
    y_min: int
    x_max: int
    y_max: int
class FittedSheetClassificationResponse(BaseModel):
    label: str
    pred: int
    conf: float
    bbox: Optional[BoundingBox] = None
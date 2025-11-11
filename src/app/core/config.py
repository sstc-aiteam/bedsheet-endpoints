from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    # Model paths - assumes running from project root
    YOLO_MODEL_PATH: str = "weights/yolo_finetuned/best.pt"
    KEYPOINT_MODEL_PATH: str = "weights_depth/keypoint_model_vit_depth.pth"
    YOLO_BASE_MODEL_PATH: str = 'yolov8l.pt'

    # Detection parameters
    ALLOWED_CLASSES: List[int] = [1]  # Only bedsheet
    KEYPOINT_THRESHOLD: float = 0.0003

    # RealSense depth scale (can be overridden by environment variables)
    # Note: This is device-specific.
    DEPTH_SCALE: float = 0.0010000000474974513

settings = Settings()
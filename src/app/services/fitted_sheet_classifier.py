"""
Service for classifying fitted sheets using a CNN classifier and YOLO segmenter.
Refactored from src/classify_fitted_sheet.py.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Dict, Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Attempt imports assuming src is the root (standard for this project structure)
from app.api.param_schema import FittedSheetClassificationResponse
from app.models.fitted_sheet_cnn_classifier import FittedSheetCNNClassifier
from app.models.yolo_segmenter import YoloFittedSheetSegmenter, crop_masked_square_rgb
from app.api.param_schema import BoundingBox


logger = logging.getLogger(__name__)


class FittedSheetClassifierService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.segmenter = None
        self.label_map = {0: "purple", 1: "pink", 2: "off-white"}

        # Default paths - can be overridden via environment variables
        self.yolo_weights = "weights/yolo_finetuned/best.pt"
        self.classifier_checkpoint = "weights/fitted_sheet_classifier/best.pth"
        self.labels_json = "weights/fitted_sheet_classifier/labels.json"
        
        self.imgsz = 640
        self.seg_conf = 0.25
        self.crop_size = 224

        self.tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        self.load_models()

    def load_models(self):
        # Load Label Map
        if os.path.exists(self.labels_json):
            try:
                with open(self.labels_json, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                self.label_map = {int(k): str(v) for k, v in raw.items()}
                logger.info(f"Loaded label map from {self.labels_json}")
            except Exception as e:
                logger.error(f"Failed to load labels.json: {e}")

        # Load Classifier
        if os.path.exists(self.classifier_checkpoint):
            try:
                # Assuming num_classes matches label_map length, default to 3 if map is default
                num_classes = len(self.label_map)
                self.model = FittedSheetCNNClassifier(num_classes=num_classes).to(self.device)
                ckpt = torch.load(self.classifier_checkpoint, map_location=self.device)
                state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
                self.model.load_state_dict(state)
                self.model.eval()
                logger.info(f"Loaded FittedSheetCNNClassifier from {self.classifier_checkpoint}")
            except Exception as e:
                logger.error(f"Failed to load classifier: {e}")
                self.model = None
        else:
            logger.warning(f"Classifier checkpoint not found at {self.classifier_checkpoint}")

        # Load Segmenter
        if os.path.exists(self.yolo_weights):
            try:
                self.segmenter = YoloFittedSheetSegmenter(
                    self.yolo_weights,
                    allowed_class_ids=(1,),
                    conf=self.seg_conf,
                    imgsz=self.imgsz,
                )
                logger.info(f"Loaded YoloFittedSheetSegmenter from {self.yolo_weights}")
            except Exception as e:
                logger.error(f"Failed to load segmenter: {e}")
                self.segmenter = None
        else:
            logger.warning(f"YOLO weights not found at {self.yolo_weights}")

    def classify(self, img_rgb: np.ndarray) -> Dict:
        if self.model is None:
            return {"error": "Classifier model not loaded"}

        bbox = None
        crop_rgb = None

        # Attempt segmentation
        if self.segmenter:
            seg_res = self.segmenter.segment_largest(img_rgb)
            if seg_res:
                crop_rgb = crop_masked_square_rgb(
                    img_rgb,
                    seg_res.mask01,
                    seg_res.bbox_xyxy,
                    pad_ratio=0.08,
                    out_size=self.crop_size,
                )
                bbox = seg_res.bbox_xyxy

        # Fallback if segmentation failed or not available
        if crop_rgb is None:
            crop_rgb = cv2.resize(img_rgb, (self.crop_size, self.crop_size), interpolation=cv2.INTER_LINEAR)

        # Inference
        x = self.tfm(Image.fromarray(crop_rgb)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            prob = torch.softmax(logits, dim=1)[0]
            pred = int(prob.argmax().item())
            conf = float(prob[pred].item())

        label = self.label_map.get(pred, str(pred))

        return {
            "label": label,
            "pred": pred,
            "conf": conf,
            "bbox": bbox,
        }

fitted_sheet_classifier_service = FittedSheetClassifierService()
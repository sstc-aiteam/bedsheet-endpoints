import os
import cv2
import numpy as np
import torch
import logging
from typing import List, Tuple, Optional

from app.models.clip_heatmap_model import ClipHeatmapModel
from app.models.utils import thresholded_locations, combine_nearby_peaks
from app.core.config import settings

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("ultralytics is not installed. Segmentation will be skipped.")


class MetaClipKeypointDetectorService:
    """
    A service for keypoint detection using the Meta CLIP heatmap model.
    This service is adapted from the `SimpleKeypointInference` class.
    """

    def __init__(self, model_type: str = 'bedsheet'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        self.model_config = self._get_model_config(model_type)
        self.model = self._load_model()
        self.yolo_model = self._load_yolo_model()

    def _get_model_config(self, model_type: str):
        # In a production system, this could come from a config file.
        configs = {
            'bedsheet': {
                'path': 'weights/meta_clip_style_bedsheet_post_pretrained',
                'lora_r': 16, 'lora_alpha': 32, 'image_size': 256, 'use_text_prior': True,
                'seg_classes': [3]
            },
            'mattress': {
                'path': 'weights/meta_clip_style_mattress_post_original',
                'lora_r': 16, 'lora_alpha': 32, 'image_size': 256, 'use_text_prior': True,
                'seg_classes': [0, 1, 2, 3]
            },
            'fitted_sheet': {
                'path': 'weights/meta_clip_style_fitted_sheet_post_original',
                'lora_r': 16, 'lora_alpha': 32, 'image_size': 256, 'use_text_prior': True,
                'seg_classes': [1]
            }
        }
        if model_type not in configs:
            raise ValueError(f"Unknown model_type: {model_type}. Available types are {list(configs.keys())}")
        return configs[model_type]

    def _load_model(self):
        """Load the trained ClipHeatmapModel."""
        model_path = self.model_config['path']
        logging.info(f"Loading {self.model_type} model from {model_path}")

        model = ClipHeatmapModel(
            model_name='facebook/metaclip-b16-fullcc2.5b',
            image_size=self.model_config['image_size'],
            use_lora=True,
            lora_r=self.model_config['lora_r'],
            lora_alpha=self.model_config['lora_alpha'],
            use_text_prior=self.model_config['use_text_prior']
        )

        complete_model_path = os.path.join(model_path, 'complete_model.pth')
        if not os.path.exists(complete_model_path):
            raise FileNotFoundError(f"Complete model not found at: {complete_model_path}")

        checkpoint = torch.load(complete_model_path, map_location=self.device)
        model.load_state_dict(checkpoint)
        model.to(self.device)
        model.eval()
        logging.info(f"✅ Loaded {self.model_type} model successfully to {self.device}")
        return model

    def _load_yolo_model(self):
        """Load the YOLO model for segmentation."""
        if not YOLO_AVAILABLE:
            return None
        yolo_path = 'weights/yolo_finetuned/best_2.pt'
        if os.path.exists(yolo_path):
            try:
                yolo_model = YOLO(yolo_path)
                logging.info(f"✅ Loaded YOLO model from {yolo_path}")
                return yolo_model
            except Exception as e:
                logging.error(f"Failed to load YOLO model: {e}")
                return None
        logging.warning(f"⚠️ YOLO model not found at {yolo_path}, proceeding without segmentation")
        return None

    def _apply_segmentation(self, image: np.ndarray) -> np.ndarray:
        """Apply YOLO segmentation to the input image."""
        if self.yolo_model is None:
            return image

        try:
            results = self.yolo_model(image, task="segment", verbose=False)
            if not (results and results[0].masks):
                return image

            mask_all = np.zeros(image.shape[:2], dtype=np.uint8)
            masks = results[0].masks.data.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()

            for mask, cls_id in zip(masks, classes):
                if int(cls_id) in self.model_config['seg_classes']:
                    mask_binary = (mask > 0.5).astype(np.uint8) * 255
                    mask_resized = cv2.resize(mask_binary, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                    mask_all = cv2.bitwise_or(mask_all, mask_resized)

            if np.any(mask_all > 0):
                logging.info(f"✅ Applied YOLO segmentation for classes {self.model_config['seg_classes']}")
                masked_image = image.copy()
                masked_image[mask_all == 0] = 0
                return masked_image
        except Exception as e:
            logging.error(f"⚠️ YOLO processing failed: {e}")

        return image

    def detect_keypoints(self, color_image: np.ndarray, depth_image: np.ndarray) -> Tuple[np.ndarray, List[dict]]:
        """Detects keypoints in the given color image."""
        orig_h, orig_w = color_image.shape[:2]

        masked_image = self._apply_segmentation(color_image)

        target_size = self.model_config['image_size']
        img_resized = cv2.resize(masked_image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        image_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            heatmap = self.model(image_tensor).squeeze().cpu().numpy()

        heatmap /= heatmap.max()
        peaks = thresholded_locations(heatmap, threshold=0.3)
        combined_peaks = combine_nearby_peaks(peaks, distance_threshold=10)

        scale_x, scale_y = orig_w / target_size, orig_h / target_size
        final_keypoints = []
        processed_image = color_image.copy()

        for p in combined_peaks:
            y, x = p
            orig_x, orig_y = int(x * scale_x), int(y * scale_y)
            distance_m = float(depth_image[orig_y, orig_x]) * settings.DEPTH_SCALE
            final_keypoints.append({"x": orig_x, "y": orig_y, "depth_m": round(distance_m, 5)})
            cv2.circle(processed_image, (orig_x, orig_y), 5, (0, 255, 0), -1)

        return processed_image, final_keypoints

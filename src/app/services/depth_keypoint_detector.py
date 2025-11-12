import numpy as np
import cv2
import torch
import logging
from scipy.ndimage import label
from ultralytics import YOLO

from app.models.yolo_vit import HybridKeypointNet
from app.services.realsense_capture import RealSenseCaptureService
from app.models.utils import YoloBackbone
from app.core.config import settings
from app.common.utils import format_3d_coordinates


class DepthKeypointDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.yolo_model_finetuned = None
        self.keypoint_model = None
        logging.info(f"Using device: {self.device}")

    def load_models(self):
        """Loads the ML models into memory."""
        logging.info("Loading YOLO segmentation model...")
        self.yolo_model_finetuned = YOLO(settings.YOLO_MODEL_PATH)

        logging.info("Loading keypoint detection model...")
        yolo_model = YOLO(settings.YOLO_BASE_MODEL_PATH)
        backbone_seq = yolo_model.model.model[:10]
        backbone = YoloBackbone(backbone_seq, selected_indices=list(range(10)))
        in_channels_list = [f.shape[1] for f in backbone(torch.randn(1, 3, 128, 128))]
        keypoint_net = HybridKeypointNet(backbone, in_channels_list)
        model = keypoint_net.to(self.device)
        compiled_model = torch.compile(model)
        compiled_model.load_state_dict(torch.load(settings.KEYPOINT_MODEL_PATH, map_location=self.device))
        compiled_model.eval()
        self.keypoint_model = compiled_model
        logging.info("All models loaded successfully!")

    def _extract_mask(self, image: np.ndarray) -> np.ndarray:
        """Extracts a segmentation mask from the image."""
        results = self.yolo_model_finetuned(image, task="segment")[0]
        h, w = image.shape[:2]
        mask_all = np.zeros((h, w), dtype=np.uint8)

        for r in results:
            if r.masks is None:
                continue
            masks = r.masks.data.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            for m, cls_id in zip(masks, classes):
                if int(cls_id) not in settings.ALLOWED_CLASSES:
                    continue
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                mask_all = cv2.bitwise_or(mask_all, (m * 255).astype(np.uint8))
        
        return mask_all

    def detect_keypoints(self,
                         color_image: np.ndarray,
                         depth_image: np.ndarray,
                         rs_service: RealSenseCaptureService = None
                         ):
        """The main detection pipeline."""
        c_copy = color_image.copy()
        d_copy = depth_image.copy()
        
        mask = self._extract_mask(c_copy)
        final_keypoints = []
        
        if np.sum(mask) > 0:
            c_copy[mask == 0] = 0
            d_copy[mask == 0] = 0
            H, W = c_copy.shape[:2]
            
            d_copy_meters = d_copy * settings.DEPTH_SCALE
            depth_visualized = self._depth_map_to_image(d_copy_meters)
            depth_image_resized = cv2.resize(depth_visualized, (128, 128), interpolation=cv2.INTER_AREA)
            depth_image_resized_3ch = cv2.cvtColor(depth_image_resized, cv2.COLOR_GRAY2BGR)

            with torch.no_grad():
                batch_image = torch.Tensor(np.transpose(depth_image_resized_3ch, (2, 0, 1))).unsqueeze(0).to(self.device)
                outputs = self.keypoint_model(batch_image)
                kp = outputs[0].cpu().numpy()[0, :, :]
                points = self._thresholded_locations(kp, settings.KEYPOINT_THRESHOLD)
                
                for p in points:
                    y, x = p
                    orig_x = int(x * (W / 128))
                    orig_y = int(y * (H / 128))

                    distance_meters = float(depth_image[orig_y, orig_x]) * settings.DEPTH_SCALE
                    point_3d = rs_service.deproject_pixel_to_point(orig_x, orig_y, distance_meters) if rs_service else []
                    final_keypoints.append({
                        "color": {"x": orig_x, 
                                  "y": orig_y, 
                                  "depth_m": round(distance_meters, 5)},
                        "point3d_m": format_3d_coordinates(point_3d)
                    })
                    cv2.circle(color_image, (orig_x, orig_y), 5, (0, 0, 255), -1)
            
        return color_image, final_keypoints

    @staticmethod
    def _depth_map_to_image(depth_map: np.ndarray) -> np.ndarray:
        """Converts a raw depth map to an 8-bit grayscale image."""
        valid = depth_map > 0
        if not np.any(valid):
            return np.zeros_like(depth_map, dtype=np.uint8)

        dmin, dmax = np.min(depth_map[valid]), np.max(depth_map)
        if dmax <= dmin:
            return np.zeros_like(depth_map, dtype=np.uint8)

        depth_img = (depth_map - dmin) / (dmax - dmin) * 255
        return np.clip(depth_img, 0, 255).astype(np.uint8)

    @staticmethod
    def _thresholded_locations(data_2d: np.ndarray, threshold: float) -> list:
        """Finds centroids of connected components above a threshold."""
        thresholded_2d = (data_2d >= threshold).astype(np.uint8)
        structure = np.ones((3, 3), dtype=int)
        labeled, num_features = label(thresholded_2d, structure=structure)
        
        centroids = []
        if num_features > 0:
            for mesh_label in range(1, num_features + 1):
                positions = np.argwhere(labeled == mesh_label)
                centroid = positions.mean(axis=0)
                centroids.append(centroid)
        return centroids

# Create a single, reusable instance of the detector
depth_detector_service = DepthKeypointDetector()
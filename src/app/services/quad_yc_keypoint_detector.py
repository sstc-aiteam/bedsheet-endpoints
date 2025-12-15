import logging
import os
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
import itertools
from scipy.ndimage import distance_transform_edt

from ultralytics import YOLO

from app.api.param_schema import DetectionParams
from app.core.config import settings
from app.common.utils import format_3d_coordinates
from app.services.realsense_capture import RealSenseCaptureService

logger = logging.getLogger(__name__)

# --- Helper functions from box_v3.py ---

def polygon_area(pts: np.ndarray) -> float:
    """Shoelace formula to calculate polygon area."""
    if pts.shape[0] < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def merge_close_points(points: List[np.ndarray], threshold: int) -> List[np.ndarray]:
    """Merge points that are closer than the threshold."""
    merged = []
    for p in points:
        if not any(np.linalg.norm(p - m) < threshold for m in merged):
            merged.append(p)
    return merged

def _find_nearest_nonzero_depth(point: Tuple[int, int], depth_image: np.ndarray, mask: np.ndarray) -> Optional[Tuple[int, int]]:
    """
    If the given point has zero depth, find the nearest point within the mask that has a non-zero depth.
    """
    x, y = point
    h, w = depth_image.shape

    # 1. Check if the point is valid and has depth
    if 0 <= y < h and 0 <= x < w and depth_image[y, x] > 0:
        return point

    # 2. Create a map of all valid depth locations within the mask
    valid_depth_mask = (depth_image > 0) & (mask > 0)
    if not np.any(valid_depth_mask):
        return None # No valid depth points exist in the mask

    # 3. Find the coordinates of the closest valid depth point
    _, indices = distance_transform_edt(np.logical_not(valid_depth_mask), return_indices=True)
    return indices[1, y, x], indices[0, y, x]

class QuadYCKeypointDetectorService:
    def __init__(self, model_path: str = "weights/yolo_finetuned/best.pt"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self._load_model(model_path)
        
        # --- Constants from main_seg.py ---
        self.conf_thres = 0.5
        self.bedbag_class_id = 1
        self.morph_kernel = np.ones((5, 5), np.uint8)
        self.min_area = 500

    def _load_model(self, model_path: str):
        if not os.path.exists(model_path):
            logger.error(f"YOLO model for Legacy Quad detection not found at {model_path}")
            raise FileNotFoundError(f"YOLO model not found at {model_path}")
        try:
            model = YOLO(model_path)
            model.to(self.device)
            
            # This custom preprocess function skips the BGR->RGB conversion,
            # as the model was likely trained on BGR images from OpenCV.
            def preprocess_bgr(img):
                return model.transforms(img)
            model.preprocess = preprocess_bgr

            logger.info(f"Legacy Quad detector loaded YOLO model from {model_path} to {self.device}")
            return model
        except Exception as e:
            logger.error(f"Failed to load YOLO model from {model_path}: {e}", exc_info=True)
            raise

    def _segment_bedbag(self, image_bgr: np.ndarray) -> np.ndarray:
        """Performs segmentation to get the bedbag mask, adapted from main_seg.py."""
        img_resized = cv2.resize(image_bgr, (640, 640))
        results = self.model(img_resized)

        bedbag_mask = np.zeros((640, 640), dtype=np.uint8)

        for r in results:
            if r.masks is None or r.boxes is None:
                continue

            for i in range(len(r.boxes)):
                cls = int(r.boxes.cls[i].item())
                conf = float(r.boxes.conf[i].item())

                if cls == self.bedbag_class_id and conf >= self.conf_thres:
                    mask = r.masks.data[i].cpu().numpy().astype(np.uint8)
                    
                    # Refine mask
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morph_kernel)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.morph_kernel)

                    # Remove small noise by finding connected components
                    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
                    for j in range(1, num_labels):
                        if stats[j, cv2.CC_STAT_AREA] >= self.min_area:
                            bedbag_mask[labels == j] = 255
        
        # Resize mask back to original image dimensions
        orig_h, orig_w = image_bgr.shape[:2]
        return cv2.resize(bedbag_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    def _find_max_area_quad(self, mask: np.ndarray, depth_image: np.ndarray) -> Optional[np.ndarray]:
        """Finds the quadrilateral with the maximum area from the mask, adapted from box_v3.py."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        possible_corners = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 2000:
                continue
            hull = cv2.convexHull(cnt)
            epsilon = 0.05 * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)
            for point in approx:
                x, y = point[0]
                # If point has no depth, find the nearest valid one within the mask
                valid_point = _find_nearest_nonzero_depth((x, y), depth_image, mask)
                if valid_point:
                    # Use the point with valid depth
                    possible_corners.append(np.array(valid_point))


        if len(possible_corners) < 4:
            return None

        merged_corners = merge_close_points(possible_corners, threshold=50)
        if len(merged_corners) < 4:
            return None

        best_quad, max_area = None, 0
        for quad_candidate in itertools.combinations(merged_corners, 4):
            area = polygon_area(np.array(quad_candidate))
            if area > max_area:
                max_area = area
                best_quad = np.array(quad_candidate, dtype=np.int32)
        
        return best_quad

    def detect_keypoints(self, color_image: np.ndarray, depth_image: np.ndarray, rs_service: RealSenseCaptureService = None, params: Optional[DetectionParams] = None) -> Tuple[np.ndarray, List[dict]]:
        """Detects four corner keypoints by segmenting and finding the max area quadrilateral."""
        # Convert RGB from endpoint to BGR for the model
        color_image_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

        mask = self._segment_bedbag(color_image_bgr)
        if not np.any(mask):
            logger.warning("Legacy Quad method: Segmentation did not find a bedbag.")
            return color_image, []

        quad = self._find_max_area_quad(mask, depth_image)
        if quad is None:
            logger.warning("Legacy Quad method: Could not infer quadrilateral from mask.")
            return color_image, []

        final_keypoints = []
        processed_image = color_image.copy()
        cv2.polylines(processed_image, [quad], isClosed=True, color=(0, 255, 0), thickness=2)

        for point in quad:
            x, y = int(point[0]), int(point[1])
            depth_m = 0.0
            if 0 <= y < depth_image.shape[0] and 0 <= x < depth_image.shape[1]:
                depth_m = float(depth_image[y, x]) * settings.DEPTH_SCALE
            
            point_3d = rs_service.deproject_pixel_to_point(x, y, depth_m) if rs_service else []
            final_keypoints.append({
                "rgbd": {"x": x, "y": y, "depth_m": round(depth_m, 5)},
                "point3d_m": format_3d_coordinates(point_3d)
            })
            cv2.circle(processed_image, (x, y), 6, (0, 255, 0), -1)

        return processed_image, final_keypoints

# Instantiate the service for the API
try:
    quadYC_detector_service = QuadYCKeypointDetectorService()
except Exception as e:
    logger.error(f"Failed to initialize LegacyQuadDetectorService: {e}", exc_info=True)
    quadYC_detector_service = None


if __name__ == '__main__':
    import argparse
    import glob
    import json

    # Configure basic logging for command-line use
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="Detect bedsheet corner keypoints in a folder of images.")
    parser.add_argument("--input_folder", help="Path to the folder containing input color images (JPG, PNG).")
    parser.add_argument("--output_folder", help="Path to the folder where processed images and keypoint data will be saved.")
    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
        logger.info(f"Created output directory: {args.output_folder}")

    try:
        detector = QuadYCKeypointDetectorService()
    except Exception as e:
        logger.error(f"Failed to initialize the detector. Exiting. Error: {e}")
        exit(1)

    image_extensions = ('*.jpg', '*.jpeg', '*.png')
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(args.input_folder, ext)))

    logger.info(f"Found {len(image_paths)} images to process.")

    for img_path in image_paths:
        # if not any(id in img_path for id in ['717575', '839856', '831958', '057626']):
        #     continue

        try:
            logger.info(f"Processing {img_path}...")
            color_image = cv2.imread(img_path)
            if color_image is None:
                logger.warning(f"Could not read image {img_path}, skipping.")
                continue

            # Look for a corresponding .npy depth file
            base_filename, _ = os.path.splitext(img_path)
            color_fn = base_filename.split('/')[-1]
            depth_fn = color_fn.replace('color', 'depth')
            depth_path = f"{base_filename.replace(color_fn, depth_fn)}.npy"
            
            if os.path.exists(depth_path):
                depth_image = np.load(depth_path)
                logger.info(f"Loaded depth image from {depth_path}")
            else:
                # Create a dummy depth image if not available
                logger.warning(f"Depth file not found at {depth_path}. Using dummy depth.")
                depth_image = np.zeros((color_image.shape[0], color_image.shape[1]), dtype=np.uint16)

            processed_image, keypoints = detector.detect_keypoints(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB), depth_image)

            base_filename = os.path.splitext(os.path.basename(img_path))[0]
            cv2.imwrite(os.path.join(args.output_folder, f"{base_filename}_processed.png"), cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))
            with open(os.path.join(args.output_folder, f"{base_filename}_keypoints.json"), 'w') as f:
                json.dump(keypoints, f, indent=4)
        except Exception as e:
            logger.error(f"An error occurred while processing {img_path}: {e}", exc_info=True)
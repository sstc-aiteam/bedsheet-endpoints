import logging
import os
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from app.core.config import settings
from app.common.utils import format_3d_coordinates
from app.services.realsense_capture import RealSenseCaptureService

logger = logging.getLogger(__name__)

# --- Constants adapted from the script ---
ALLOWED_CLASSES = (1, 3)  # Fitted sheet classes
MAX_CONTOUR_POINTS = 200
MAX_CANDIDATE_POINTS = 20
UNIFORM_SAMPLE_POINTS = 40


# --- Helper functions from fitted_sheet_homography_keypoint_extraction.py ---

def refine_mask(mask: np.ndarray) -> np.ndarray:
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    return opened


def extract_primary_contour(mask: np.ndarray, min_area_ratio: float = 0.005) -> Optional[np.ndarray]:
    h, w = mask.shape[:2]
    min_area = min_area_ratio * h * w
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            return contour
    return None


def downsample_contour(contour: np.ndarray, max_points: int) -> np.ndarray:
    points = contour.reshape(-1, 2).astype(np.float32)
    if len(points) <= max_points:
        return points
    step = max(1, len(points) // max_points)
    return points[::step]


def farthest_point_indices(points: np.ndarray, max_candidates: int) -> np.ndarray:
    n = len(points)
    if n <= max_candidates:
        return np.arange(n, dtype=int)
    selected = [0]
    distances = np.linalg.norm(points - points[0], axis=1)
    for _ in range(1, max_candidates):
        idx = int(np.argmax(distances))
        if idx in selected:
            break
        selected.append(idx)
        new_dist = np.linalg.norm(points - points[idx], axis=1)
        distances = np.minimum(distances, new_dist)
    if len(selected) < 4:
        uniform = np.linspace(0, n - 1, 4, dtype=int).tolist()
        selected.extend(uniform)
    selected = sorted(set(selected))[:max_candidates]
    return np.array(selected, dtype=int)


def uniform_sample_indices(length: int, count: int) -> np.ndarray:
    count = min(length, max(4, count))
    if count <= 0:
        return np.arange(0)
    indices = np.linspace(0, length - 1, count, dtype=int)
    return np.unique(indices)


def signed_polygon_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    x, y = points[:, 0], points[:, 1]
    return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polygon_area(points: np.ndarray) -> float:
    return abs(signed_polygon_area(points))


def ensure_clockwise(points: np.ndarray) -> np.ndarray:
    return points[::-1] if signed_polygon_area(points) < 0 else points


def is_convex_polygon(points: np.ndarray) -> bool:
    if len(points) < 3:
        return False
    cross_sign, n = 0, len(points)
    for i in range(n):
        a = points[(i + 1) % n] - points[i]
        b = points[(i + 2) % n] - points[(i + 1) % n]
        cross = np.cross(a, b)
        if cross != 0:
            if cross_sign == 0:
                cross_sign = np.sign(cross)
            elif np.sign(cross) != cross_sign:
                return False
    return True


def clip_polygon_with_convex(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    def inside(p, a, b):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0

    def compute_intersection(p1, p2, a, b):
        s1, s2 = p2 - p1, b - a
        denom = s1[0] * s2[1] - s2[0] * s1[1]
        if abs(denom) < 1e-9: return p2
        t = ((a[0] - p1[0]) * s2[1] - (a[1] - p1[1]) * s2[0]) / denom
        return p1 + t * s1

    output = subject.copy()
    for i in range(len(clip)):
        input_list = output.copy()
        if len(input_list) == 0: break
        output, A, B = [], clip[i], clip[(i + 1) % len(clip)]
        for j in range(len(input_list)):
            P, Q = input_list[j], input_list[(j + 1) % len(input_list)]
            if inside(Q, A, B):
                if not inside(P, A, B): output.append(compute_intersection(P, Q, A, B))
                output.append(Q)
            elif inside(P, A, B): output.append(compute_intersection(P, Q, A, B))
        output = np.array(output, dtype=np.float32)
    return output


def quad_iou(candidate: np.ndarray, subject_polygon: np.ndarray, subject_area: float) -> Tuple[float, Optional[np.ndarray]]:
    if len(candidate) != 4: return -1.0, None
    candidate = ensure_clockwise(candidate)
    if np.any(np.linalg.norm(np.diff(np.vstack([candidate, candidate[0]]), axis=0), axis=1) < 1.0): return -1.0, None
    if polygon_area(candidate) < 1.0 or not is_convex_polygon(candidate): return -1.0, None
    clipped = clip_polygon_with_convex(subject_polygon, candidate)
    if len(clipped) < 3: return -1.0, None
    inter_area, quad_area = polygon_area(clipped), polygon_area(candidate)
    union = subject_area + quad_area - inter_area
    return (inter_area / union, candidate) if union > 0 else (-1.0, None)


def search_best_quad(points: np.ndarray, subject_polygon: np.ndarray, subject_area: float) -> Optional[np.ndarray]:
    n = len(points)
    if n < 4: return None
    best_score, best_quad = -1.0, None
    for i in range(n - 3):
        for j in range(i + 1, n - 2):
            for k in range(j + 1, n - 1):
                for l in range(k + 1, n):
                    candidate = np.array([points[i], points[j], points[k], points[l]], dtype=np.float32)
                    score, quad = quad_iou(candidate, subject_polygon, subject_area)
                    if score > best_score and quad is not None:
                        best_score, best_quad = score, quad
                    if best_score >= 0.99: return best_quad
    return best_quad


def ordered_quadrilateral_from_contour(contour: np.ndarray) -> Optional[np.ndarray]:
    subject_area = cv2.contourArea(contour)
    if subject_area < 10: return None
    points_full = downsample_contour(contour, MAX_CONTOUR_POINTS)
    if len(points_full) < 4: return None
    subject_polygon = ensure_clockwise(points_full)
    candidate_sets = [
        points_full[farthest_point_indices(points_full, MAX_CANDIDATE_POINTS)],
        points_full[uniform_sample_indices(len(points_full), UNIFORM_SAMPLE_POINTS)]
    ]
    if len(points_full) <= UNIFORM_SAMPLE_POINTS: candidate_sets.append(points_full)
    for candidates in candidate_sets:
        if (quad := search_best_quad(candidates, subject_polygon, subject_area)) is not None:
            return quad
    return None


def segment_fitted_sheet(image_rgb: np.ndarray, model: YOLO, allowed_classes: Sequence[int]) -> Optional[np.ndarray]:
    results = model(image_rgb, task="segment", verbose=False)
    if not results or results[0].masks is None or results[0].boxes is None: return None
    mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    masks, classes = results[0].masks.data.cpu().numpy(), results[0].boxes.cls.cpu().numpy()
    for mask_data, cls in zip(masks, classes):
        if int(cls) not in allowed_classes: continue
        resized = cv2.resize(mask_data, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = cv2.bitwise_or(mask, (resized > 0.5).astype(np.uint8) * 255)
    return mask if np.any(mask) else None


class QuadDKeypointDetectorService:
    def __init__(self, model_path: str = "weights/yolo_finetuned/best.pt"):
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: str):
        if not os.path.exists(model_path):
            logger.error(f"YOLO model for Quadrilateral detection not found at {model_path}")
            raise FileNotFoundError(f"YOLO model not found at {model_path}")
        try:
            model = YOLO(model_path)
            # Dummy inference to warm up
            _ = model(np.zeros((100, 100, 3), dtype=np.uint8), verbose=False)
            logger.info(f"Quadrilateral detection service loaded YOLO model from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load YOLO model from {model_path}: {e}", exc_info=True)
            raise

    def detect_keypoints(self, 
                         color_image: np.ndarray, 
                         depth_image: np.ndarray, 
                         rs_service: RealSenseCaptureService = None
                         ) -> Tuple[np.ndarray, List[dict]]:
        """
        Detects four corner keypoints of a fitted sheet by fitting a quadrilateral to its segmentation mask.
        """
        if rs_service is None:
            logger.warning("RealSense service is not found; 3D points will not be calculated.")
        
        # The script expects BGR, but the endpoint provides RGB. We use the provided RGB.
        mask = segment_fitted_sheet(color_image, self.model, ALLOWED_CLASSES)
        if mask is None:
            logger.warning("Quadrilateral method: Segmentation failed.")
            return color_image, []

        mask_refined = refine_mask(mask)
        contour = extract_primary_contour(mask_refined)
        if contour is None:
            logger.warning("Quadrilateral method: No primary contour found.")
            return color_image, []

        quad = ordered_quadrilateral_from_contour(contour)
        if quad is None or quad.shape != (4, 2):
            logger.warning("Quadrilateral method: Could not infer quadrilateral.")
            return color_image, []

        # The script `order_points_clockwise` is for axis-aligned rectangles.
        # The quad from `search_best_quad` is already ordered along the contour.
        # We just need to ensure it starts top-left for consistency.
        centroid = quad.mean(axis=0)
        angles = np.arctan2(quad[:, 1] - centroid[1], quad[:, 0] - centroid[0])
        sorted_indices = np.argsort(angles)
        quad_sorted = quad[sorted_indices]

        # Find the point closest to the top-left corner of the image to be the first point
        distances_to_origin = np.linalg.norm(quad_sorted, axis=1)
        start_index = np.argmin(distances_to_origin)
        quad_final = np.roll(quad_sorted, -start_index, axis=0)

        final_keypoints = []
        processed_image = color_image.copy()

        # Draw the polygon and corners on the image
        poly = np.round(quad_final).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(processed_image, [poly], True, (0, 255, 0), 2)

        for i, point in enumerate(quad_final):
            x, y = int(round(point[0])), int(round(point[1]))
            depth_m = float(depth_image[y, x]) * settings.DEPTH_SCALE if 0 <= y < depth_image.shape[0] and 0 <= x < depth_image.shape[1] else 0.0
            point_3d = rs_service.deproject_pixel_to_point(x, y, depth_m) if rs_service else []
            final_keypoints.append({
                "rgbd": {"x": x, 
                         "y": y, 
                         "depth_m": round(depth_m, 5)},
                "point3d_m": format_3d_coordinates(point_3d)
            })

            cv2.circle(processed_image, (x, y), 6, (0, 255, 0), -1)
            # mark the index of the corner
            #cv2.putText(processed_image, str(i), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        return processed_image, final_keypoints


# Instantiate the service for the API
try:
    quadD_detector_service = QuadDKeypointDetectorService()
except Exception as e:
    logger.error(f"Failed to initialize QuadKeypointDetectorService: {e}", exc_info=True)
    quadD_detector_service = None
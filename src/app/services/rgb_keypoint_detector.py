import os
import cv2
import numpy as np
import torch
import logging
from ultralytics import YOLO

from app.models_rgb.hybrid_keypoint_net import HybridKeypointNet
from app.services.realsense_capture import RealSenseCaptureService
from app.models_rgb.model_utils import EnhancedYoloBackbone, thresholded_locations
from app.core.config import settings
from app.common.utils import format_3d_coordinates


def _get_base_module(module):
    return getattr(module, "_orig_mod", module)

def load_model_safely(model, load_path: str, map_location="cpu", strict: bool = False):
    state = torch.load(load_path, map_location=map_location)
    cleaned = {}
    for key, value in state.items():
        if key.startswith("_orig_mod."):
            cleaned[key[len("_orig_mod."):]] = value
        else:
            cleaned[key] = value
    target = _get_base_module(model)
    return target.load_state_dict(cleaned, strict=strict)

def combine_nearby_peaks(peaks, distance_threshold=10):
    if not peaks:
        return []
    
    peaks = np.array(peaks)
    if len(peaks) == 1:
        return peaks.tolist()
    
    from scipy.spatial.distance import pdist, squareform
    distances = squareform(pdist(peaks))
    
    clusters = []
    used = set()
    
    for i in range(len(peaks)):
        if i in used:
            continue
        cluster = [i]
        used.add(i)
        for j in range(i + 1, len(peaks)):
            if j not in used and distances[i, j] <= distance_threshold:
                cluster.append(j)
                used.add(j)
        clusters.append(cluster)
    
    combined_peaks = []
    for cluster in clusters:
        cluster_peaks = peaks[cluster]
        centroid = cluster_peaks.mean(axis=0)
        combined_peaks.append(centroid)
    
    return combined_peaks

def _build_allowed_mask(result, allowed_classes=None):
    """Build a combined boolean mask for allowed classes from a YOLO result in the result image space."""
    masks = getattr(result, 'masks', None)
    boxes = getattr(result, 'boxes', None)
    if masks is None or masks.data is None:
        return None
    cls_ids = None
    if boxes is not None and getattr(boxes, 'cls', None) is not None:
        cls_ids = boxes.cls.cpu().numpy().astype(int)
    mask_accum = None
    for idx, m in enumerate(masks.data):
        if allowed_classes is not None and cls_ids is not None:
            if cls_ids[idx] not in set(allowed_classes):
                continue
        m_np = (m.cpu().numpy() > 0.5).astype(np.uint8)
        if mask_accum is None:
            mask_accum = m_np
        else:
            mask_accum = np.maximum(mask_accum, m_np)
    return mask_accum

class RGBKeypointDetectorService:
    def __init__(self, 
                 model_path="weights_rgb/keypoint_model_vit_post.pth",
                 segmenter_path="weights_rgb/yolo_finetuned/best.pt"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        logging.info(f"RGB Keypoint detection model loaded on {self.device}.")
        self.segmenter = self._load_segmenter(segmenter_path)

    def _load_segmenter(self, segmenter_path: str):
        if not os.path.exists(segmenter_path):
            logging.warning(f"Segmentation model not found at {segmenter_path}. Proceeding without segmentation.")
            return None
        try:
            segmenter = YOLO(segmenter_path)
            # Perform a dummy inference to ensure it's loaded correctly
            _ = segmenter(np.zeros((100, 100, 3), dtype=np.uint8), verbose=False)
            logging.info(f"Segmentation model loaded from {segmenter_path}.")
            return segmenter
        except Exception as e:
            logging.error(f"Failed to load segmentation model from {segmenter_path}: {e}", exc_info=True)
            return None

    def _load_model(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"RGB keypoint model not found at {model_path}")

        yolo_model = YOLO('yolo11l-pose.pt')
        backbone = EnhancedYoloBackbone(
            yolo_model, 
            include_neck=True,
            selected_indices=[2, 4, 6, 8, 10, 13, 16, 19, 22]
        )
        
        input_dummy = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            feats = backbone(input_dummy)
        in_channels_list = [f.shape[1] for f in feats]
        
        model = HybridKeypointNet(backbone, in_channels_list)
        model = model.to(self.device)
        
        load_model_safely(model, model_path, map_location=self.device, strict=False)
        model.eval()
        return model

    def detect_keypoints(self,
                         color_image: np.ndarray,
                         depth_image: np.ndarray,
                         rs_service: RealSenseCaptureService
                         ):
        orig_h, orig_w = color_image.shape[:2]

        # --- Segmentation (Optional) ---
        masked_image = color_image.copy()
        if self.segmenter:
            try:
                # Resize for robust segmentation, maintaining aspect ratio
                seg_max_side = 1280
                scale = min(seg_max_side / max(orig_h, orig_w), 1.0)
                if scale < 1.0:
                    seg_w, seg_h = int(orig_w * scale), int(orig_h * scale)
                    seg_input_img = cv2.resize(color_image, (seg_w, seg_h), interpolation=cv2.INTER_AREA)
                else:
                    seg_input_img = color_image

                results = self.segmenter(seg_input_img, verbose=False)
                if results and len(results) > 0:
                    # Class 2 is assumed to be the bedsheet/relevant class
                    mask_small = _build_allowed_mask(results[0], allowed_classes=[2])
                    if mask_small is not None:
                        # Resize mask back to original image size
                        mask_full = cv2.resize(mask_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                        # Apply mask: zero out non-bedsheet pixels
                        masked_image[mask_full == 0] = 0
                        logging.info("Applied segmentation mask to image.")
            except Exception as e:
                logging.error(f"Segmentation failed during inference: {e}", exc_info=True)

        # Preprocess
        img_rgb = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (128, 128))
        # The demo script normalizes to [0, 1], which is different from the original implementation.
        # Let's follow the demo script's approach as it seems to be the latest.
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_tensor = np.transpose(img_normalized, (2, 0, 1))
        img_tensor = np.expand_dims(img_tensor, axis=0)

        # Inference
        with torch.no_grad():
            input_tensor = torch.from_numpy(img_tensor).to(self.device)
            outputs = self.model(input_tensor)
            heatmap = outputs.cpu().numpy()[0, 0, :, :]

        # Post-process
        edge = 3
        heatmap[:edge, :] = 0
        heatmap[-edge:, :] = 0
        heatmap[:, :edge] = 0
        heatmap[:, -edge:] = 0
        
        peaks = thresholded_locations(heatmap, 0.003)
        combined_peaks = combine_nearby_peaks(peaks, distance_threshold=10)

        # Scale keypoints and prepare output
        scale_x = orig_w / 128.0
        scale_y = orig_h / 128.0
        
        final_keypoints = []
        processed_image = color_image.copy()

        if combined_peaks:
            for p in combined_peaks:
                row, col = p
                orig_y = int(row * scale_y)
                orig_x = int(col * scale_x)
                
                distance_meters = float(depth_image[orig_y, orig_x]) * settings.DEPTH_SCALE
                point_3d = rs_service.deproject_pixel_to_point(orig_x, orig_y, distance_meters) if rs_service else []
                final_keypoints.append({
                    "color": {"x": orig_x, 
                              "y": orig_y,
                              "depth_m": round(distance_meters, 5)},
                    "point3d_m": format_3d_coordinates(point_3d)
                })
                cv2.circle(processed_image, (orig_x, orig_y), 5, (0, 0, 255), -1)
        else:
            # Fallback to argmax if no peaks found
            max_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            row, col = int(max_idx[0]), int(max_idx[1])
            orig_y = int(row * scale_y)
            orig_x = int(col * scale_x)
            # Not adding to keypoints list as it's a fallback, but marking on image
            cv2.circle(processed_image, (orig_x, orig_y), 24, (0, 255, 0), 2)

        return processed_image, final_keypoints

# Instantiate the service to be imported by the API
try:
    rgb_detector_service = RGBKeypointDetectorService()
except Exception as e:
    logging.error(f"Failed to initialize RGBKeypointDetectorService: {e}", exc_info=True)
    rgb_detector_service = None
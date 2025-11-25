import os
import cv2
import numpy as np
import torch
import logging
from typing import List, Tuple

from app.models.clip_heatmap_model import ClipHeatmapModel
from app.services.realsense_capture import RealSenseCaptureService
from app.models.utils import thresholded_locations, combine_nearby_peaks
from app.core.config import settings
from app.common.utils import get_image_hash, format_3d_coordinates

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("ultralytics is not installed. Segmentation will be skipped.")

logger = logging.getLogger(__name__)

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
                'path': 'weights/meta_clip_style_bedsheet_post_original',
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
                'seg_classes': [1,3]
            },
            'fitted_sheet_inverse': {
                'path': 'weights/meta_clip_style_fitted_sheet_inverse_post_original',
                'lora_r': 16, 'lora_alpha': 32, 'image_size': 256, 'use_text_prior': True,
                'seg_classes': [1,3]
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
        yolo_path = 'weights/yolo_finetuned/best.pt'
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

        target_size = self.model_config['image_size']
        image_resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        image_resized = image_resized.astype(np.float32)
        if self.yolo_model is None:
            return image_resized

        try:
            logger.info(f"RGB color image hash: {get_image_hash(image)}")
            results = self.yolo_model(image, task="segment", verbose=False)
            if not (results and results[0].masks):
                return image_resized

            mask_all = np.zeros(image.shape[:2], dtype=np.uint8)
            masks = results[0].masks.data.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()

            for mask, cls_id in zip(masks, classes):
                if int(cls_id) in self.model_config['seg_classes']:
                    mask_binary = (mask > 0.5).astype(np.uint8) * 255
                    mask_resized = cv2.resize(mask_binary, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                    mask_all = cv2.bitwise_or(mask_all, mask_resized)

            if np.any(mask_all > 0):
                logger.info(f"✅ Applied YOLO segmentation for classes {self.model_config['seg_classes']}")
                mask_all_resized = cv2.resize(mask_all, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
                image_resized[mask_all_resized == 0] = 0

        except Exception as e:
            logging.error(f"⚠️ YOLO processing failed: {e}")

        return image_resized

    def detect_keypoints(self,
                         color_image: np.ndarray,
                         depth_image: np.ndarray,
                         rs_service: RealSenseCaptureService = None
                         ) -> Tuple[np.ndarray, List[dict]]:
        """Detects keypoints in the given color image."""
        orig_h, orig_w = color_image.shape[:2]

        target_size = self.model_config['image_size']
        image_resized_masked = self._apply_segmentation(color_image)
        logger.info(f"hash of image_resized_masked before tensor conversion: {get_image_hash(image_resized_masked)}")
        image_tensor = torch.from_numpy(image_resized_masked).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(self.device)
        logger.info(f"Input shape: {image_tensor.shape}, Original size: {(orig_w, orig_h)}")

        with torch.no_grad():
            heatmap = self.model(image_tensor).squeeze().cpu().numpy()
        logger.info(f"Heatmap shape: {heatmap.shape}")
        logger.info(f"hash of heatmap: {get_image_hash(heatmap)}")

        peaks = thresholded_locations(heatmap, threshold=0.3)
        
        combined_peaks = combine_nearby_peaks(peaks, distance_threshold=10)

        scale_x, scale_y = orig_w / target_size, orig_h / target_size
        final_keypoints = []
        processed_image = color_image.copy()

        for p in combined_peaks:
            y, x = p
            orig_x, orig_y = int(x * scale_x), int(y * scale_y)
            distance_m = float(depth_image[orig_y, orig_x]) * settings.DEPTH_SCALE
            point_3d = rs_service.deproject_pixel_to_point(orig_x, orig_y, distance_m) if rs_service else []
            final_keypoints.append({
                "rgbd": {"x": orig_x, 
                         "y": orig_y,
                         "depth_m": round(distance_m, 5)},
                "point3d_m": format_3d_coordinates(point_3d)
            })
            cv2.circle(processed_image, (orig_x, orig_y), 5, (0, 255, 0), -1)

        return processed_image, final_keypoints


if __name__ == '__main__':
    import argparse
    import glob
    import sys

    # # Add the project's 'src' directory to the Python path to resolve module imports
    # # when running the script directly.
    # project_src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    # if project_src_path not in sys.path:
    #     sys.path.insert(0, project_src_path)


    parser = argparse.ArgumentParser(description="Detect keypoints in images using MetaClip model.")
    parser.add_argument("--image_folder", type=str, default="images",
                         help="Path to the folder containing images.")
    parser.add_argument("--model_type", type=str, default="mattress",
                        choices=['bedsheet', 'mattress', 'fitted_sheet', 'fitted_sheet_inverse'],
                        help="Type of model to use for detection.")
    parser.add_argument("--output_folder", type=str, default="result_metaclip", 
                        help="Optional path to save output images. If not provided, images will be displayed.")
    args = parser.parse_args()

    if args.output_folder:
        os.makedirs(args.output_folder, exist_ok=True)

    try:
        detector = MetaClipKeypointDetectorService(model_type=args.model_type)
    except FileNotFoundError as e:
        logger.error(f"Failed to initialize detector: {e}")
        exit(1)

    image_paths = sorted(glob.glob(os.path.join(args.image_folder, '*.[jJ][pP][gG]')) +
                       glob.glob(os.path.join(args.image_folder, '*.[pP][nN][gG]')))

    for image_path in image_paths:
        if not os.path.isfile(image_path):
            continue

        if not image_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        print(f"Processing {image_path}...")
        color_image_bgr = cv2.imread(image_path)
        if color_image_bgr is None:
            logger.warning(f"Could not read image {image_path}, skipping.")
            continue

        color_image_rgb = cv2.cvtColor(color_image_bgr, cv2.COLOR_BGR2RGB)
        h, w = color_image_rgb.shape[:2]
        dummy_depth = np.zeros((h, w), dtype=np.uint16)

        processed_image_rgb, keypoints = detector.detect_keypoints(color_image_rgb, dummy_depth, rs_service=None)
        logger.info(f"Found {len(keypoints)} keypoints.")

        processed_image_bgr = cv2.cvtColor(processed_image_rgb, cv2.COLOR_RGB2BGR)

        if args.output_folder:
            output_filename = f"{args.model_type}_{os.path.basename(image_path)}"
            output_path = os.path.join(args.output_folder, output_filename)
            cv2.imwrite(output_path, processed_image_bgr)
            logger.info(f"Saved result to {output_path}")
        else:
            cv2.imshow(f'Keypoints for {os.path.basename(image_path)}', processed_image_bgr)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

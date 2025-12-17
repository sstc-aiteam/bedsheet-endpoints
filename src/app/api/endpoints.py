import numpy as np
import cv2
import io
import base64
import logging
from PIL import Image

from typing import Optional

from fastapi.responses import JSONResponse
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Query
from fastapi.responses import Response
from fastapi import Body

from app.services.depth_keypoint_detector import depth_detector_service
from app.services.metaclip_keypoint_detector import MetaClipKeypointDetectorService
from app.services.realsense_capture import RealSenseCaptureService, NoDeviceError, FrameCaptureError, RealSenseError
from app.services.fitted_sheet_classifier import fitted_sheet_classifier_service
from app.services.rgb_keypoint_detector import rgb_detector_service
from app.services.quad_d_keypoint_detector import quadD_detector_service
from app.services.quad_yc_keypoint_detector import quadYC_detector_service
from app.services.quad_yc_sb_keypoint_detector import quadYC_sb_detector_service
from app.api.param_schema import DetectionMethod, ModelType, ProcessedImagePayload, DetectionParams
from app.api.param_schema import FittedSheetClassificationResponse, BoundingBox
from app.common.utils import get_image_hash, save_captured_images, decode_image_bytes_to_rgb


logger = logging.getLogger(__name__)

router = APIRouter()


async def _process_and_detect(
    method: DetectionMethod,
    model_type: Optional[ModelType],
    color_file: UploadFile,
    depth_file: UploadFile,
    params: DetectionParams
):
    """Helper function to process images and run detection."""
    color_contents = await color_file.read()
    color_image = decode_image_bytes_to_rgb(color_contents)

    if color_image is None:
        raise HTTPException(status_code=400, detail="Could not decode color image.")

    depth_contents = await depth_file.read()
    try:
        depth_image = np.load(io.BytesIO(depth_contents))
    except (ValueError, OSError) as e:
        raise HTTPException(status_code=400, detail=f"Could not load depth .npy file: {e}")

    if color_image.shape[:2] != depth_image.shape:
        raise HTTPException(
            status_code=400,
            detail=f"Image dimensions must match. Color: {color_image.shape[:2]}, Depth: {depth_image.shape}"
        )

    detector_map = {
        DetectionMethod.METACLIP: MetaClipKeypointDetectorService(model_type=model_type),
        DetectionMethod.RGB: rgb_detector_service,
        DetectionMethod.DEPTH: depth_detector_service,
        DetectionMethod.QUAD_D: quadD_detector_service,
        DetectionMethod.QUAD_YC: quadYC_detector_service,
        DetectionMethod.QUAD_YC_SB: quadYC_sb_detector_service,
    }
    
    detector = detector_map.get(method)
    if not detector:
        raise HTTPException(status_code=400, detail=f"Invalid detection method: {method}")

    processed_image, keypoints = detector.detect_keypoints(color_image, depth_image, rs_service=None, params=params) # rs_service is not available for file uploads
    # Convert RGB back to BGR for CV2 encoding to image file
    processed_image_bgr = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)

    return processed_image_bgr, keypoints

@router.post("/detect_keypoints/")
async def detect_keypoints(
    method: DetectionMethod = Query(
        default=DetectionMethod.METACLIP,
        description="The keypoint detection method to use."
    ),
    model_type: Optional[ModelType] = Query(
        default=ModelType.MATTRESS,
        description="The model type for MetaCLIP method: 'bedsheet', 'mattress', 'fitted_sheet', or 'fitted_sheet_inverse'. Only used when method is 'metaclip'."
    ),
    color_file: UploadFile = File(...), 
    depth_file: UploadFile = File(...),
    params: DetectionParams = Depends()
):
    """
    Accepts color and depth images and returns detected keypoints.

    - **color_file**: An image file (e.g., .png, .jpg).
    - **depth_file**: A .npy file containing the raw depth map (float).
    - **ntile_divisor**: (Optional) Divisor for Ntile calculations in 'quad_yc_sb' method, e.g., 4 for quartiles, 6 or 10 for finer divisions.
    """
    try:
        processed_image_bgr, keypoints = await _process_and_detect(method, model_type, color_file, depth_file, params)
        _, encoded_img = cv2.imencode('.PNG', processed_image_bgr)
        processed_img_base64 = base64.b64encode(encoded_img.tobytes()).decode('utf-8')

        return JSONResponse(content={"keypoints": keypoints, "processed_image": processed_img_base64})

    except Exception as e:
        logging.error(f"Error processing request in /detect_keypoints: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")


@router.post("/detect_keypoints_visualization/", response_class=Response)
async def detect_keypoints_visualization(
    method: DetectionMethod = Query(
        default=DetectionMethod.METACLIP,
        description="The keypoint detection method to use."
    ),
    model_type: Optional[ModelType] = Query(
        default=ModelType.MATTRESS,
        description="The model type for MetaCLIP method: 'bedsheet', 'mattress', 'fitted_sheet', or 'fitted_sheet_inverse'. Only used when method is 'metaclip'."
    ),
    color_file: UploadFile = File(...),
    depth_file: UploadFile = File(...),
    params: DetectionParams = Depends()
):
    """
    Accepts color and depth images, detects keypoints, and returns the processed image.

    - **color_file**: An image file (e.g., .png, .jpg).
    - **depth_file**: A .npy file containing the raw depth map (float).
    - **ntile_divisor**: (Optional) Divisor for Ntile calculations in 'quad_yc_sb' method, e.g., 4 for quartiles, 6 or 10 for finer divisions.
    """
    try:
        processed_image_bgr, _ = await _process_and_detect(method, model_type, color_file, depth_file, params)
        success, encoded_img = cv2.imencode('.png', processed_image_bgr)
        if not success:
            raise HTTPException(status_code=500, detail="Could not encode processed image.")

        return Response(content=encoded_img.tobytes(), media_type="image/png")

    except Exception as e:
        logging.error(f"Error processing request in /detect_keypoints_image: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

async def _capture_and_detect(
    method: DetectionMethod,
    model_type: Optional[ModelType],
    rs_capture_service: RealSenseCaptureService,
    params: DetectionParams
):
    """Helper function to capture images and run detection."""
    # Use the service to capture images
    color_bgr_image, depth_image = rs_capture_service.capture_images()
    # Save raw captured images using the utility function
    save_captured_images(color_bgr_image, depth_image)

    color_image = cv2.cvtColor(color_bgr_image, cv2.COLOR_BGR2RGB)

    detector_map = {
        DetectionMethod.METACLIP: MetaClipKeypointDetectorService(model_type=model_type),
        DetectionMethod.RGB: rgb_detector_service,
        DetectionMethod.DEPTH: depth_detector_service,
        DetectionMethod.QUAD_D: quadD_detector_service,
        DetectionMethod.QUAD_YC: quadYC_detector_service,
        DetectionMethod.QUAD_YC_SB: quadYC_sb_detector_service,
    }

    detector = detector_map.get(method)
    if not detector:
        # This case should ideally not be hit if enums are used correctly
        # but is good for robustness.
        raise HTTPException(status_code=400, detail=f"Invalid detection method: {method}")

    if not hasattr(detector, 'detect_keypoints'):
        raise HTTPException(status_code=500, detail=f"Detector for method '{method}' is not properly configured.")

    processed_image, keypoints = detector.detect_keypoints(color_image, depth_image, rs_capture_service, params=params)
    # Convert RGB back to BGR for CV2 encoding to image file
    processed_image_bgr = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)

    return processed_image_bgr, keypoints


@router.post("/capture_and_detect_keypoints/")
async def capture_and_detect_keypoints(
    method: DetectionMethod = Query(
        default=DetectionMethod.METACLIP,
        description="The keypoint detection method to use."
    ),
    model_type: Optional[ModelType] = Query(
        default=ModelType.MATTRESS,
        description="The model type for MetaCLIP method: 'bedsheet', 'mattress', 'fitted_sheet', or 'fitted_sheet_inverse'. Only used when method is 'metaclip'."
    ),
    rs_capture_service: RealSenseCaptureService = Depends(RealSenseCaptureService),
    params: DetectionParams = Depends()
):
    """
    Captures color and depth images from a connected Intel RealSense camera
    and returns detected keypoints using the specified method.
    The camera must be connected to the server.

    - **ntile_divisor**: (Optional) Divisor for Ntile calculations in 'quad_yc_sb' method, e.g., 4 for quartiles, 6 or 10 for finer divisions.
    """
    try:
        processed_image, keypoints = await _capture_and_detect(method, model_type, rs_capture_service, params)
        _, encoded_img = cv2.imencode('.PNG', processed_image)
        processed_img_base64 = base64.b64encode(encoded_img.tobytes()).decode('utf-8')

        return JSONResponse(content={"keypoints": keypoints, "processed_image": processed_img_base64})

    except NoDeviceError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except (FrameCaptureError, RealSenseError) as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logging.error(f"Error processing request in /capture_and_detect_keypoints: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")


@router.post("/capture_and_detect_keypoints_visualization/", response_class=Response)
async def capture_and_detect_keypoints_visualization(
    method: DetectionMethod = Query(
        default=DetectionMethod.METACLIP,
        description="The keypoint detection method to use."
    ),
    model_type: Optional[ModelType] = Query(
        default=ModelType.MATTRESS,
        description="The model type for MetaCLIP method: 'bedsheet', 'mattress', 'fitted_sheet', or 'fitted_sheet_inverse'. Only used when method is 'metaclip'."
    ),
    rs_capture_service: RealSenseCaptureService = Depends(RealSenseCaptureService),
    params: DetectionParams = Depends()
):
    """
    Captures images from a RealSense camera and returns the processed image with keypoints.
    The camera must be connected to the server.

    - **ntile_divisor**: (Optional) Divisor for Ntile calculations in 'quad_yc_sb' method, e.g., 4 for quartiles, 6 or 10 for finer divisions.
    """
    try:
        processed_image, _ = await _capture_and_detect(method, model_type, rs_capture_service, params)
        success, encoded_img = cv2.imencode('.png', processed_image)
        if not success:
            raise HTTPException(status_code=500, detail="Could not encode processed image.")

        return Response(content=encoded_img.tobytes(), media_type="image/png")

    except NoDeviceError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except (FrameCaptureError, RealSenseError) as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logging.error(f"Error processing request in /capture_and_detect_keypoints_visualization: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")


@router.post("/show_image_from_base64/", response_class=Response)
async def show_image_from_base64(
    item: ProcessedImagePayload = Body(
        ..., description="A JSON object containing a base64 encoded string of an image."
    )
):
    """
    Accepts a base64 encoded image string, decodes it, and returns it as an image, e.g., 
    {"keypoints":[{"x":0,"y":0,"depth_m":0}],"processed_image":"string"}

    - **processed_image**: A base64 encoded string of an image.
    """
    try:
        # Decode the base64 string
        decoded_image = base64.b64decode(item.processed_image)
        
        # The images are consistently encoded as PNGs by other endpoints.
        media_type = "image/png"

        # Return the image as a response.
        # The browser will render this as an image.
        return Response(content=decoded_image, media_type=media_type)

    except (base64.binascii.Error, ValueError) as e:
        logging.error(f"Invalid base64 string: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Invalid base64 string: {e}")
    except Exception as e:
        logging.error(f"Error processing request in /show_image_from_base64: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")


@router.post("/detect_fitted_sheet/", response_model=FittedSheetClassificationResponse)
async def detect_fitted_sheet(file: UploadFile = File(...)):
    """
    Classify a fitted sheet image.
    """
    contents = await file.read()
    img_rgb = decode_image_bytes_to_rgb(contents)

    if img_rgb is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    rs = fitted_sheet_classifier_service.classify(img_rgb)
    
    if isinstance(rs, dict) and "error" in rs:
        raise HTTPException(status_code=500, detail=rs["error"])
    
    bbox = BoundingBox(
        x_min=rs["bbox"][0], y_min=rs["bbox"][1],
        x_max=rs["bbox"][2], y_max=rs["bbox"][3]) if rs["bbox"] else None

    return FittedSheetClassificationResponse(
            label=rs["label"],
            pred=rs["pred"],
            conf=rs["conf"],
            bbox=bbox)


@router.post("/capture_and_detect_fitted_sheet/", response_model=FittedSheetClassificationResponse)
async def capture_and_detect_fitted_sheet(
    rs_capture_service: RealSenseCaptureService = Depends(RealSenseCaptureService)
):
    """
    Captures an image from the RealSense camera and classifies the fitted sheet.
    """
    try:
        color_bgr, _ = rs_capture_service.capture_images()

        img_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

        rs = fitted_sheet_classifier_service.classify(img_rgb)

        if isinstance(rs, dict) and "error" in rs:
            raise HTTPException(status_code=500, detail=rs["error"])

        bbox = BoundingBox(
            x_min=rs["bbox"][0], y_min=rs["bbox"][1],
            x_max=rs["bbox"][2], y_max=rs["bbox"][3]) if rs["bbox"] else None

        return FittedSheetClassificationResponse(
            label=rs["label"],
            pred=rs["pred"],
            conf=rs["conf"],
            bbox=bbox)

    except NoDeviceError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except (FrameCaptureError, RealSenseError) as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logging.error(f"Error processing request in /capture_and_detect_fitted_sheet: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")



    
    

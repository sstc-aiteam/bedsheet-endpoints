import numpy as np
import cv2
import io
import base64
import logging

from typing import Optional

from fastapi.responses import JSONResponse
import magic
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import Response
from fastapi import Body

from app.services.depth_keypoint_detector import depth_detector_service
from app.services.metaclip_keypoint_detector import MetaClipKeypointDetectorService
from app.services.realsense_capture import capture_images, NoDeviceError, FrameCaptureError, RealSenseError
from app.services.rgb_keypoint_detector import rgb_detector_service
from app.api.param_type import DetectionMethod, ModelType
from app.api.base64image import Base64Image


router = APIRouter()

async def _process_and_detect(
    method: DetectionMethod,
    model_type: Optional[ModelType],
    color_file: UploadFile,
    depth_file: UploadFile
):
    """Helper function to process images and run detection."""
    color_contents = await color_file.read()
    color_nparr = np.frombuffer(color_contents, np.uint8)
    color_image = cv2.imdecode(color_nparr, cv2.IMREAD_COLOR)
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
        DetectionMethod.RGB: rgb_detector_service,
        DetectionMethod.DEPTH: depth_detector_service,
        DetectionMethod.METACLIP: MetaClipKeypointDetectorService(model_type=model_type)
    }
    
    detector = detector_map.get(method)
    if not detector:
        raise HTTPException(status_code=400, detail=f"Invalid detection method: {method}")

    return detector.detect_keypoints(color_image, depth_image)

@router.post("/detect_keypoints/")
async def detect_keypoints(
    method: DetectionMethod = Query(
        default=DetectionMethod.METACLIP,
        description="The keypoint detection method to use: 'metaclip' (default), 'depth', or 'rgb'."
    ),
    model_type: Optional[ModelType] = Query(
        default=ModelType.MATTRESS,
        description="The model type for MetaCLIP method: 'bedsheet', 'mattress', or 'fitted_sheet'. Only used when method is 'metaclip'."
    ),
    color_file: UploadFile = File(...), 
    depth_file: UploadFile = File(...)
):
    """
    Accepts color and depth images and returns detected keypoints.

    - **color_file**: An image file (e.g., .png, .jpg).
    - **depth_file**: A .npy file containing the raw depth map (float).
    """
    try:
        processed_image, keypoints = await _process_and_detect(method, model_type, color_file, depth_file)
        _, encoded_img = cv2.imencode('.PNG', processed_image)
        processed_img_base64 = base64.b64encode(encoded_img.tobytes()).decode('utf-8')

        return JSONResponse(content={"keypoints": keypoints, "processed_image": processed_img_base64})

    except Exception as e:
        logging.error(f"Error processing request in /detect_keypoints: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")


@router.post("/detect_keypoints_image/", response_class=Response)
async def detect_keypoints_image(
    method: DetectionMethod = Query(
        default=DetectionMethod.METACLIP,
        description="The keypoint detection method to use: 'metaclip' (default), 'depth', or 'rgb'."
    ),
    model_type: Optional[ModelType] = Query(
        default=ModelType.MATTRESS,
        description="The model type for MetaCLIP method: 'bedsheet', 'mattress', or 'fitted_sheet'. Only used when method is 'metaclip'."
    ),
    color_file: UploadFile = File(...),
    depth_file: UploadFile = File(...)
):
    """
    Accepts color and depth images, detects keypoints, and returns the processed image.

    - **color_file**: An image file (e.g., .png, .jpg).
    - **depth_file**: A .npy file containing the raw depth map (float).
    """
    try:
        processed_image, _ = await _process_and_detect(method, model_type, color_file, depth_file)
        success, encoded_img = cv2.imencode('.png', processed_image)
        if not success:
            raise HTTPException(status_code=500, detail="Could not encode processed image.")

        return Response(content=encoded_img.tobytes(), media_type="image/png")

    except Exception as e:
        logging.error(f"Error processing request in /detect_keypoints_image: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")


@router.post("/capture_and_detect_keypoints/")
async def capture_and_detect_keypoints(
    method: DetectionMethod = Query(
        default=DetectionMethod.RGB,
        description="The keypoint detection method to use: 'metaclip' (default), 'depth', or 'rgb'."
    ),
    model_type: Optional[ModelType] = Query(
        default=ModelType.MATTRESS,
        description="The model type for MetaCLIP method: 'bedsheet', 'mattress', or 'fitted_sheet'. Only used when method is 'metaclip'."
    )
):
    """
    Captures color and depth images from a connected Intel RealSense camera
    and returns detected keypoints using the specified method.
    The camera must be connected to the server.
    """
    try:
        color_image, depth_image = capture_images()

        if method == DetectionMethod.RGB:
            processed_image, keypoints = rgb_detector_service.detect_keypoints(color_image, depth_image)
        elif method == DetectionMethod.DEPTH:
            processed_image, keypoints = depth_detector_service.detect_keypoints(color_image, depth_image)
        else: # method == DetectionMethod.METACLIP
            metaclip_service = MetaClipKeypointDetectorService(model_type=model_type)
            processed_image, keypoints = metaclip_service.detect_keypoints(color_image, depth_image)

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


@router.post("/show_image_from_base64/", response_class=Response)
async def show_image_from_base64(
    item: Base64Image = Body(
        ..., description="A JSON object containing a URL-safe base64 encoded string of an image."
    )
):
    """
    Accepts a base64 encoded image string, decodes it, and returns it as an image.

    - **image_data**: A URL-safe base64 encoded string of an image.
    """
    try:
        # Decode the base64 string
        decoded_image = base64.urlsafe_b64decode(item.image_data)

        # Detect the media type from the image content
        media_type = magic.from_buffer(decoded_image, mime=True)

        # Return the image as a response.
        # The browser will render this as an image.
        return Response(content=decoded_image, media_type=media_type)

    except (base64.binascii.Error, ValueError) as e:
        logging.error(f"Invalid base64 string: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Invalid base64 string: {e}")
    except Exception as e:
        logging.error(f"Error processing request in /show_image_from_base64: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")
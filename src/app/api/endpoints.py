import numpy as np
import cv2
import io
import base64
import logging

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from app.services.keypoint_detector import detector_service
from app.services.realsense_capture import capture_images, NoDeviceError, FrameCaptureError, RealSenseError


router = APIRouter()

@router.post("/detect_keypoints/")
async def detect_keypoints(color_file: UploadFile = File(...), depth_file: UploadFile = File(...)):
    """
    Accepts color and depth images and returns detected keypoints.

    - **color_file**: An image file (e.g., .png, .jpg).
    - **depth_file**: A .npy file containing the raw depth map (float).
    """
    try:
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

        processed_image, keypoints = detector_service.detect_keypoints(color_image, depth_image)

        _, encoded_img = cv2.imencode('.PNG', processed_image)
        processed_img_base64 = base64.b64encode(encoded_img.tobytes()).decode('utf-8')

        return JSONResponse(content={"keypoints": keypoints, "processed_image": processed_img_base64})

    except Exception as e:
        logging.error(f"Error processing request in /detect_keypoints: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")


@router.post("/capture_and_detect_keypoints/")
async def capture_and_detect_keypoints():
    """
    Captures color and depth images from a connected Intel RealSense camera
    and returns detected keypoints. The camera must be connected to the server.
    """
    try:
        color_image, depth_image = capture_images()

        processed_image, keypoints = detector_service.detect_keypoints(color_image, depth_image)

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
import numpy as np
import cv2
import io
import base64
import logging
import pyrealsense2 as rs

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from app.services.keypoint_detector import detector_service


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


def _capture_realsense_images():
    """
    Initializes a RealSense camera, captures, and aligns one pair of color and depth frames.

    Returns:
        A tuple containing (color_image, depth_image).

    Raises:
        HTTPException: If no camera is found or frames cannot be captured.
    """
    pipeline = rs.pipeline()
    config = rs.config()

    # Check for a connected RealSense device
    context = rs.context()
    if len(context.devices) == 0:
        raise HTTPException(status_code=503, detail="No RealSense device connected.")

    # Configure and start the pipeline
    # Using common settings from the reference script.
    width, height, fps = 848, 480, 30
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    try:
        profile = pipeline.start(config)

        # Create an align object to align depth frames to color frames
        align_to = rs.stream.color
        align = rs.align(align_to)

        # The first few frames can be dark/overexposed.
        # Allow auto-exposure to settle by capturing a few frames.
        for _ in range(5):
            pipeline.wait_for_frames()

        # Get a coherent pair of frames
        frames = pipeline.wait_for_frames(timeout_ms=5000)
        aligned_frames = align.process(frames)
        
        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            raise HTTPException(status_code=500, detail="Could not capture valid frames from RealSense camera.")

        # Convert frames to numpy arrays
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        return color_image, depth_image

    except RuntimeError as e:
        logging.error(f"RealSense runtime error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error with RealSense camera: {e}")
    finally:
        # Ensure the pipeline is stopped
        pipeline.stop()


@router.post("/capture_and_detect_keypoints/")
async def capture_and_detect_keypoints():
    """
    Captures color and depth images from a connected Intel RealSense camera
    and returns detected keypoints. The camera must be connected to the server.
    """
    try:
        color_image, depth_image = _capture_realsense_images()

        processed_image, keypoints = detector_service.detect_keypoints(color_image, depth_image)

        _, encoded_img = cv2.imencode('.PNG', processed_image)
        processed_img_base64 = base64.b64encode(encoded_img.tobytes()).decode('utf-8')

        return JSONResponse(content={"keypoints": keypoints, "processed_image": processed_img_base64})

    except HTTPException as e:
        # Re-raise HTTP exceptions from the capture function
        raise e
    except Exception as e:
        logging.error(f"Error processing request in /capture_and_detect: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")
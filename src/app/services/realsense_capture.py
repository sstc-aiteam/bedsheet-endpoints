import cv2
import numpy as np
import pyrealsense2 as rs
import logging

# --- Custom Exceptions ---

class RealSenseError(Exception):
    """Base exception for RealSense camera errors."""
    pass

class NoDeviceError(RealSenseError):
    """Exception raised when no RealSense device is found."""
    pass

class FrameCaptureError(RealSenseError):
    """Exception raised when frames cannot be captured."""
    pass

# --- Capture Service ---

def capture_images():
    """
    Initializes a RealSense camera, captures, and aligns one pair of color and depth frames.

    Returns:
        A tuple containing (color_image, depth_image).

    Raises:
        NoDeviceError: If no camera is found.
        FrameCaptureError: If frames cannot be captured.
        RealSenseError: For other camera runtime errors.
    """
    pipeline = rs.pipeline()
    config = rs.config()

    # Check for a connected RealSense device
    context = rs.context()
    if len(context.devices) == 0:
        raise NoDeviceError("No RealSense device connected.")

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
        # for _ in range(5):
        #     pipeline.wait_for_frames()

        # Get a coherent pair of frames
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            raise FrameCaptureError("Could not capture valid frames from RealSense camera.")

        # Convert frames to numpy arrays
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        cv2.imwrite("captured_color_image_bef_return.png", color_image)


        return color_image, depth_image

    except RuntimeError as e:
        logging.error(f"RealSense runtime error: {e}", exc_info=True)
        # Wrap the generic RuntimeError in our custom exception
        raise RealSenseError(f"Error with RealSense camera: {e}") from e
    finally:
        # Ensure the pipeline is stopped
        pipeline.stop()
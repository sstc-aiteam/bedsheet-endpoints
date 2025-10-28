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

class RealSenseCaptureService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RealSenseCaptureService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initializes the RealSense pipeline and configuration."""
        self.pipeline = rs.pipeline()
        config = rs.config()

        context = rs.context()
        if len(context.devices) == 0:
            raise NoDeviceError("No RealSense device connected.")

        width, height, fps = 1280, 720, 30
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        try:
            self.profile = self.pipeline.start(config)
            logging.info("RealSense pipeline started.")
        except RuntimeError as e:
            raise RealSenseError(f"Failed to start RealSense pipeline: {e}") from e

        self.align = rs.align(rs.stream.color)

    def capture_images(self):
        """
        Captures and aligns one pair of color and depth frames.

        Returns:
            A tuple containing (color_image, depth_image).
        """
        try:
            for _ in range(5):
                self.pipeline.wait_for_frames()

            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            aligned_frames = self.align.process(frames)

            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not aligned_depth_frame or not color_frame:
                raise FrameCaptureError("Could not capture valid frames from RealSense camera.")

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            return color_image, depth_image

        except RuntimeError as e:
            logging.error(f"RealSense runtime error: {e}", exc_info=True)
            raise RealSenseError(f"Error with RealSense camera: {e}") from e

    def shutdown(self):
        """Stops the RealSense pipeline."""
        self.pipeline.stop()
        logging.info("RealSense pipeline stopped.")

# For dependency injection, we can use a single instance of the service.
rs_capture_service = RealSenseCaptureService()

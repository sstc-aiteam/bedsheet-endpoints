import numpy as np
import pyrealsense2 as rs
import logging

# Set up logging for clarity (Optional, but good practice)
logger = logging.getLogger(__name__)

# --- Custom Exceptions 
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
    
    # Define default stream parameters as class/instance attributes for easy access
    DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_FPS = 848, 480, 30

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RealSenseCaptureService, cls).__new__(cls)

            cls._instance.is_initialized = False
            cls._instance.align = None # Initialize to None
            cls._instance.depth_intrinsics = None
        return cls._instance

    def _initialize(self):
        """Initializes the RealSense pipeline and configuration, only if not already initialized."""
        
        if self.is_initialized:
            logger.info("RealSense pipeline is already initialized.")
            return

        try:
            logger.info("Attempting to initialize RealSense pipeline...")
            self.pipeline = rs.pipeline()
            config = rs.config()
            
            # 1. Check for device connection
            context = rs.context()
            if len(context.devices) == 0:
                logger.warning("No RealSense device connected.")
                raise NoDeviceError("No RealSense device connected.")
        
            # 2. Configure streams
            width, height, fps = self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT, self.DEFAULT_FPS
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

            # 3. Start pipeline
            self.profile = self.pipeline.start(config)
            self.align = rs.align(rs.stream.color)

            # 4. Get and store intrinsics
            depth_profile = self.profile.get_stream(rs.stream.depth).as_video_stream_profile()
            self.depth_intrinsics = depth_profile.get_intrinsics()
            
            self.is_initialized = True
            logging.info("✅ RealSense pipeline started successfully.")
            
        except RuntimeError as e:
            # Catch all expected (and unexpected) errors during initialization
            self.is_initialized = False
            # Clean up the pipeline if it was partially started
            try:
                self.pipeline.stop()
            except RuntimeError:
                pass # Ignore if stop fails (likely because it wasn't started)
            
            raise RealSenseError(f"Failed to start RealSense pipeline: {e}") from e


    def capture_images(self):
        """
        Captures and aligns one pair of color and depth frames.
        Will attempt to re-initialize the device if not connected.

        Returns:
            A tuple containing (color_image, depth_image).
        """
        try:
            # 1. Check/Re-initialize device if not ready       
            if not self.is_initialized:
                self._initialize() # Retry connection
                if not self.is_initialized:
                    # If retry failed, return empty images as requested
                    raise RealSenseError("RealSense device is not initialized after retry.")

            # 2. Capture frames (only runs if self.is_initialized is True)
            # Skip initial frames for auto-exposure/gain to settle
            for _ in range(5):
                # Using 100ms timeout for quick discard if device is flaky
                self.pipeline.wait_for_frames(timeout_ms=5000) 

            # Wait for the actual frames to process
            frames = self.pipeline.wait_for_frames(timeout_ms=3000)
            aligned_frames = self.align.process(frames)

            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not aligned_depth_frame or not color_frame:
                raise FrameCaptureError("Could not capture valid frames from RealSense camera.")

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            return color_image, depth_image

        except RuntimeError as e:
            # This handles errors during frame capture (e.g., device unplugged mid-run)
            logging.error(f"RealSense runtime error during capture: {e}. Device is now considered uninitialized.", exc_info=True)
            raise RealSenseError(f"Error with RealSense camera: {e}") from e

    def deproject_pixel_to_point(self, x: int, y: int, depth: float) -> list:
        """
        Deprojects a 2D pixel with depth to a 3D point in camera space.

        Args:
            x: The x-coordinate of the pixel.
            y: The y-coordinate of the pixel.
            depth: The depth value at (x, y) in meters.

        Returns:
            A list [X, Y, Z] representing the 3D point.
        """
        return rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, [x, y], depth)

    def shutdown(self):
        """Stops the RealSense pipeline, checking if it was ever initialized."""
        if self.is_initialized:
            try:
                self.pipeline.stop()
                logger.info("RealSense pipeline stopped.")
            except Exception as e:
                logger.error(f"Error while stopping pipeline: {e}")
            finally:
                self.is_initialized = False
                self.align = None
                self.depth_intrinsics = None
        else:
            logger.info("RealSense pipeline was not active/initialized, nothing to stop.")

# For dependency injection, we can use a single instance of the service.
# This will now attempt to initialize, but won't crash if it fails.
rs_capture_service = RealSenseCaptureService()

# --- Example Usage ---

def main():
    # Service might fail to initialize here if no device is connected, but the app continues
    service = rs_capture_service
    
    print("\n--- First Capture Attempt ---")
    color, depth = service.capture_images()
    print(f"Color Image Shape: {color.shape}")
    print(f"Depth Image Shape: {depth.shape}")
    print(f"Service initialized status: {service.is_initialized}")
    
    if not service.is_initialized:
        print("\n* If you connect a RealSense device now, the next call will try to connect! *")
    
    # Imagine a delay here where the device is either connected or disconnected
    import time
    time.sleep(1) 
    
    print("\n--- Second Capture Attempt (Retry or Normal Capture) ---")
    color2, depth2 = service.capture_images()
    print(f"Color Image Shape: {color2.shape}")
    print(f"Depth Image Shape: {depth2.shape}")
    print(f"Service initialized status: {service.is_initialized}")
    
    service.shutdown()

if __name__ == "__main__":
    main()
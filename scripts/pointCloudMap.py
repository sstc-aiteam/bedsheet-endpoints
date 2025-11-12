# First, import the necessary libraries
import pyrealsense2 as rs
import numpy as np
import cv2
from datetime import datetime

# Create a pipeline
pipeline = rs.pipeline()

# Create a config object
config = rs.config()

# Note: 848x480 is a recommended resolution for D400 series cameras.
# Enable depth and color streams
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 848, 480, rs.format.rgb8, 30)

# Start streaming
profile = pipeline.start(config)

# Get the depth sensor's depth scale
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print(f"Depth Scale is: {depth_scale}")

# Get stream profile and camera intrinsics
profile = pipeline.get_active_profile()
intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# Global variables to store mouse coordinates and 3D point
mouse_coords = (0, 0)
point_3d = (0, 0, 0)

def mouse_callback(event, x, y, flags, param):
    global mouse_coords
    if event == cv2.EVENT_MOUSEMOVE:
        # Ensure x is within the bounds of the color image
        color_image_width = 848 # As configured
        if x < color_image_width:
            mouse_coords = (x, y)

cv2.namedWindow('RealSense - Color and Depth', cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback('RealSense - Color and Depth', mouse_callback)

# Streaming loop
try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # Get the 2D coordinates from the mouse callback
        x, y = mouse_coords
        
        # Get depth value at the mouse cursor position
        depth_value = aligned_depth_frame.get_distance(x, y)
        
        # Deproject 2D pixel to 3D point
        if depth_value > 0:
            point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth_value)

        # Convert RGB to BGR for OpenCV
        color_image_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

        # Display the 3D coordinates on the color image
        coord_text = f"X: {point_3d[0]:.3f}m, Y: {point_3d[1]:.3f}m, Z: {point_3d[2]:.3f}m"
        cv2.putText(color_image_bgr, coord_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Apply colormap on depth image (for visualization)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        
        # Stack images horizontally
        images = np.hstack((color_image_bgr, depth_colormap))

        cv2.imshow('RealSense - Color and Depth', images)
        key = cv2.waitKey(1)

        # Press 's' to save the point cloud and color image
        if key in (ord('s'), ord('S')):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ply_filename = f"pointcloud_{timestamp}.ply"
            color_filename = f"color_image_{timestamp}.png"
            
            pc = rs.pointcloud()
            pc.map_to(color_frame)
            points = pc.calculate(aligned_depth_frame)
            print(f"Saving to {ply_filename}...")
            points.export_to_ply(ply_filename, color_frame)
            cv2.imwrite(color_filename, color_image_bgr)
            print(f"Saved {color_filename} and {ply_filename}")

        # Press esc or 'q' to close the image window
        if key & 0xFF in (ord('q'), ord('Q')) or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()

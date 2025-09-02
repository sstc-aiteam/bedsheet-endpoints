# file_client.py

import requests
import numpy as np
import cv2
import argparse
import logging
import sys
import base64
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def send_request(color_image_path: str, depth_image_path: str, api_url: str):
    """
    Reads color and depth images from files and sends them to the keypoint detection API.

    Args:
        color_image_path: Path to the color image file (e.g., .png, .jpg).
        depth_image_path: Path to the raw depth image file (.npy, containing depth in meters).
        api_url: URL of the API endpoint.
    """
    try:
        # 1. Open files in binary read mode to send them directly.
        # The server now expects the raw .png and .npy files, so no client-side conversion is needed.
        logging.info(f"Opening color image from {color_image_path}")
        with open(color_image_path, 'rb') as color_f:
            logging.info(f"Opening depth image from {depth_image_path}")
            with open(depth_image_path, 'rb') as depth_f:
                # 2. Prepare files for multipart upload.
                # We send the files as-is, letting the server handle decoding.
                files = {
                    'color_file': (os.path.basename(color_image_path), color_f, 'image/png'),
                    'depth_file': (os.path.basename(depth_image_path), depth_f, 'application/octet-stream')
                }

                # 3. Send request to the API
                logging.info(f"Sending request to {api_url}...")
                response = requests.post(api_url, files=files, timeout=30)
                response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)

        # 4. Process the response
        result = response.json()
        logging.info("Successfully received response.")

        keypoints = result.get('keypoints', [])
        logging.info(f"Detected keypoints: {keypoints}")

        # Decode and display the processed image returned by the API
        if 'processed_image' in result and result['processed_image']:
            img_base64 = result['processed_image']
            img_bytes = base64.b64decode(img_base64)
            img_nparr = np.frombuffer(img_bytes, np.uint8)
            processed_image = cv2.imdecode(img_nparr, cv2.IMREAD_COLOR)

            # Save the image to a file instead of displaying it
            base, ext = os.path.splitext(color_image_path)
            output_path = f"{base}_processed.png"
            
            cv2.imwrite(output_path, processed_image)
            logging.info(f"Saved processed image with keypoints to: {output_path}")
        else:
            logging.warning("Response did not contain a processed image.")

    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed: {e}")
        logging.error("Please ensure the FastAPI server is running and the URL is correct.")
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)


def main():
    """
    Main function to parse arguments and run the client.
    """
    parser = argparse.ArgumentParser(
        description="Client to send color and depth images to the keypoint detection API.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Example usage:
  python scripts/app_client.py color_1672531200.png depth_raw_1672531200.npy
"""
    )
    parser.add_argument("color_image", help="Path to the color image file (e.g., .png, .jpg).")
    parser.add_argument("depth_image", help="Path to the raw depth image file (.npy, containing depth in meters).")
    parser.add_argument("--url", default="http://127.0.0.1:8000/api/v1/detect_keypoints/", help="URL of the API endpoint.")
    
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    args = parser.parse_args()

    send_request(args.color_image, args.depth_image, args.url)

if __name__ == "__main__":
    main()
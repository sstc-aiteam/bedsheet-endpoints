
import logging
import sys

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from app.api.endpoints import router as api_router
from app.services.depth_keypoint_detector import depth_detector_service
from app.services.realsense_capture import rs_capture_service, NoDeviceError


app_logger = logging.getLogger("app")
app_logger.setLevel(logging.INFO)


app = FastAPI(
    title="Bed-making Keypoint Detection API",
    description="An API for detecting keypoints on a bedsheet using color and depth images.",
    version="0.2.0",

    # Set docs_url to None to disable the default CDN-based Swagger UI
    docs_url=None,
    # Optionally, disable ReDoc if you are not using it
    redoc_url=None,
    # Configure URLs for locally served Swagger UI assets.
    # These paths will be served by the StaticFiles mount below.
    swagger_ui_oauth2_redirect_url="/docs/oauth2-redirect",
    swagger_js_url="/static/swagger-ui-bundle.js",
    swagger_css_url="/static/swagger-ui.css"
)

# Mount the 'static' directory to serve Swagger UI files.
# The path is relative to where the application is run (the 'src' directory).
static_files_path = Path(__file__).parent / "app" / "static"
app.mount("/static", StaticFiles(directory=static_files_path), name="static")

from fastapi.openapi.docs import get_swagger_ui_html, get_swagger_ui_oauth2_redirect_html
from swagger_redirect import get_swagger_ui_html

@app.on_event("startup")
async def startup_event():
    """Load machine learning models on startup."""
    try:
        depth_detector_service.load_models()
    except NoDeviceError as e:
        app_logger.warning(f"Could not initialize RealSense camera on startup: {e}. The capture endpoints will not be available.")
    except Exception as e:
        app_logger.error(f"An unexpected error occurred during startup: {e}", exc_info=True)

@app.on_event("shutdown")
async def shutdown_event():
    """Gracefully shut down the RealSense pipeline."""
    rs_capture_service.shutdown()

app.include_router(api_router, prefix="/api/v1", tags=["Keypoint Detection"])

# Add custom endpoints to serve the Swagger UI.
# This replaces the default behavior that was disabled by setting docs_url=None.
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html()

@app.get(app.swagger_ui_oauth2_redirect_url, include_in_schema=False)
async def swagger_ui_redirect():
    return get_swagger_ui_oauth2_redirect_html()

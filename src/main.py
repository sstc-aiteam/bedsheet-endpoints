
import logging

from fastapi import FastAPI

from app.api.endpoints import router as api_router
from app.services.keypoint_detector import detector_service

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(
    title="Bed-making Keypoint Detection API",
    description="An API for detecting keypoints on a bedsheet using color and depth images.",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """Load machine learning models on startup."""
    detector_service.load_models()

app.include_router(api_router, prefix="/api/v1", tags=["Keypoint Detection"])
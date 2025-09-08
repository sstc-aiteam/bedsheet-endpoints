This repository contains a machine learning-powered web service designed to detect keypoints on a bedsheet from the aligned Color and Depth images. 
The system is built with Python and uses the FastAPI framework for the web API.

### Quick Start
1. Create python virtual runtime environment
```
python -m venv .bedsheet-endpoints
source .bedsheet-endpoints/bin/activate
```

2. Install python packages  
```
pip install -r requirements.txt
```

3. Start FastAPI service
```
cd src
nohup python -m uvicorn main:app --host ${HOST_FastAPI} --port 8000 --reload &
```

### Folder Structure
```
bedsheet-endpoints/
├── scripts/
│   └── app_client.py                        # client script here
|   └── save_align-depth2color.py            # script to save the aligned Color and Depth images 
├── src/
│   └── app/                                 # Main application package
│       ├── __init__.py
│       ├── api/
│       │   └── endpoints.py                 # API routes and endpoints
│       ├── core/
│       │   └── config.py                    # Configuration and settings
│       ├── services/
│       │   └── keypoint_detector.py         # Business logic for detection
│       ├── models/                          # Model network
│       │   ├── utils.py
│       │   └── yolo_vit.py
│       ├── weights/                         # Directory for model weight files
│       │   ├── keypoint_model_vit_depth.pth 
│       │   └── yolo_finetuned/
│       │        ├── best.pt
│       │        └── yolov8l.pt     
│       └── main.py                          # FastAPI app instantiation & startup
|
└── README.md                                # This file
```

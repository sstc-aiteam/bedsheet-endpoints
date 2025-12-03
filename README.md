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
|       |   └── realsense_capture.py         # Captures the spatial alignment between a color and a depth frame from an Intel RealSense         
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

### Commands
- check Jetpack version  
`apt show nvidia-jetpack`

- upgrade PIP to the latest available version  
`pip install --upgrade pip`



### Reference
- [SDK Manager](https://developer.nvidia.com/sdk-manager)
  - [Jetson AGX Orin 完整刷機](https://medium.com/@EricChou711/nvidia-jetson-agx-orin-%E5%AE%8C%E6%95%B4%E5%88%B7%E6%A9%9F-%E5%AE%89%E8%A3%9D-tensorflow-pytorch-opencv-%E6%95%99%E5%AD%B8-ubuntu%E7%AF%87-sdk-manager-b3395f654f75) 
- [Install PyTorch on the Jetson Orin Nano running JetPack 6.2](https://ninjalabo.ai/blogs/jetson_pytorch.html)
- [Installing PyTorch for Jetson Platform](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html)  
- [Your First Jetson Container](https://developer.nvidia.com/embedded/learn/tutorials/jetson-container)
  - [Cloud-Native on Jetson](https://developer.nvidia.com/embedded/jetson-cloud-native)
- [Jetson Orin + RealSense in 5 minutes](https://jetsonhacks.com/2025/03/20/jetson-orin-realsense-in-5-minutes/)
- [Intel® RealSense™ SDK](https://github.com/IntelRealSense/librealsense)
- [Azure Kinect SDK](https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/docs/usage.md)
- [NoMachine -Jetson Remote Desktop](https://jetsonhacks.com/2023/12/03/nomachine-jetson-remote-desktop/)
- Chromium browser installation
  - `sudo apt install chromium-browser` 
  - [Fixing Browser on Jetson Orin After Update](https://www.cytron.io/tutorial/fixing-browser-on-orin-nano?srsltid=AfmBOooYB3fVsAevCR_Jti3VefWsatnt-JAfQbPPSNdbTzsp0kFWLnzc)

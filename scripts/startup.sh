#!/bin/bash
source /home/nvidia/.bedsheet-endpoints/bin/activate
cd /home/nvidia/github/bedsheet-endpoints/src
nohup python -m uvicorn main:app --host 192.168.100.226 --port 8000 &

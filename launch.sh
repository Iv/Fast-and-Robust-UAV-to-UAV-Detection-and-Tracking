#!/bin/bash
nohup /home/iv/miniconda3/envs/uav_tracking/bin/python3.10 -u predict_drone.py --input ~/datasets/kal/DJI_0004.MOV --output Experiment_Results/Final/Video/DJI_0004_full.mov > processing.log 2>&1 &
echo $! > pid.txt

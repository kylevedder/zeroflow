#!/bin/bash

## Resource Request
#SBATCH --job-name=waymo_remove_ground
#SBATCH --output=waymo_remove_ground.out
#SBATCH --qos=ee-med
#SBATCH --partition=eaton-compute
#SBATCH --cpus-per-task=120
#SBATCH --mem-per-cpu=2GB
#SBATCH --time=04:00:00
#SBATCH --container-mounts=../../datasets/:/efs/,./:/project
#SBATCH --container-image=/home/kvedder/kylevedder+offline_sceneflow_waymo+latest.sqsh

## Job Steps
python extract_flow_and_remove_ground.py /efs/waymo_open_flow/ /efs/waymo_open_raster_heights/ /efs/waymo_open_processed_flow --cpus 120
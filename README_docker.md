# Docker Documentation for UAV Detection

This project provides a Dockerized environment for easy deployment and execution with GPU support.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (for GPU support)

## Building the Image

To build the Docker image:

```bash
docker build -t uav-drone-detection .
```

## Batch Processing All Videos

A helper script `run_batch_processing.sh` is provided to process all `.mov` files in a directory and save the results to another directory. It ensures that the output files are owned by the current user (not root).

### Usage

```bash
./run_batch_processing.sh <input_directory> <output_directory>
```

Example:

```bash
./run_batch_processing.sh /home/iv/datasets/Data/Videos /home/iv/runs/results
```

### Features
- **Ownership Handling**: Uses `--user $(id -u):$(id -g)` to ensure you can edit/delete output files without `sudo`.
- **Sanity Check**: Automatically runs a 100-frame test on the first video to verify GPU and Docker configuration before starting the full batch.
- **Resumable**: Skips videos that already have an output file in the results directory.

## Running a Single Video

You can also run a single video using the Docker CLI directly:

```bash
docker run --rm --gpus all \
  --user $(id -u):$(id -g) \
  -e HOME=/tmp \
  -v $(pwd)/Data:/app/Data:ro \
  -v $(pwd)/Experiment_Results:/app/Experiment_Results \
  -v $(pwd)/models:/app/models:ro \
  uav-drone-detection \
  python predict_drone.py --input "/app/Data/Videos/Clip_1.mov" --output "/app/Experiment_Results/output.mp4"
```

## Volumes

- `/app/Data/Videos`: Input video files (mounted as read-only).
- `/app/Experiment_Results`: Output detection results and videos.
- `/app/models`: Pre-trained models (mounted as read-only).

## GPU Support

The Docker image is based on `nvidia/cuda:12.2.2-runtime-ubuntu22.04`. It requires an NVIDIA GPU and the NVIDIA Container Toolkit installed on the host machine.

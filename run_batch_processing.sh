#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_directory> <output_directory>"
    exit 1
fi

# Resolve absolute paths
INPUT_DIR=$(realpath "$1")
mkdir -p "$2"
OUTPUT_DIR=$(realpath "$2")
PROJECT_ROOT=$(pwd)

# Verify if the input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory $INPUT_DIR does not exist."
    exit 1
fi

# Find the first .mov file for a sanity test
TEST_VIDEO=$(ls "$INPUT_DIR"/*.mov 2>/dev/null | head -n 1)
if [ -z "$TEST_VIDEO" ]; then
    echo "Error: No .mov files found in $INPUT_DIR."
    exit 1
fi
TEST_FILENAME=$(basename "$TEST_VIDEO")

echo "Sanity check on $TEST_FILENAME (100 frames)..."
# Use --user to preserve current user ownership of output files
# Set HOME=/tmp to avoid permission issues when Keras/TF writes cache/config
docker run --rm --gpus all \
    --user $(id -u):$(id -g) \
    -e HOME=/tmp \
    -v "$INPUT_DIR":/app/Data/Videos:ro \
    -v "$OUTPUT_DIR":/app/Experiment_Results \
    -v "$PROJECT_ROOT/models":/app/models:ro \
    uav-drone-detection:latest \
    python predict_drone.py --input "/app/Data/Videos/$TEST_FILENAME" --output "/app/Experiment_Results/test_run.mp4" --max_frames 100

# If the test run succeeded, proceed to batch processing
if [ $? -eq 0 ]; then
    echo "Test successful. Starting batch processing..."
    for video in "$INPUT_DIR"/*.mov; do
        filename=$(basename "$video")
        output_filename="${filename%.*}.mp4"
        
        # Skip processing if output file already exists
        if [ -f "$OUTPUT_DIR/$output_filename" ]; then
            echo "Skipping $filename (already exists)"
            continue
        fi

        echo "Processing $filename..."
        
        docker run --rm --gpus all \
            --user $(id -u):$(id -g) \
            -e HOME=/tmp \
            -v "$INPUT_DIR":/app/Data/Videos:ro \
            -v "$OUTPUT_DIR":/app/Experiment_Results \
            -v "$PROJECT_ROOT/models":/app/models:ro \
            uav-drone-detection:latest \
            python predict_drone.py --input "/app/Data/Videos/$filename" --output "/app/Experiment_Results/$output_filename"
    done
    echo "All videos processed. Results are saved in $OUTPUT_DIR"
else
    echo "Test failed. Please check your GPU/Docker configuration."
    exit 1
fi

# Automatic Hand Tracker  
*by Nevin Mathews Kuruvilla (nm4709)*

## Introduction  
**Automatic Hand Tracker** is a simple hand tracking tool designed for video inputs using Google MediaPipe for hand detection and SAM2 by Meta for tracking segmented hands. 

## Setup for macOS Users

1. **Create and activate the Conda environment**:
   ```bash
   conda create -n sam2 python=3.11
   conda activate sam2
   ```
   
2. **Install the required dependencies**:
   ```bash
    pip3 install -r requirements.txt
    ```
3. **Create necessary directories**:
   ```bash
    !mkdir -p checkpoints/
    !mkdir -p data/
    !mkdir -p data/frames/
    !mkdir -p logs
   ```
4. **Download the checkpoint model**:
  ```bash
  !wget -P /content/checkpoints/ https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
  !wget -q /content/checkpoints/ https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
  ```
**Note**: For Mac users, Ensure you use Python 3.11. There's a conflict with a package named decord when trying to run SAM2 on Python 3.12. (Source: https://github.com/dmlc/decord/issues/213)


**Upload your input video to the ./data directory with the name test.mp4.**

**To Run the Code:**
  ```bash
cd src
python3 -u main.py
  ```

**The result will be stored in the ./data directory with the name test_result.mp4**

## For Colab Users
If you're using Google Colab, there is a notebook available in the ./notebook directory named main.ipynb. You can run this notebook sequentially in Colab using GPU runtime.

## Recommended:
It is  recommended to use Google Colab as it provides GPU runtime, which will speed up the processing. On macOS, running SAM2 might take around 30-40 minutes.
Output

## Edge Cases Handled
The code base has been designed to handle the following edge cases:

- Single Hand: The tracker works if there is only one hand in the video.
- Multiple Hands of the Same Type: The tracker can handle cases where there are multiple hands of the same type (e.g., multiple left hands or multiple right hands).

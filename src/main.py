import torch
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2
import json
import os
from PIL import Image, ImageDraw
from sam2.build_sam import build_sam2_video_predictor
import supervision as sv

from utils import frames_generator, save_bounding_boxes

DATA_DIR = '../data'
FRAMES_DIR = '../data/frames'
CHECKPOINTS = '../checkpoints'
LOGS = '../logs'

device = torch.device("cpu")
CHECKPOINT = f"{CHECKPOINTS}/sam2.1_hiera_large.pt"
CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

predictor = build_sam2_video_predictor(CONFIG, CHECKPOINT, device=device)

#Part 1 - Detect hands in the first frame
def detect_hands(input_video_path):
    
    frames_generator(input_video_path)
    model_file = open(f'{CHECKPOINTS}/hand_landmarker.task', "rb")
    model_data = model_file.read()
    model_file.close()
    base_options = python.BaseOptions(model_asset_buffer=model_data)
    options = vision.HandLandmarkerOptions(base_options=base_options,num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)

    frames = sorted(f for f in os.listdir(f'{FRAMES_DIR}') if f.endswith(('.jpg')))
    first_frame_path = os.path.join(FRAMES_DIR, frames[0])
    
    image = mp.Image.create_from_file(first_frame_path)
    detection_result = detector.detect(image)

    #Save Bounding Boxes to prevent data loss
    save_bounding_boxes(detection_result)
    print('----------HAND DETECTION COMPLETED-------------------')


#Part 2 - SAM2 to track hands
def track_hands(input_video_path, output_video_path):
    #Initialize Inference state with all the frames
    inference_state = predictor.init_state(video_path=FRAMES_DIR)
    #Reset predictor if it has been used before
    predictor.reset_state(inference_state)

    with open(f'{DATA_DIR}/hand_bbox.json', 'r') as json_file:
        bbox = json.load(json_file)

    #types of hand classes
    hands = [0,1]
    for hand in hands:
        box = [b for b in bbox if b['hand_index'] == hand]

        #Handling the case where there is only one hand in the video
        if len(box) == 0:
            continue

        #Handling the case where there are multiple hands of the same type 
        #i.e multiple left hands or multiple right hands
        boxx = np.array([[b["bounding_box"]["x_min"], 
            b["bounding_box"]["y_min"], 
            b["bounding_box"]['x_max'], 
            b["bounding_box"]['y_max']]
            for b in box], 
        dtype = np.float32)

        _, object_ids, mask_logits = predictor.add_new_points(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=hand,
            box = boxx
        )

    #Propogating the masks to original video
    video_info = sv.VideoInfo.from_video_path(input_video_path)
    video_info.width = int(video_info.width)
    video_info.height = int(video_info.height)

    COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700']
    mask_annotator = sv.MaskAnnotator(
        color=sv.ColorPalette.from_hex(COLORS),
        color_lookup=sv.ColorLookup.CLASS)

    frame_sample = []
    SOURCE_FRAME_PATHS = sorted(sv.list_files_with_extensions(FRAMES_DIR, extensions=["jpg"]))
    with sv.VideoSink(output_video_path, video_info=video_info) as sink:
        for frame_idx, object_ids, mask_logits in predictor.propagate_in_video(inference_state):
            frame_path = SOURCE_FRAME_PATHS[frame_idx]
            frame = cv2.imread(frame_path)
            masks = (mask_logits > 0.0).cpu().numpy()
            masks = np.squeeze(masks).astype(bool)

            detections = sv.Detections(
                xyxy=sv.mask_to_xyxy(masks=masks),
                mask=masks,
                class_id=np.array(object_ids)
            )

            annotated_frame = mask_annotator.annotate(scene=frame.copy(), detections=detections)

            sink.write_frame(annotated_frame)
            if frame_idx % video_info.fps == 0:
                frame_sample.append(annotated_frame)    

    print(f'----------OUTPUT VIDEO SAVED AT {output_video_path}-------------------')


if __name__ == "__main__":
    input_video_path = f'{DATA_DIR}/test.mp4'
    output_video_path = f'{DATA_DIR}/test_result.mp4'

    detect_hands(input_video_path)
    track_hands(input_video_path, output_video_path)
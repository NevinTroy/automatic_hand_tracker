from PIL import Image, ImageDraw
import supervision as sv
import os
import json
DATA_DIR = '../data'
FRAMES_DIR = '../data/frames'

def frames_generator(source_video):
    videoInfo = sv.VideoInfo.from_video_path(source_video)

    START_IDX = 0
    END_IDX = videoInfo.total_frames 

    frames_generator = sv.get_video_frames_generator(source_video, start=START_IDX, end=END_IDX)
    images_sink = sv.ImageSink(
        target_dir_path=FRAMES_DIR,
        overwrite=True,
        image_name_pattern="{:05d}.jpg"
    )

    with images_sink:
        for frame in frames_generator:
            images_sink.save_image(frame)


def save_bounding_boxes(detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    bounding_boxes = []
    
    frames = sorted(f for f in os.listdir(FRAMES_DIR) if f.endswith(('.jpg', '.png')))
    first_frame_path = os.path.join(FRAMES_DIR, frames[0])
    img = Image.open(first_frame_path) 
  

    image_width = img.width 
    image_height = img.height 
    
    for idx, hand_landmarks in enumerate(hand_landmarks_list):
        
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]

        #Computing the Bounding Box
        x_min = min(x_coordinates)
        y_min = min(y_coordinates)
        x_max = max(x_coordinates)
        y_max = max(y_coordinates)


        x_min_pixel = int(x_min * image_width)
        y_min_pixel = int(y_min * image_height)
        x_max_pixel = int(x_max * image_width)
        y_max_pixel = int(y_max * image_height)

        # Store the bounding box with hand index
        bounding_boxes.append({
            "hand_index": idx,
            "bounding_box": {
                "x_min": x_min_pixel,
                "y_min": y_min_pixel,
                "x_max": x_max_pixel,
                "y_max": y_max_pixel
            }
        })

    if len(bounding_boxes) == 1:
        bounding_boxes.append({
            "hand_index": 2,
            "bounding_box": {
                "x_min": 0,
                "y_min": 0,
                "x_max": 0,
                "y_max": 0
            }
        })

    with open(f'{DATA_DIR}/hand_bbox.json', "w") as json_file:
        json.dump(bounding_boxes, json_file, indent=4)


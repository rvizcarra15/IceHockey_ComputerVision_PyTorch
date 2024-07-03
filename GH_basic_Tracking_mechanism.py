# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 20:44:30 2024

@author: Raúl Vizcarra Chirinos
"""

#**********************************LIBRARIES*********************************#
from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2


# INPUT-video file
video_path = 'D:/PYTHON/video_input.mp4'
# OUTPUT-Video File
output_video_path = 'D:/PYTHON/output_video.mp4'

# PICKLE FILE (IF ITS AVAILABLE LOADS IT, IF NOT, CREATES AND SAVES A NEW ONE IN THIS PATH)
pickle_path = 'D:/PYTHON/stubs/track_stubs.pkl'

#*********************************TRACKING MECHANISM**************************#

class HockeyAnalyzer:
    def __init__(self, model_path):
        self.model = YOLO(model_path) 
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        batch_size = 20 
        detections = [] 
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch
        return detections


#********LOAD TRACKS FROM FILE OR DETECT OBJECTS-SAVES PICKLE FILE************#

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks = {"person": []}

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # Tracking Mechanism
            detection_supervision = sv.Detections.from_ultralytics(detection)
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            tracks["person"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv.get('person', None):
                    tracks["person"][frame_num][track_id] = {"bbox": bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks
    
#**** Design of Ellipse for tracking players instead of Bounding boxes********#
    
    def draw_ellipse(self, frame, bbox, color, track_id=None, team=None):
        y2 = int(bbox[3])
        x_center = (int(bbox[0]) + int(bbox[2])) // 2
        width = int(bbox[2]) - int(bbox[0])
        color = (255, 0, 0)
        text_color = (255, 255, 255)
        
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width) // 2, int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )
    
        if track_id is not None:
            rectangle_width = 40
            rectangle_height = 20
            x1_rect = x_center - rectangle_width // 2
            x2_rect = x_center + rectangle_width // 2
            y1_rect = (y2 - rectangle_height // 2) + 15
            y2_rect = (y2 + rectangle_height // 2) + 15
    
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED)
    
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10
            font_scale = 0.4
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                thickness=2
            )

        return frame
    
#*********************** LABELS AND TRACK-IDs*********************************#

    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy() 
            player_dict = tracks["person"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                bbox = player["bbox"]
                
                # Draw ellipse in each player and Labels
                self.draw_ellipse(frame, bbox, (0, 255, 0), track_id)
            
                x1, y1, x2, y2 = map(int, bbox)
             
            output_video_frames.append(frame)

        return output_video_frames
    
#*************** EXECUTES TRACKING MECHANISM AND OUTPUT VIDEO****************#

# Read the video frames
video_frames = []
cap = cv2.VideoCapture(video_path)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    video_frames.append(frame)
cap.release()

#********************* EXECUTE TRACKING METHOD WITH YOLO**********************#
tracker = HockeyAnalyzer('D:/PYTHON/yolov8x.pt')
tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path=pickle_path)
annotated_frames = tracker.draw_annotations(video_frames, tracks)

#*********************** SAVES VIDEO FILE ************************************#
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
height, width, _ = annotated_frames[0].shape
out = cv2.VideoWriter(output_video_path, fourcc, 30, (width, height))

for frame in annotated_frames:
    out.write(frame)
out.release()

#*****************************************************************************#

#This method, takes in consideration some basic principles of the tracking 
#method approach developed by Abdullah Tarek (@codeinajiffy)in his tutorial: 
#"Build an AI/ML Football Analysis system with YOLO, OpenCV, and Python"
#Tutorial: https://www.youtube.com/watch?v=neBZ6huolkg



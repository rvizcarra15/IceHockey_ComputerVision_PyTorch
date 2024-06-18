# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:45:42 2024

@author: Raul Vizcarra Chirinos
"""

#
#This is an adaptation of the tracking method developed by Abdullah Tarek (@codeinajiffy) 
#in his tutorial: "Build an AI/ML Football Analysis system with YOLO, OpenCV, and Python"
#@codeinajiffy Tutorial: https://www.youtube.com/watch?v=neBZ6huolkg
#GitHub Repository: https://github.com/abdullahtarek/football_analysis/blob/main/main.py

import cv2
import numpy as np
from ultralytics import YOLO
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

class_names = ['Referee', 'Tm_white', 'Tm_yellow']


#******************************** RINK AND ZONE COORDINATES********************#

class HockeyAnalyzer:
    def __init__(self, model_path, classifier_path):
        self.model = YOLO(model_path)
        self.classifier = self.load_classifier(classifier_path)
        self.transform = transforms.Compose([
            transforms.Resize((150, 150)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.rink_coordinates = np.array([[-450, 710], [2030, 710], [948, 61], [352, 61]])
        self.previous_positions = {}
        self.team_stats = {
            'Tm_white': {'distance': 0, 'speed': [], 'count': 0, 'offensive_pressure': 0},
            'Tm_yellow': {'distance': 0, 'speed': [], 'count': 0, 'offensive_pressure': 0}
        }
        self.zone_white = [(180, 150), (1200, 150), (948, 61), (352, 61)]
        self.zone_yellow = [(-450, 681), (2030, 681), (1160, 150), (200, 150)]
        self.pixel_to_meter_conversion()


#**************** Homologation with real measures of the Hockey Rink********************#

    def pixel_to_meter_conversion(self):
        # Rink real dimensions in meters
        rink_width_m = 15
        rink_height_m = 30

        # Pixel coordinates for rink dimensions
        left_pixel, right_pixel = self.rink_coordinates[0][0], self.rink_coordinates[1][0]
        top_pixel, bottom_pixel = self.rink_coordinates[2][1], self.rink_coordinates[0][1]

        # Conversion factor
        self.pixels_per_meter_x = (right_pixel - left_pixel) / rink_width_m
        self.pixels_per_meter_y = (bottom_pixel - top_pixel) / rink_height_m

       # Conversion Factor applied
    def convert_pixels_to_meters(self, distance_pixels):
        return distance_pixels / self.pixels_per_meter_x, distance_pixels / self.pixels_per_meter_y

#******************************* Speed metrics******************************************#

    def calculate_speed(self, track_id, x_center, y_center, fps):
        current_position = (x_center, y_center)
        if track_id in self.previous_positions:
            prev_position = self.previous_positions[track_id]
            distance_pixels = np.linalg.norm(np.array(current_position) - np.array(prev_position))
            distance_meters_x, distance_meters_y = self.convert_pixels_to_meters(distance_pixels)
            speed_meters_per_second = (distance_meters_x**2 + distance_meters_y**2)**0.5 * fps
        else:
            speed_meters_per_second = 0
        self.previous_positions[track_id] = current_position
        return speed_meters_per_second

#*********************** Team Prediction using a CNN Model**********************************#

    def load_classifier(self, classifier_path):
        model = CNNModel()
        model.load_state_dict(torch.load(classifier_path, map_location=torch.device('cpu')))
        model.eval()
        return model

    def predict_team(self, image):
        with torch.no_grad():
            output = self.classifier(image)
            _, predicted = torch.max(output, 1)
            predicted_index = predicted.item()
            team = class_names[predicted_index]
        return team

#************ Design of Ellipse for tracking players instead of Bounding boxes**************#
    def draw_ellipse(self, frame, bbox, color, track_id=None, team=None):
        y2 = int(bbox[3])
        x_center = (int(bbox[0]) + int(bbox[2])) // 2
        width = int(bbox[2]) - int(bbox[0])
    
        if team == 'Referee':
            color = (0, 255, 255)
            text_color = (0, 0, 0)
        else:
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
    
#********************************* Frame detection****************************************#
    def detect_frames(self, frames):
        batch_size = 20 
        detections = [] 
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch
        return detections
    
#************************************** Game Stats****************************************#

    def update_team_stats(self, team, speed, distance, position):
        if team in self.team_stats:
            self.team_stats[team]['speed'].append(speed)
            self.team_stats[team]['distance'] += distance
            self.team_stats[team]['count'] += 1

            if team == 'Tm_white':
                if self.is_inside_zone(position, self.zone_white):
                    self.team_stats[team]['offensive_pressure'] += distance
            elif team == 'Tm_yellow':
                if self.is_inside_zone(position, self.zone_yellow):
                    self.team_stats[team]['offensive_pressure'] += distance

   
    def draw_stats(self, frame):
        avg_speed_white = np.mean(self.team_stats['Tm_white']['speed']) if self.team_stats['Tm_white']['count'] > 0 else 0
        avg_speed_yellow = np.mean(self.team_stats['Tm_yellow']['speed']) if self.team_stats['Tm_yellow']['count'] > 0 else 0
        distance_white = self.team_stats['Tm_white']['distance']
        distance_yellow = self.team_stats['Tm_yellow']['distance']

        offensive_pressure_white = self.team_stats['Tm_white'].get('offensive_pressure', 0)
        offensive_pressure_yellow = self.team_stats['Tm_yellow'].get('offensive_pressure', 0)
        
        Pressure_ratio_W = offensive_pressure_white/distance_white   *100  if self.team_stats['Tm_white']['distance'] > 0 else 0
        Pressure_ratio_Y = offensive_pressure_yellow/distance_yellow *100  if self.team_stats['Tm_yellow']['distance'] > 0 else 0

        table = [
            ["", "Away_White", "Home_Yellow"],
            ["Average Speed\nPlayer", f"{avg_speed_white:.2f} m/s", f"{avg_speed_yellow:.2f} m/s"],
            ["Distance\nCovered", f"{distance_white:.2f} m", f"{distance_yellow:.2f} m"],
            ["Offensive\nPressure %", f"{Pressure_ratio_W:.2f} %", f"{Pressure_ratio_Y:.2f} %"],
        ]

        text_color = (0, 0, 0)  
        start_x, start_y = 10, 590  
        row_height = 30  # Manage Height
        column_width = 150  # Manage Width
        font_scale = 1  

        def put_multiline_text(frame, text, position, font, font_scale, color, thickness, line_type, line_spacing=1.0):
            y0, dy = position[1], int(font_scale * 20 * line_spacing) 
            for i, line in enumerate(text.split('\n')):
                y = y0 + i * dy
                cv2.putText(frame, line, (position[0], y), font, font_scale, color, thickness, line_type)
        
        for i, row in enumerate(table):
            for j, text in enumerate(row):
                if i in [1,2, 3]:  
                    put_multiline_text(
                        frame,
                        text,
                        (start_x + j * column_width, start_y + i * row_height),
                        cv2.FONT_HERSHEY_PLAIN,
                        font_scale,
                        text_color,
                        1,
                        cv2.LINE_AA,
                        line_spacing= 0.8  
                    )
                else:
                    cv2.putText(
                        frame,
                        text,
                        (start_x + j * column_width, start_y + i * row_height),
                        cv2.FONT_HERSHEY_PLAIN,
                        font_scale,
                        text_color,
                        1,
                        cv2.LINE_AA,
                    )
  
    def is_inside_zone(self, position, zone):
          x, y = position
          n = len(zone)
          inside = False
          p1x, p1y = zone[0]
          for i in range(n + 1):
              p2x, p2y = zone[i % n]
              if y > min(p1y, p2y):
                  if y <= max(p1y, p2y):
                      if x <= max(p1x, p2x):
                          if p1y != p2y:
                              xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                          if p1x == p2x or x <= xinters:
                              inside = not inside
              p1x, p1y = p2x, p2y
          return inside

    def draw_semi_transparent_rectangle(self, frame):
        overlay = frame.copy()
        alpha = 0.7  # Transparency factor
        
        # Draw semi-transparent rectangle for Stats
        bottom_left = (0, 710)
        bottom_right = (450, 710)
        upper_left = (0, 570)
        upper_right = (450, 570)
        
        # Borders
        border_color = (169, 169, 169)  # Color
        border_thickness = 3
        cv2.rectangle(frame, upper_left, bottom_right, border_color, border_thickness)
        cv2.rectangle(overlay, upper_left, bottom_right, (128, 128, 128), -1)
        # Blend with the original frame
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw stats 
        self.draw_stats(frame)
        
        # Draw Offensive zone- White Team
        #cv2.polylines(frame, [np.array(self.zone_white)], isClosed=True, color=(128, 0, 128), thickness=2)
        
        # Draw Offensive zone- Yellow Team
        #cv2.polylines(frame, [np.array(self.zone_yellow)], isClosed=True, color=(0, 165, 255), thickness=2)


#******************* Tracking Mechanism (From pickle file)**********************************#

    def analyze_video(self, video_path, output_path, tracks_path):
          with open(tracks_path, 'rb') as f:
              tracks = pickle.load(f)

          cap = cv2.VideoCapture(video_path)
          if not cap.isOpened():
              print("Error: Could not open video.")
              return
          
          fps = cap.get(cv2.CAP_PROP_FPS)
          frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
          frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

          # Codec and VideoWriter object
          fourcc = cv2.VideoWriter_fourcc(*'XVID')
          out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

          frame_num = 0
          while cap.isOpened():
              ret, frame = cap.read()
              if not ret:
                  break
              
              # Draw Ice Hockey Rink Area
              mask = np.zeros(frame.shape[:2], dtype=np.uint8)
              cv2.fillConvexPoly(mask, self.rink_coordinates, 1)
              mask = mask.astype(bool)
              # Draw rink area
              #cv2.polylines(frame, [self.rink_coordinates], isClosed=True, color=(0, 255, 0), thickness=2)
              
              # Draw Stats-Rectangle
              self.draw_semi_transparent_rectangle(frame)
              
              # Get tracks from frame
              player_dict = tracks["person"][frame_num]
              for track_id, player in player_dict.items():
                  bbox = player["bbox"]

              # Check if the tracked object is within the Rink Area
                  x_center = int((bbox[0] + bbox[2]) / 2)
                  y_center = int((bbox[1] + bbox[3]) / 2)

                  if not mask[y_center, x_center]:
                      continue  

                  # Team Prediction
                  x1, y1, x2, y2 = map(int, bbox)
                  cropped_image = frame[y1:y2, x1:x2]
                  cropped_pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
                  transformed_image = self.transform(cropped_pil_image).unsqueeze(0)
                  team = self.predict_team(transformed_image)
                  
                  # Draw ellipse in each player and Labels
                  self.draw_ellipse(frame, bbox, (0, 255, 0), track_id, team)
                  
                  font_scale = 1  
                  text_offset = -20  
                  
                  if team == 'Referee':
                      rectangle_width = 60
                      rectangle_height = 25
                      x1_rect = x1
                      x2_rect = x1 + rectangle_width
                      y1_rect = y1 - 30
                      y2_rect = y1 - 5
                  
                      cv2.rectangle(frame,
                                    (int(x1_rect), int(y1_rect)),
                                    (int(x2_rect), int(y2_rect)),
                                    (0, 0, 0),  # Black color for rectangle
                                    cv2.FILLED)
                      text_color = (255, 255, 255)  # White color for text
                  else:
                      if team == 'Tm_white':
                          text_color = (255, 215, 0)  # BLUE
                      else:
                          text_color = (0, 255, 255)  # Yellow 
                  
                  # Draw Team label
                  cv2.putText(
                      frame,
                      team,
                      (int(x1), int(y1) + text_offset), 
                      cv2.FONT_HERSHEY_PLAIN,            
                      font_scale,
                      text_color,
                      thickness=2
                  )

                  speed = self.calculate_speed(track_id, x_center, y_center, fps)
                  
                  # Speed Label 
                  speed_font_scale = 0.8  
                  speed_y_position = int(y1) + 20
                  if speed_y_position > int(y1) - 5:
                      speed_y_position = int(y1) - 5

                  cv2.putText(
                      frame,
                      f"Speed: {speed:.2f} m/s",  
                      (int(x1), speed_y_position),  
                      cv2.FONT_HERSHEY_PLAIN,      
                      speed_font_scale,
                      text_color,
                      thickness=2
                  )

                  distance = speed / fps
                  position = (x_center, y_center)
                  self.update_team_stats(team, speed, distance, position)

              # Write output video
              out.write(frame)
              frame_num += 1

          cap.release()
          out.release()

# CNN -MODEL TEAMS AND REFEREE PREDICTIONS
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 18 * 18, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, len(class_names))  
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 18 * 18)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# MODEL INPUTS
model_path = 'D:/PYTHON/yolov8x.pt'
video_path = 'D:/PYTHON/video_input.mp4'
output_path = 'D:/PYTHON/output_video.mp4'
tracks_path = 'D:/PYTHON/stubs/track_stubs.pkl'
classifier_path = 'D:/PYTHON/hockey_team_classifier.pth'

# Execute YOLO-HockeyAnalyzer/classifier and Save Output
analyzer = HockeyAnalyzer(model_path, classifier_path)
analyzer.analyze_video(video_path, output_path, tracks_path)


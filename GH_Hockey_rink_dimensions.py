# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 10:42:03 2024

@author: Raúl Vizcarra Chirinos
"""
#********************* Border Definition for Frame***********************
import cv2

video_path = 'D:/PYTHON/video_input.mp4'
cap = cv2.VideoCapture(video_path)

#**************Read, Define and Draw corners of the frame****************
ret, frame = cap.read()

bottom_left = (0, 720)
bottom_right = (1280, 720)
upper_left = (0, 0)
upper_right = (1280, 0)

cv2.line(frame, bottom_left, bottom_right, (0, 255, 0), 2)
cv2.line(frame, bottom_left, upper_left, (0, 255, 0), 2)
cv2.line(frame, bottom_right, upper_right, (0, 255, 0), 2)
cv2.line(frame, upper_left, upper_right, (0, 255, 0), 2)

#*******************Save the frame with marked corners*********************
output_image_path = 'rink_area_marked_VALIDATION.png'
cv2.imwrite(output_image_path, frame)
print("Rink area saved:", output_image_path)

#********PLOT RINK IN CANVAS & CALIBRATE OFFENSIVE PRESSURE ZONES****************************

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# Offensive Zone - White Team
white_coords = [(180, 150), (1100, 150), (900, 61), (352, 61)]

# Ice Hockey Rink
rink_coords = [(-450, 710), (2030, 710), (948, 61), (352, 61)]

# Offensive Zone - Yellow Team
yellow_coords = [(-450, 710), (2030, 710), (1160, 150), (200, 150)]

#******************************PLOT THREE AREAS**********************************************

fig, ax = plt.subplots()
white_zone = Polygon(white_coords, closed=True, facecolor='orange', edgecolor='black')
rink_zone = Polygon(rink_coords, closed=True, fill=False, edgecolor='black')
yellow_zone = Polygon(yellow_coords, closed=True, facecolor='purple', edgecolor='black')


ax.add_patch(white_zone)
ax.add_patch(rink_zone)
ax.add_patch(yellow_zone)
plt.xlim(-600, 2200)
plt.ylim(0, 800)

# Show Plot
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()


        
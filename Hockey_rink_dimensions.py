# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 20:44:08 2024

@author: Raúl Vizcarra Chirinos
"""

#*************CALIBRATE RINK DIMENSIONS & OFFENSIVE PRESSURE ZONES****************************

import cv2

# Load video and read one frame
video_path = 'D:/PYTHON/video_input.mp4'
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
if not ret:
    print("Error: Could not read frame from the video.")
    exit()

# Get frame dimensions
height, width, _ = frame.shape

# Define corners of the bottom side of the rink 
original_bottom_left = (-450, height - 0)
original_bottom_right = (width + 750, height - 0)

# Increase or decrease Height
height_change_percentage = 1.3
height_change = int((height - 690) * height_change_percentage)

# y-coordinate of bottom corners
bottom_left = (original_bottom_left[0], original_bottom_left[1] - height_change)
bottom_right = (original_bottom_right[0], original_bottom_right[1] - height_change)

# CALIBRATE UPPER SIDE 
upper_width = int((width - 1) * 0.45)  
upper_left = (10 + (width - 20 - upper_width) // 2, 100 - height_change)
upper_right = (width - 80 - (width - 200 - upper_width) // 2, 100 - height_change)

#*********************************DRAW & PRINT COORDINATES**********************************
print("Bottom Left Corner:", bottom_left)
print("Bottom Right Corner:", bottom_right)
print("Upper Left Corner:", upper_left)
print("Upper Right Corner:", upper_right)

cv2.line(frame, bottom_left, bottom_right, (0, 255, 0), 2)
cv2.line(frame, bottom_left, upper_left, (0, 255, 0), 2)
cv2.line(frame, bottom_right, upper_right, (0, 255, 0), 2)
cv2.line(frame, upper_left, upper_right, (0, 255, 0), 2)

#****************************** SAVE & PRINT THE FRAME ********************************
output_image_path = 'D:/PYTHON/rink_area_marked.png'
cv2.imwrite(output_image_path, frame)
print("Rink area marked and saved to:", output_image_path)


#********PLOT RINK IN CANVAS & CALIBRATE OFFENSIVE PRESSURE ZONES****************************

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# Offensive Zone - White Team
white_coords = [(180, 199), (1200, 199), (948, 61), (352, 61)]

# Ice Hockey Rink
rink_coords = [(-450, 681), (2030, 681), (948, 61), (352, 61)]

# Offensive Zone - Yellow Team
yellow_coords = [(-450, 681), (2030, 681), (1160, 200), (200, 200)]

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







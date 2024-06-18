# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 19:48:50 2024

@author: Raúl Vizcarra Chirinos
"""
#
#This is an adaptation of the tracking method developed by Abdullah Tarek (@codeinajiffy) 
#in his tutorial: "Build an AI/ML Football Analysis system with YOLO, OpenCV, and Python"
#@codeinajiffy Tutorial: https://www.youtube.com/watch?v=neBZ6huolkg
#GitHub Repository: https://github.com/abdullahtarek/football_analysis/blob/main/main.py


#*********************************SETTINGS******************************************
import sys
import os

# Directory containing 'utils' 
utils_path = 'D:/PYTHON/utils'
if utils_path not in sys.path:
    sys.path.append(utils_path)

# Directory containing 'Tracker'
trackers_path = 'D:/PYTHON/Trackers'
if trackers_path not in sys.path:
    sys.path.append(trackers_path)
    
# Updated Python path
print("Updated Python path:", sys.path)

try:
    from video_utils import read_video, save_video
    from tracker import Tracker  # Tracking Method

except ImportError as e:
    print(f"ImportError: {e}")
    print("Check the location and contents of the 'video_utils' and 'tracker' modules.")

def main():
    # Read Video
    video_frames = read_video('D:/PYTHON/video_input.mp4')
    
#*********************************Initialize Tracker******************************************
    # CREATES TRACKER PKL FILE IF IT DOESNT EXISTS ALREADY
    tracker = Tracker('D:/PYTHON/yolov8x.pt')
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')
  

    # SAVES OUTPUT ON VIDEO
    output_video_frames = tracker.draw_annotations(video_frames, tracks)
    save_video(output_video_frames, 'D:/PYTHON/output_video.mp4')

if __name__ == '__main__':
    main()
    
    
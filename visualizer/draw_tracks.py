# foggy_uav_system/visualizer/draw_tracks.py

import cv2
import random

# 用于为每个 track_id 分配固定颜色
track_colors = {}

def get_color(track_id):
    if track_id not in track_colors:
        random.seed(track_id)
        track_colors[track_id] = tuple(random.randint(0, 255) for _ in range(3))
    return track_colors[track_id]

def draw_tracks(image, tracks):
    for track in tracks:
        x1, y1, x2, y2 = map(int, track['bbox'])
        track_id = track['track_id']
        color = get_color(track_id)
        label = f"ID {track_id}"

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

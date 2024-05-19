import pickle
import cv2
import numpy as np
import os
import sys
sys.path.append("../")
from utils import measure_distance_sqr


class CameraMovementEstimator:
    def __init__(self, frame):
        self.min_distance = 3       # minimum camera movement to "care" about (pixels)
        # Defining lk_params to use in OpticalFlow:
        self.lk_params = dict(
            winSize=(15, 15),       # window size
            maxLevel=2,             # downscaling
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        # Creating our mask:
        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # We choose to track the bottom and top of the frame, since those areas won't move much.
        #   We will extract features from here.
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:, 0:70] = 1
        mask_features[: 900:1050] = 1

        self.features = dict(
            maxCorners=100,        # the max. number of corners we utilize for goodFeaturesToTrack
            qualityLevel=0.3,      # quality of the features (higher => less features)
            minDistance=3,         # pixels distance between features
            blockSize=7,           # search size of the features
            mask=mask_features
        )

    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info["position"]
                    camera_movement = camera_movement_per_frame[frame_num]
                    position_adjusted = (position[0] - camera_movement[0],
                                         position[1] - camera_movement[1])
                    tracks[object][frame_num][track_id]["position_adjusted"] = position_adjusted

    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        # Read from the stub, if available:
        if read_from_stub and (stub_path is not None) and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                return pickle.load(f)

        # Calculating camera movement:
        camera_movement = [[0, 0]]*len(frames)
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        prev_features = cv2.goodFeaturesToTrack(prev_gray, **self.features)

        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, prev_features,
                                                          None, **self.lk_params)
            max_distance_sqr = 0
            camera_movement_x, camera_movement_y = 0, 0

            for i, (new, old) in enumerate(zip(new_features, prev_features)):
                new_features_point = new.ravel()
                old_features_point = old.ravel()

                # Calculating squared distance:
                distance_sqr = measure_distance_sqr(new_features_point, old_features_point)
                if distance_sqr > max_distance_sqr:
                    max_distance_sqr = distance_sqr
                    camera_movement_x = new_features_point[0] - old_features_point[0]
                    camera_movement_y = new_features_point[1] - old_features_point[1]

            if max_distance_sqr >= self.min_distance**2:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                prev_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)

            prev_gray = frame_gray.copy()

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(camera_movement, f)

        return camera_movement

    def draw_camera_movement(self, frames, camera_movement_per_frame):
        output_frames = []

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            x_movement, y_movement = camera_movement_per_frame[frame_num]
            frame = cv2.putText(frame, f"Camera Movement (x): {x_movement:.2f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            frame = cv2.putText(frame, f"Camera Movement (y): {y_movement:.2f}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

            output_frames.append(frame)

        return output_frames

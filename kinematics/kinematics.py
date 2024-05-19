import sys
sys.path.append("../")
from utils import measure_distance_sqr, get_bbox_foot_center
import math
import cv2


class Kinematics:
    def __init__(self):
        self.frame_window = 3       # calculating player speed once every frame_window frames
        self.framerate = 24         # FPS

    def add_kinematics_to_tracks(self, tracks):
        total_distance = {}

        for object, object_tracks in tracks.items():
            # We don't detect the kinematics of balls and referees:
            if (object == "ball") or (object == "referee"):
                continue

            number_of_frames = len(object_tracks)
            # Iterating through the frames in size self.frame_window:
            for frame_num in range(0, number_of_frames, self.frame_window):
                # To avoid going out of bounds (IndexError):
                last_frame = min(frame_num + self.frame_window, number_of_frames) - 1

                for track_id, _ in object_tracks[frame_num].items():
                    # If the track is in the first frame but not in the last frame, we skip.
                    #   (we need the player to exist in the first and last frame to calculate
                    #    both speed and distance).
                    if track_id not in object_tracks[last_frame]:
                        continue

                    # We use the transformed position (in meters) for the start position:
                    start_position = object_tracks[frame_num][track_id]["position_transformed"]
                    end_position = object_tracks[last_frame][track_id]["position_transformed"]

                    if start_position is None or end_position is None:
                        continue

                    # We calculate the distance covered, and speed:
                    distance_covered_sqr = measure_distance_sqr(start_position, end_position)
                    distance_covered = math.sqrt(distance_covered_sqr)
                    time_elapsed = (last_frame - frame_num) / self.framerate
                    speed_meters_per_second = distance_covered / time_elapsed
                    speed_km_per_hour = speed_meters_per_second * 3.6

                    # Now we add this instance to a dictionary to store total distances:
                    if object not in total_distance:
                        total_distance[object] = {}
                    # Initializing total_distance of this object to zero if non-existent:
                    if track_id not in total_distance[object]:
                        total_distance[object][track_id] = 0

                    total_distance[object][track_id] += distance_covered

                    for frame_num_batch in range(frame_num, last_frame):
                        if track_id not in tracks[object][frame_num_batch]:
                            continue
                        tracks[object][frame_num_batch][track_id]["speed_kmph"] = speed_km_per_hour
                        tracks[object][frame_num_batch][track_id]["distance"] = total_distance[object][track_id]

    def draw_speed_and_distance(self, frames, tracks):
        output_frames = []

        for frame_num, frame in enumerate(frames):
            for object, object_tracks in tracks.items():
                if (object == "ball") or (object == "referee"):
                    continue
                for _, track_info in object_tracks[frame_num].items():
                    if "speed_kmph" in track_info:
                        speed = track_info.get("speed_kmph", None)
                        distance = track_info.get("distance", None)

                        if speed is None or distance is None:
                            continue

                        bbox = track_info["bbox"]
                        position = get_bbox_foot_center(bbox)
                        position = list(position)
                        # We add a buffer before drawing the text for speed:
                        position[1] += 40

                        position = tuple(map(int, position))

                        # Adding the speed:
                        cv2.putText(frame, f"{speed:.2f} km/h", position,
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                        # Adding the distance:
                        cv2.putText(frame, f"{distance:.2f} m", (position[0], position[1]+16),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            output_frames.append(frame)

        return output_frames



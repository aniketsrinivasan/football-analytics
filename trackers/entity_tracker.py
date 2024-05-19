from utils import get_bbox_center, get_bbox_width
from ultralytics import YOLO
import supervision as sv
import numpy as np
import pandas as pd
import pickle
import os
import cv2
import sys
sys.path.append("../")


class Tracker:
    def __init__(self, model_path):
        # Load the model and load the tracker:
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def interpolate_ball_positions(self, ball_positions):
        # We want to convert the ball positions into a Pandas dataframe first.
        #   1. create a list using current ball_positions
        #       x.get(1, {}) gets track_id "1" from the positions (this is our ball)
        #       (otherwise empty dict)
        #   2. get the bbox of this track_id
        #       (otherwise empty list)
        #   3. convert to DataFrame
        ball_positions = [x.get(1, {}).get("bbox", []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=["x1", "y1", "x2", "y2"])

        # Interpolate any empty values:
        #   edge case:  if first frame is missing a ball bbox
        df_ball_positions = df_ball_positions.interpolate()
        #   solution:   we can backfill it
        df_ball_positions = df_ball_positions.bfill()

        # Converting back to our original format:
        #   1:            track_id of this ball
        #   {"bbox": x}:  how the bbox was stored originally where x is the bbox itself (list)
        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # Draw semi-transparent rectangle:
        overlay = frame.copy()
        cv2.rectangle(overlay, pt1=(1400, 900), pt2=(1920, 1000), color=(255, 255, 255))
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)

        # Calculating the ball control percentage:
        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]
        team_1_percentage = team_1_num_frames/(team_1_num_frames+team_2_num_frames)
        team_2_percentage = team_2_num_frames/(team_1_num_frames+team_2_num_frames)

        cv2.putText(frame, f"Team 1 Ball Acquisition: {team_1_percentage*100:.2f}%",
                    (1420, 920), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"Team 2 Ball Acquisition: {team_2_percentage * 100:.2f}%",
                    (1420, 970), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        return frame

    def detect_frames(self, frames) -> list:
        # Restrict BATCH_SIZE to avoid memory issues:
        #   (detections will be done in batches)
        BATCH_SIZE = 20

        detections = []
        # Iterating over all the frames, doing self.model.predict in batches:
        for i in range(0, len(frames), BATCH_SIZE):
            detections_batch = self.model.predict(frames[i:i+BATCH_SIZE], conf=0.1)
            detections += detections_batch      # Accumulating predicted batches
        # Note:
        #   We predict instead of track here because our model is not great at consistently
        #   detecting the goalkeeper. Therefore, we will first predict, and then override
        #   (merge) the goalkeeper and player detection classes, and then track normally.
        return detections

    def get_entity_tracks(self, frames, read_from_stub=False, stub_path=None) -> dict[str, list]:
        # We will check whether read_from_stub==True, in which case we will
        #   read directly from here rather than running predictions.
        # Else if stub_path is provided, we will run and save our results to this path.

        if read_from_stub and (stub_path is not None) and (os.path.exists(stub_path)):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        # Getting our predictions:
        detections = self.detect_frames(frames)

        # Initializing a "tracks" structure (a dictionary of lists) to easily reference:
        #   For each frame, we will append a dictionary to the relevant key in "tracks"
        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }
        tracks["players"].append({})
        tracks["referees"].append({})
        tracks["ball"].append({})

        for frame_i, detection in enumerate(detections):
            # Get the detection classes from this detection:
            class_names = detection.names       # {0:"person", 1:"referee", ...}
            class_names_inverse = {value:key for key, value in class_names.items()}

            # Convert to supervision detection format sv.Detection:
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert "goalkeeper" to "player" class:
            for object_idx, class_id in enumerate(detection_supervision.class_id):
                if class_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_idx] = class_names_inverse["player"]

            # Tracking objects:
            detections_tracked = self.tracker.update_with_detections(detection_supervision)

            for frame_detection in detections_tracked:
                # Extracting bounding box:
                #   Detections(array[xyxy: array], mask: bool, conf: list[float], class_id: array[int], ...)
                #   => frame_detection[0] gives an array of all the bounding boxes in this frame
                bbox = frame_detection[0].tolist()
                # Extracting class IDs:
                class_id = frame_detection[3]
                # Extracting track IDs:
                track_id = frame_detection[4]

                if class_id == class_names_inverse["player"]:
                    tracks["players"][frame_i][track_id] = {"bbox":bbox}
                if class_id == class_names_inverse["referee"]:
                    tracks["referees"][frame_i][track_id] = {"bbox":bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]
                if class_id == class_names_inverse["ball"]:
                    tracks["ball"][frame_i][1] = {"bbox":bbox}

        # By the end of this process, tracks will look something like this:
        #   {
        #    "players": [
        #                {0:{"bbox":[xm, ym, xM, yM]}, 1:{"bbox":[xm, ym, xM, yM]}, ...},
        #                {0:{"bbox":[xm, ym, xM, yM]}, 1:{"bbox":[xm, ym, xM, yM]}, ...},
        #                ...
        #               ],
        #    ...
        #   }
        # i.e. "players" is a list of dictionaries, where
        #       each dictionary consists of player_id:bounding_box pairs, encompassing all players,
        #       and there is one such dictionary for each frame of the video.
        # This is for all three classes tracked.

        # Checking if stub_path is provided, for savefile purposes:
        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)

        return tracks

    @staticmethod
    def draw_ellipse(frame, bbox, colour, player_id=None):
        # We want the ellipse drawn at the bottom of the bbox.
        yM = int(bbox[3])       # bbox[3] is bottom y-coordinate
        # Note: we don't use y_center, since we want yM instead
        x_center, _ = get_bbox_center(bbox)     # storing y_center in _ (unused)
        width = get_bbox_width(bbox)

        cv2.ellipse(frame,                                  # where to draw ellipse
                    center=(x_center, yM),                  # defining center
                    axes=(int(width), int(0.3 * width)),    # ellipse dimensions
                    angle=0.0,                              # angle of rotation
                    startAngle=-50,                         # angle to start drawing
                    endAngle=230,                           # angle to end drawing
                    color=colour,                           # colour of the ellipse
                    thickness=2,                            # thickness of ellipse
                    lineType=cv2.LINE_4                     # type of line drawn
                    )

        # We also draw the rectangle and the player_id inside it.
        rect_width = 40
        rect_height = 20
        rect_x1 = x_center - rect_width // 2
        rect_x2 = x_center + rect_width // 2
        rect_y1 = (yM - rect_height // 2) + 15              # 15 is a buffer
        rect_y2 = (yM + rect_height // 2) + 15              # 15 is a buffer

        # If the current frame is for a player, we add a label for player_id:
        if player_id is not None:
            cv2.rectangle(frame,
                          (int(rect_x1), int(rect_y1)),
                          (int(rect_x2), int(rect_y2)),
                          colour,
                          cv2.FILLED
                          )

            text_x1 = rect_x1 + 12                          # 12 is padding
            if player_id > 9:
                text_x1 += -5                               # in case number is large
            if player_id > 99:
                text_x1 += -5                               # in case number is large

            cv2.putText(frame,
                        f"{player_id}",
                        (int(text_x1), int(rect_y1 + 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2
                        )

        return frame

    def draw_triangle(self, frame, bbox, colour):
        # Setting the height and desired half-width of the triangle:
        DEL_HEIGHT = 15
        DEL_WIDTH_HALF = 7
        # We want the triangle to be above the ball.
        coord_y = int(bbox[1])
        coord_x, _ = get_bbox_center(bbox)      # x-coord should be centered

        # Defining the triangle:
        triangle_points = np.array([
            [coord_x, coord_y],                                 # "bottom point" of triangle
            [coord_x - DEL_WIDTH_HALF, coord_y - DEL_HEIGHT],   # "top left" of triangle
            [coord_x + DEL_WIDTH_HALF, coord_y - DEL_HEIGHT]    # "top right" of triangle
        ])

        # Drawing "inside" of the triangle:
        cv2.drawContours(frame, [triangle_points], 0, colour, thickness=cv2.FILLED)
        # Drawing "outline" of the triangle:
        cv2.drawContours(frame, [triangle_points], 0, colour, thickness=2)

        return frame

    # We want to create a custom visualization for the analytics:
    def annotations(self, video_frames, tracks, team_ball_control):
        # Initialize an empty list for the output frames:
        output_frames = []
        # Iterating over the video frames we have:
        for frame_i, frame in enumerate(video_frames):
            # We create a copy of our original frame to avoid modifying it:
            #   (want to preserve original list video_frames)
            frame = frame.copy()

            # Draw a circle beneath the player/object:
            #   e.g. tracks["players"][frame_i] gets the BBoxes with
            #   class "player" at frame_i (with IDs)
            player_dict = tracks["players"][frame_i]
            referee_dict = tracks["referees"][frame_i]
            ball_dict = tracks["ball"][frame_i]

            # Draw the player annotations:
            #   (iterates over each player in a given frame)
            for player_id, player in player_dict.items():
                # player_id is the tracked ID
                # player is a dictionary of the form {"bbox":[xm, ym, xM, yM]}

                # Setting the team colour of this player:
                team_colour = player.get("team_colour", (0, 0, 255))
                frame = self.draw_ellipse(frame=frame, bbox=player["bbox"],
                                          colour=team_colour, player_id=player_id)

                # Drawing player ball-acquisition if the player has the ball:
                if player.get("has_ball", False):
                    frame = self.draw_triangle(frame=frame, bbox=player["bbox"], colour=(0, 0, 255))

            # Draw the referee annotations:
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame=frame, bbox=referee["bbox"], colour=(0, 0, 0))

            # Draw the ball annotations:
            for ball_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame=frame, bbox=ball["bbox"], colour=(0, 255, 0))

            # Drawing team acquisition information:
            frame = self.draw_team_ball_control(frame=frame, frame_num=frame_i,
                                                team_ball_control=team_ball_control)

            # Appending this modified frame to output_frames:
            output_frames.append(frame)

        return output_frames

from utils import read_video, save_video
from trackers import Tracker
from team_clustering import TeamAssigner
from ball_control_assigner import PlayerBallAssigner
import numpy as np

ROOT_DIR = "/Users/aniket/PycharmProjects/footballAnalytics"


def main():
    # Read video:
    video_frames = read_video(ROOT_DIR + "/football_data/sample_data/0a2d9b_5.mp4")
    print(f"Successfully read video frames: {ROOT_DIR + '/football_data/sample_data/0a2d9b_5.mp4'}")

    # Create the tracker (using the current model):
    tracker = Tracker(ROOT_DIR + "/models/best.pt")
    print(f"Successfully created tracker: {ROOT_DIR + 'models/best.pt'}")

    # Get the tracks (detections) of the current video:
    tracks = tracker.get_entity_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path=ROOT_DIR+"stubs/track_stubs.pkl")
    # Interpolate ball positions:
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    print(f"Tracked objects.")

    # Assign player teams using k-means clustering:
    team_assigner = TeamAssigner()
    # Clustering teams based on only the first frame:
    team_assigner.assign_team_colour(video_frames[0], tracks["players"[0]])

    # Iterating over each frame in "tracks":
    for frame_number, player_track in enumerate(tracks["players"]):
        # Iterating over each player in the frame:
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_number],
                                                 track["bbox"],
                                                 player_id)
            tracks["players"][frame_number][player_id]["team"] = team
            tracks["players"][frame_number][player_id]["team_colour"] = team_assigner.team_colours[team]

    # Initializing ball acquisition data collection:
    team_ball_control = []
    # Assigning ball to player:
    player_assigner = PlayerBallAssigner()
    for frame_number, player_tracks in enumerate(tracks["players"]):
        ball_bbox = tracks["ball"][frame_number][1]["bbox"]
        assigned_player = player_assigner.assign_player_ball(player_tracks, ball_bbox)

        if assigned_player is not None:
            # Creating a new attribute in the dictionary, called "has_ball":
            tracks["players"][frame_number][assigned_player]["has_ball"] = True
            # Adding this team number to the ball control list:
            team_ball_control.append(tracks["players"][frame_number][assigned_player]["team"])
        else:
            # Append the last team to have the ball:
            team_ball_control.append(team_ball_control[-1])

    # Convert to np.ndarray to draw ball control percentage annotations:
    team_ball_control = np.array(team_ball_control)

    # Drawing the annotations to get the output video frames:
    output_video_frames = tracker.annotations(video_frames, tracks, team_ball_control)

    # Save video
    save_video(output_video_frames, ROOT_DIR + "/football_data/output_videos/output_video.avi")
    print(f"Saved: {ROOT_DIR + '/football_data/output_videos/output_video.avi'}")


if __name__ == "__main__":
    main()
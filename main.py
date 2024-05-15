from utils import read_video, save_video
from trackers import Tracker
from team_clustering import TeamAssigner

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

    # Drawing the annotations to get the output video frames:
    output_video_frames = tracker.annotations(video_frames, tracks)

    # Save video
    save_video(output_video_frames, ROOT_DIR + "/football_data/output_videos/output_video.avi")
    print(f"Saved: {ROOT_DIR + '/football_data/output_videos/output_video.avi'}")


if __name__ == "__main__":
    main()
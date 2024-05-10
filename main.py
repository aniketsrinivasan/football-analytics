from utils import read_video, save_video
from trackers import Tracker

ROOT_DIR = "/Users/aniket/PycharmProjects/footballAnalytics"

def main():
    # Read video:
    video_frames = read_video(ROOT_DIR + "/football_data/sample_data/0a2d9b_5.mp4")
    print(f"Successfully read video frames: {ROOT_DIR + '/football_data/sample_data/0a2d9b_5.mp4'}")

    tracker = Tracker(ROOT_DIR + "/models/best.pt")
    print(f"Successfully created tracker: {ROOT_DIR + 'models/best.pt'}")

    '''tracks = tracker.get_entity_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path=ROOT_DIR+"stubs/track_stubs.pkl")'''
    print(f"Tracked objects.")

    # Save video
    save_video(video_frames, ROOT_DIR + "/football_data/output_videos/output_video.avi")
    print(f"Saved: {ROOT_DIR + '/football_data/output_videos/output_video.avi'}")

if __name__ == "__main__":
    main()
from utils import read_video,save_video
from trackers import Tracker

def main():
    # We first read the video
    video_frames = read_video('input_videos/08fd33_4.mp4')

    tracker = Tracker('google_colab_trained_models/best.pt')
    tracks = tracker.get_object_tracks(video_frames,read_tracks=True,tracks_path='tracks/tracks_stub.pkl')

    # Then we save the video
    save_video(video_frames,'output_videos/output_video.avi')

if __name__=='__main__':
    main()
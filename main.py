import cv2
from utils import read_video,save_video
from trackers import Tracker
from team_assigner import TeamAssigner

def main():
    # We first read the video
    video_frames = read_video('input_videos/08fd33_4.mp4')

    tracker = Tracker('google_colab_trained_models/best.pt')
    tracks = tracker.get_object_tracks(video_frames,read_tracks=True,tracks_path='tracks/tracks_stub.pkl')

    # Assign teams to players
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0],tracks["players"][0])

    for frame_num,frame in enumerate(video_frames):
        for player_id, player in tracks["players"][frame_num].items():
            team = team_assigner.get_player_team(frame,player["bbox"],player_id)
            tracks["players"][frame_num][player_id]["team"] = team
            tracks["players"][frame_num][player_id]["team_color"] = team_assigner.team_colors[team]

    # Draw outputs (tracks)
    output_video_frames = tracker.draw_tracking_ids(video_frames,tracks)

    # Then we save the video
    save_video(output_video_frames,'output_videos/output_video.avi')

if __name__=='__main__':
    main()
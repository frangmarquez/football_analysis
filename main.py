import cv2
import numpy as np
from utils import read_video,save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from posession_asigner import PlayerAssigner

def main():

    # We first read the video
    video_frames = read_video('input_videos/08fd33_4.mp4')

    tracker = Tracker('google_colab_trained_models/best.pt')
    tracks = tracker.get_object_tracks(video_frames,
                                       read_tracks=True,
                                       tracks_path='tracks/tracks_stub.pkl')
    
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])
             
    # Assign teams to players
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0],tracks["players"][0])

    for frame_num,frame in enumerate(video_frames):
        for player_id, player in tracks["players"][frame_num].items():
            team = team_assigner.get_player_team(frame,player["bbox"],player_id)
            tracks["players"][frame_num][player_id]["team"] = team
            tracks["players"][frame_num][player_id]["team_color"] = team_assigner.team_colors[team]

    
    # Assign ball possession to player and team
    team_with_possesion = []

    player_assigner = PlayerAssigner()
    for frame_num, player_tracks in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        player_with_posession = player_assigner.assign_ball_to_player(player_tracks, ball_bbox)

        if player_with_posession != -1:
            tracks['players'][frame_num][player_with_posession]['possesion'] = True
            team_with_possesion.append(player_tracks[player_with_posession]['team'])
        else:
            team_with_possesion.append(team_with_possesion[-1])
            
    team_with_possesion = np.array(team_with_possesion)

    # Draw outputs (tracks)
    output_video_frames = tracker.draw_output(video_frames,tracks,team_with_possesion)

    # Then we save the video
    save_video(output_video_frames,'output_videos/output_video.avi')

if __name__=='__main__':
    main()
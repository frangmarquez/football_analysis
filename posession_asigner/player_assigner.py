import sys
sys.path.append('../')
from utils import get_center_of_bbox, measure_distance

class PlayerAssigner:
    def __init__(self):
        self.max_player_ball_distance = 65

    def assign_ball_to_player(self, players, ball_bbox):
        ball_pos = get_center_of_bbox(ball_bbox)

        minimum_distance = 9999999999
        assigned_player = -1

        for player_id,player_info in players.items():
            player_bbox = player_info["bbox"]

            distance_left_foot = measure_distance((player_bbox[0], player_bbox[3]),ball_pos)
            distance_right_foot = measure_distance((player_bbox[2], player_bbox[3]),ball_pos)
            distance_to_ball = min(distance_left_foot, distance_right_foot)

            if distance_to_ball < minimum_distance and distance_to_ball < self.max_player_ball_distance:
                assigned_player = player_id
                minimum_distance = distance_to_ball

        return assigned_player

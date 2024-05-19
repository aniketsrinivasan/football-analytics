import sys
sys.path.append('../')
from utils import get_bbox_center, measure_distance_sqr


class PlayerBallAssigner:
    def __init__(self):
        self.max_player_ball_distance = 70      # maximum distance for ball assignment

    def assign_player_ball(self, players, ball_bbox):
        # We get the center of the ball_bbox to get the ball's approximate location:
        ball_position = get_bbox_center(ball_bbox)

        min_distance_sqr = float("inf")
        assigned_player = None

        # Find the closest player and assign the ball to that player, if they are at most
        #   max_player_ball_distance pixels from the ball.
        for player_id, player in players.items():
            player_bbox = player["bbox"]

            # We calculate the bottom-left distance and bottom-right distance, and take the minimum:
            distance_left_sqr = measure_distance_sqr((player_bbox[0], player_bbox[-1]), ball_position)
            distance_right_sqr = measure_distance_sqr(((player_bbox[2]), player_bbox[-1]), ball_position)
            distance_sqr = min(distance_left_sqr, distance_right_sqr)

            if distance_sqr < self.max_player_ball_distance**2:
                if distance_sqr < min_distance_sqr:
                    min_distance_sqr = distance_sqr
                    assigned_player = player_id

        return assigned_player

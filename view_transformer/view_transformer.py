import numpy as np
import cv2


class ViewTransformer:
    def __init__(self):
        court_width = 68        # width of the football court (m)
        court_height = 105      # height of the football court (m)

        # Pixel vertices found manually:
        self.pixel_vertices = np.array([
            [110, 1035],
            [265, 275],
            [910, 260],
            [1640, 915]
        ])

        # The target dimensions of the initial pixel vertices:
        #   this defines the perspective transformation (trapezoid --> rectangle)
        self.target_vertices = np.array([
            [0, court_width],
            [0, 0],
            [court_height, 0],
            [court_height, court_width]
        ])

        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

    def transform_point(self, point):
        p = (int(point[0]), int(point[1]))
        # Check whether the point is inside our "trapezoid" defined by pixel_vertices
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0

        # Return None if it's outside:
        if not is_inside:
            return None

        # Reshaping so it can be read by the perspective transformer:
        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)
        transformed_point = cv2.perspectiveTransform(reshaped_point, self.perspective_transformer)

        return transformed_point.reshape(-1, 2)

    def add_transformed_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info["position_adjusted"]
                    position = np.array(position)
                    # Transform this point into rectangular co-ordinates:
                    position_transformed = self.transform_point(position)
                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze().tolist()
                    # Adding this to new attribute "position_transformed"
                    tracks[object][frame_num][track_id]["position_transformed"] = position_transformed

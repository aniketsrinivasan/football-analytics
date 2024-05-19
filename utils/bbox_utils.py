def get_bbox_center(bbox) -> tuple[int, int]:
    xm, ym, xM, yM = bbox
    # getting the center's coordinates:
    return int((xm+xM)/2), int((ym+yM)/2)


def get_bbox_width(bbox):
    # bbox[0] = xm, bbox[2] = xM
    return (bbox[2] - bbox[0])


def measure_distance_sqr(point1, point2):
    return (point1[0] - point2[0])**2 + (point1[1] - point2[1])**2
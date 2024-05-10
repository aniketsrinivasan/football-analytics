def get_bbox_center(bbox) -> tuple[int, int]:
    xm, ym, xM, yM = bbox
    # getting the center's coordinates:
    return int((xm+xM)/2), int((ym+yM)/2)


def get_bbox_width(bbox):
    # bbox[0] = xm, bbox[2] = xM
    return (bbox[2] - bbox[0])
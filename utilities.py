

def calc_intersection(bbox0, bbox1) -> float:
    # Calculate the boundaries of the intersections for
    # 2 norfair detection formatted bbox
    x0 = max(bbox0[0], bbox1[0])
    x1 = min(bbox0[2], bbox1[2])
    y0 = max(bbox0[1], bbox1[1])
    y1 = min(bbox0[3], bbox1[3])
    return max(x1 - x0, 0) * max(y1 - y0, 0)

def iou(bbox0, bbox1) -> float:
    union = ((bbox0[2]-bbox0[0]) * (bbox0[3] - bbox0[1])) + ((bbox1[2]-bbox1[0]) * (bbox1[3] - bbox1[1]))
    return calc_intersection(bbox0, bbox1) / union
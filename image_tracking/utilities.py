import numpy as np
from numpy.typing import NDArray

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


def calc_intersection2(bbox0, bbox1) -> float:
    # Calculate the boundaries of the intersections for
    # 2 YOLO formatted bbox
    x0 = max(bbox0[1], bbox1[1])
    x1 = min(bbox0[3], bbox1[3])
    y0 = max(bbox0[2], bbox1[2])
    y1 = min(bbox0[4], bbox1[4])
    # Return width * height
    return max(x1 - x0, 0) * max(y1 - y0, 0)

def iou2(bbox0, bbox1) -> float:
    """
    Calculates the IOU of two bounding boxes in YOLO format (in pixels)
    """
    # Area of intersection
    intersection = calc_intersection2(bbox0, bbox1)
    # Area of the union
    union = ((bbox0[3]-bbox0[1]) * (bbox0[4]-bbox0[2])) + ((bbox1[3]-bbox1[1]) * (bbox1[4]-bbox1[2])) - intersection
    return intersection / union


def combine_bbox(bbox0, bbox1):
    """
    Combines two bounding boxes. Draws the smallest rectangular bounding box that includes both source bboxes
    """
    x0 = min(bbox0[1], bbox1[1])
    x1 = max(bbox0[3], bbox1[3])
    y0 = min(bbox0[2], bbox1[2])
    y1 = max(bbox0[4], bbox1[4])

    return [x0,y0, x1, y1]


def filter_cyclists(detections: NDArray):
    """
    Combines cyclists with their bicycle
    """
    out_dets = []
    persons_dets = []
    bicycle_dets = []

    if 1 not in detections[:, 0].flatten():
        return detections

    # First pass filter out bicycle & persons
    for i in range(detections.shape[0]):
        if detections[i, 0] == 1 or detections[i, 0] == 3:
            bicycle_dets.append(detections[i])
        elif detections[i, 0] == 0:
            persons_dets.append(detections[i])
        else:
            out_dets.append((detections[i]))

    # Attempt to match every bicycle with a person
    for i in range(len(bicycle_dets)):
        match_idx = -1
        x_dist = (bicycle_dets[i][3] - bicycle_dets[i][1]) / 2.5
        y_dist = (bicycle_dets[i][4] - bicycle_dets[i][2]) / 4
        max_iou = 0.1
        for j in range(len(persons_dets)):
            # check if the mid-point of the person bbox is close to that of the bicycle & the bottom of the bbox is not
            # significantly lower than that of the bicycle & has a reasonably high iou
            x = abs((persons_dets[j][1] + persons_dets[j][3] - bicycle_dets[i][1] - bicycle_dets[i][3]) / 2)
            int_over_union = iou2(bicycle_dets[i], persons_dets[j])
            if x < x_dist and (bicycle_dets[i][4] - persons_dets[j][4]) > 0:
                if int_over_union > max_iou:
                    max_iou = int_over_union
                elif max_iou <= 0.1 and abs(bicycle_dets[i][2] - persons_dets[j][4]) > y_dist :
                    max_iou = max(int_over_union, 0.1)
                else:
                    continue
                x_dist = x
                match_idx = j


        # Check if we found a matching person
        if match_idx != -1:
            # Create a cyclist detection that covers both the person and the bicycle
            # averages the confidence of both
            cyclist_entry = [80] + combine_bbox(bicycle_dets[i], persons_dets[match_idx])
            cyclist_entry.append((bicycle_dets[i][5] +  persons_dets[match_idx][5])/2)
            out_dets.append(np.asarray(cyclist_entry))
            del persons_dets[match_idx]
            continue

        # If there isn't a match, add the current bicycle detection only
        out_dets.append(bicycle_dets[i])

    return np.asarray(out_dets + persons_dets)
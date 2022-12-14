import os, re
import numpy as np

from radar_utilities import load_yolo, load_mat, polar_in_cartesian, img_to_radar_cartesian
from convert import calc_conversion_grid

from numpy.typing import NDArray
from typing import List, Dict


V_FRAME_DIFF_MAX: int = 20
V_FRAME_DIFF_MIN: int = 5


def find_uid_idx(uid: int, uid_map: Dict) -> int:
    """
    Lookup a particular uid in a given uid_map

    :param uid: an integer uid
    :param uid_map: a map that maps uid's to their corresponding index in that frame
    :return: the index of the bounding box with the caller supplied uid, -1 if it's not found
    """
    if uid in uid_map:
        return uid_map[uid]

    return -1


def calc_velocity(frame0: NDArray, bbox0: int, frame1: NDArray, bbox1: int, output: NDArray, f_diff: int):
     diff = (frame1[bbox1, 1:3] - frame0[bbox0, 1:3]) / f_diff * 5
     output[bbox0, 0:2] = diff


def generate_idx_range(lower: int, upper: int, length: int):
    if lower > length - 1:
        return []

    upper = upper if upper < length else length - 1
    return range(upper, lower, -1)


def get_centroids(img_bboxes: List[NDArray], trans_mat:NDArray)->List[NDArray]:
    centroids: List[NDArray] = []
    for frame in img_bboxes:
        if len(frame) == 0:
            centroids.append(np.asarray([[]]))
            continue
        # Calculate the centroid's location
        radar_posn = img_to_radar_cartesian(frame, trans_mat)
        # Reshape frame in to 2D array if needed
        # if len(frame.shape) == 1:
        #     frame.reshape((1, frame.shape[0]))
        assert(len(frame.shape) == 2)
        assert(len(radar_posn.shape) == 2)
        # TODO: change radar_posn to be a map from UID to centroids for better perf
        centroids.append(np.concatenate((radar_posn, frame[:, [-1]]), axis=1))

    return centroids


def get_velocities(centroids: List[NDArray], uid_map:List[Dict])-> List[NDArray]:
    centroids_velocity:List[NDArray] = []
    # A map that holds the last calculated velocity for each
    last_v_by_uid = {}

    # Iterate through every frame
    for i in range(0, len(centroids)):
        if centroids[i].shape[1] == 0:
            centroids_velocity.append(np.asarray([[]]))
            continue

        # Generate numpy array for storing the velocity output, defaults velocity to be zero
        velocities = np.zeros((centroids[i].shape[0], 3))
        # Copy uid for all bboxes in the frame
        velocities[:, 2] = centroids[i][:, 3]
        # Go through every radar centroid for the ith frame
        for r in range(centroids[i].shape[0]):
            # Attempt to find a bbox with the same uid in the next few frames
            indices = generate_idx_range(i + V_FRAME_DIFF_MIN, i + V_FRAME_DIFF_MAX, len(centroids))
            idx = -1
            for f in indices:
                # Try to find a bbox in the fth frame that has the same uid
                idx = find_uid_idx(centroids[i][r, 3], uid_map[f])
                if idx != -1:
                    # Use the found bbox to calculate velocity
                    calc_velocity(centroids[i], r, centroids[f], idx, velocities, f - i)
                    last_v_by_uid[centroids[i][r, 3]] = velocities[r]
                    break

            if idx == -1:
                try:
                    val = last_v_by_uid[centroids[i][r,3]]
                    velocities[r, :] = val
                except KeyError:
                    pass


        centroids_velocity.append(velocities)

    return centroids_velocity


if __name__ == '__main__':
    src_path = "D:\\UWCR Data2\\"
    seq_name = "2019_04_30_mlms001"

    date_pattern = re.compile("\d{4}_\d{2}_\d{2}")
    date = date_pattern.search(seq_name).group()
    seq_path = os.path.join(src_path, date, seq_name)

    if not os.path.exists(src_path):
        print("Path does not exist...")
        exit(1)

    # Load image bounding boxes
    img_bboxes = load_yolo(os.path.join(seq_path, "images_0", "YOLO"))

    # Load transformation matrix
    trans_mat: NDArray = load_mat("transform.mat")

    # Calculate the conversion grid
    range_grid, angle_grid = calc_conversion_grid()
    # Pre-convert every polar coordinate in radar data into its cartesian counterpart
    x, y = polar_in_cartesian(range_grid, angle_grid)

    # Calculate object velocity based the displacement of their bounding box centroids
    centroids = get_centroids(img_bboxes, trans_mat)
    centroids_velocity = get_velocities(centroids)

    exit()




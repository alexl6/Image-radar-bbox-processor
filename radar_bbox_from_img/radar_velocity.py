import os, re
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

from img_to_radar import load_yolo, load_transform_mat, calc_conversion_grid, polar_in_cartesian, img_to_radar_cartesian


if __name__ == '__main__':
    src_path = "D:\\UWCR Data\\"
    seq_name = "2019_04_30_mlms001"

    date_pattern = re.compile("\d{4}_\d{2}_\d{2}")
    date = date_pattern.search(seq_name).group()
    seq_path = os.path.join(src_path, date, seq_name)

    if not os.path.exists(src_path):
        print("Path does not exist...")
        exit(1)

    # Load image bounding boxes
    img_bboxes = load_yolo(os.path.join(seq_path, "images_0", "YOLO"))

    # Load tranformation matrix
    trans_mat: NDArray = load_transform_mat("transform.mat")

    # Calculate the conversion grid
    range_grid, angle_grid = calc_conversion_grid()
    # Pre-convert every polar coordinate in radar data into its cartesian counterpart
    x, y = polar_in_cartesian(range_grid, angle_grid)

    radar_centroid = []
    for frame in img_bboxes:
        # Calculate the centroid's location
        radar_posn = img_to_radar_cartesian(frame, trans_mat)
        radar_centroid.append(np.concatenate((radar_posn, frame[:, [-1]]), axis=1))


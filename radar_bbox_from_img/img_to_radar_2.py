import re
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from convert import calc_conversion_grid
from radar_utilities import load_yolo, load_transform_mat, polar_in_cartesian, img_to_radar_cartesian, swap, dim_ratio
from radar_velocity import get_centroids, get_velocities

from typing import List
from numpy.typing import NDArray

MAX_RADAR_RADIUS = 30


def label_conv(label: int) -> List[float]:
    """
    Converts an object label into a tuple of estimated dimensions (x, y)

    :param label: The original label of the object
    :return: The estimated dimensions in (length, width). Both represented as floats
    """
    mapping = {0: [0.7, 0.7],
               1: [0.6, 1.7],
               2: [1.8, 4.5],
               3: [0.8, 2],
               5: [2.5, 12],
               7: [2.6, 15],
               80: [0.6, 1.7]
               }
    return mapping[label]


def get_radar_bg_intensity(f_num, seq_path):
    """
    Obtains the visual representation of a given radar frame

    :param f_num: frame number
    :param seq_path: path to the directory of radar detections
    :return: radar intensity in a flattented array
    """
    radar_dir = os.path.join(seq_path, "radar")

    frame_num = str(f_num).zfill(6) + ".npy"

    a = np.load(os.path.join(radar_dir, frame_num))
    b = a[:, :, 0] ** 2 + a[:, :, 1] ** 2

    return b.flatten()


def matched_filter_image(seq_path: str, centroids: List[NDArray], x, y, eval_frame=0):
    # Check if we have the same number of radar files (background image) and trajectory centroid entries
    num_radar_files: int = len(os.listdir(os.path.join(seq_path, "radar")))
    assert (num_radar_files == len(centroids))

    # Get the polar radar image
    base_image = get_radar_bg_intensity(eval_frame, seq_path)
    base_loc = centroids[eval_frame]
    images_multiply = []

    # for i in range(len(centroids)):
    #     images_xy =
    #     obj_loc = centroids[]


def batch_process_to_img(img_bboxes, seq_path, centroids, centroids_v, radar_label_path, x, y):
    # Process every frame in the sequence
    for f_num in range(len(img_bboxes)):
        if f_num % (len(img_bboxes) // 4) == 0:
            print("%i%% Done" % (100 * (f_num / len(img_bboxes))))
        z = get_radar_bg_intensity(f_num, seq_path)
        # Clear previous drawings
        plt.clf()
        # Draw radar background
        plt.scatter(x, y, c=z)
        # Draw every box in this frame
        for i in range(img_bboxes[f_num].shape[0]):
            # Skip bboxes outside of radar range
            if centroids[f_num][i, 2] > MAX_RADAR_RADIUS:
                continue

            # Get the bounding box dimensions from object type label
            # Assume object is 'vertical' by default
            radar_bbox_dim: List[float] = label_conv(img_bboxes[f_num][i, 0])

            # first check the velocity, prefer determining orientation by velocity
            if np.sum(np.abs(centroids_v[f_num][i, 0:2] ** 2)) > 0.08:
                # If there's greater horizontal movement
                if abs(centroids_v[f_num][i, 0]) / abs(centroids_v[f_num][i, 0]) > 1.1:
                    swap(radar_bbox_dim)
            # Check box shape if the object is reasonably far away
            elif centroids[f_num][i, 2] > 8 or (abs(centroids[f_num][i, 1]) < 1 and centroids[f_num][i, 2] > 5):
                # Use the bounding box aspect ratio to determine object orientation
                if dim_ratio(img_bboxes[f_num][i, :]) > 2:
                    swap(radar_bbox_dim)

            plt.gca().add_patch(patches.Rectangle(
                (centroids[f_num][i, 1] - radar_bbox_dim[0] / 2, centroids[f_num][i, 2] - radar_bbox_dim[1] / 2),
                radar_bbox_dim[0], radar_bbox_dim[1], fill=True, color='pink', alpha=0.35,
                zorder=100))
        # Use the same scaling for x,y-axis, configure bounds, display
        plt.axis('square')
        plt.xlim((-MAX_RADAR_RADIUS, MAX_RADAR_RADIUS))
        plt.ylim((0, MAX_RADAR_RADIUS))
        # plt.scatter(radar_posn[:,1], radar_posn[:, 2], c='red', alpha = 0.8)

        fig_name = str(f_num).zfill(10) + '.png'
        plt.savefig(os.path.join(radar_label_path, fig_name))


def batch_process_interactive(img_bboxes, seq_path, centroids, centroids_v, x, y):
    # Enable interactive mode
    plt.ion()
    while True:
        # Take user input/exit gracefully
        try:
            f_num = int(input("Frame num?\t"))
        except EOFError:
            print("Exiting...")
            exit()

        # Retry if frame number out of range
        if f_num < 0 or f_num >= len(img_bboxes):
            continue
        # Get radar background
        z = get_radar_bg_intensity(f_num, seq_path)

        # Clear previous drawings
        plt.clf()
        # Draw radar background
        plt.scatter(x, y, c=z)
        # Draw every box in this frame
        for i in range(img_bboxes[f_num].shape[0]):
            # Skip bboxes outside of radar range
            if centroids[f_num][i, 2] > MAX_RADAR_RADIUS:
                continue

            # Get the bounding box dimensions from object type label
            # Assume object is 'vertical' by default
            radar_bbox_dim: List[float] = label_conv(img_bboxes[f_num][i, 0])

            # first check the velocity, prefer determining orientation by velocity
            if np.sum(np.abs(centroids_v[f_num][i, 0:2] ** 2)) > 0.08:
                # If there's greater horizontal movement
                if abs(centroids_v[f_num][i, 0]) / abs(centroids_v[f_num][i, 0]) > 1.1:
                    swap(radar_bbox_dim)
            # Check box shape if the object is reasonably far away
            elif centroids[f_num][i, 2] > 8 or (abs(centroids[f_num][i, 1]) < 1 and centroids[f_num][i, 2] > 5):
                # Use the bounding box aspect ratio to determine object orientation
                if dim_ratio(img_bboxes[f_num][i, :]) > 2:
                    swap(radar_bbox_dim)

            plt.gca().add_patch(patches.Rectangle(
                (centroids[f_num][i, 1] - radar_bbox_dim[0] / 2, centroids[f_num][i, 2] - radar_bbox_dim[1] / 2),
                radar_bbox_dim[0], radar_bbox_dim[1], fill=True, color='pink', alpha=0.35,
                zorder=100))
        # Use the same scaling for x,y-axis, configure bounds, display
        plt.axis('square')
        plt.xlim((-MAX_RADAR_RADIUS, MAX_RADAR_RADIUS))
        plt.ylim((0, MAX_RADAR_RADIUS))
        # plt.scatter(radar_posn[:,1], radar_posn[:, 2], c='red', alpha = 0.8)
        plt.show()


if __name__ == '__main__':
    # Build path to image & radar data files
    src_path = "D:\\UWCR Data\\"
    seq_name = input("Seq name?\t")

    if seq_name == '':
        seq_name = "2019_04_09_pms1000"

    date_pattern = re.compile("\d{4}_\d{2}_\d{2}")
    date = date_pattern.search(seq_name).group()
    seq_path = os.path.join(src_path, date, seq_name)

    if not os.path.exists(src_path):
        print("Path does not exist...")
        exit(1)

    # Load YOLO detection for images
    print("Loading img bboxes")
    img_bboxes = load_yolo(os.path.join(seq_path, "images_0", "YOLO"))

    # Load camera ==> radar transformation matrix
    trans_mat: NDArray = load_transform_mat("transform.mat")
    # Calculate polar to cartesian conversion
    range_grid, angle_grid = calc_conversion_grid()
    x, y = polar_in_cartesian(range_grid, angle_grid)

    print("Calculating radar positions & velocities")
    # Calculate image bbox centroids in cartesian radar plane and their velocities
    centroids = get_centroids(img_bboxes, trans_mat)
    centroids_v = get_velocities(centroids)

    radar_label_path = os.path.join(seq_path, 'radar_label')
    if not os.path.exists(radar_label_path):
        os.makedirs(radar_label_path)

    matched_filter_image(seq_path, centroids, x, y)
    # batch_process_to_img(img_bboxes, seq_path, centroids, centroids_v, radar_label_path, x, y)
    # batch_process_interactive(img_bboxes, seq_path, centroids, centroids_v, x, y)

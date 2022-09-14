import re
import glob
import os
import csv
import scipy
from scipy import io
import numpy as np
from typing import List, Tuple
from numpy.typing import NDArray
from convert import calc_conversion_grid
# import PIL.Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

MAX_RADAR_RADIUS = 30

def label_conv(label: int) -> List[float]:
    """
    Converts an object label into a tuple of estimated dimensions (x, y)

    :param label: The original label of the object
    :return: The estimated dimensions in (length, width). Both represented as floats
    """
    mapping = {0: [0.7, 0.7],
               1: [1.7, 0.6],
               2: [4.5, 1.8],
               3: [2, 0.8],
               5: [12, 2.5],
               7: [15, 2.6],
               80: [1.7, 0.6]
               }
    return mapping[label]


def load_yolo(dir_path):
    """
    Load all yolo formatted detections from a given directory where every detection is in file named '[frame_num].txt'
    Scales dimensions to 1440x1080. Expected file is space separated where each row represents a detection formatted as:
    Object_class, center_X, center_Y, width, height, confidence


    :param dir_path: Directory to look for the detections
    :return: A list of lists of YOLO detections. Each inner list contains all detections for a given frame.
                Every detection is formatted [Object_class, center_X, center_Y, width, height, confidence, frame_num]
    """
    raw_bbox = []
    fname_pattern = re.compile("[0-9]+.txt$")
    # Get a list of plain text files in the given dir
    files = glob.glob("*.txt", root_dir=dir_path)
    # Load detections from each file
    for fname in files:
        if fname_pattern.match(fname) is None:  # skip non-YOLO files
            continue
        # Read & process each line
        with open(os.path.join(dir_path, fname), 'r') as f:
            entry = []
            reader = csv.reader(f, delimiter=' ')
            for line in reader:
                if len(line) == 0:
                    continue
                entry.append(line)
            raw_bbox.append(np.asarray(entry, dtype=float))

    # Scale bboxes to 1440x1080
    for frame_data in img_bboxes:
        if len(frame_data) > 0:
            frame_data *= np.array([1, 1440, 1080, 1440, 1080])

    return raw_bbox


# Ported from original code in MATLAB
def transform(H: NDArray, coordinates: NDArray) -> NDArray:
    """
    Transforms multiple sets of coordinates from the image plane into cartesian radar plane using matrix H

    :param H: Transformation matrix used to transform each set of coordinate
    :param coordinates: A numpy array of coordinates in YOLO format where each row is a single 2D point
    :return:
    """
    calculated_points = np.empty([coordinates.shape[0], 3])
    for j in range(coordinates.shape[0]):
        temp: NDArray = H @ coordinates[j:j + 1, :].conjugate().T
        temp = np.divide(temp, temp[2, :])
        calculated_points[j, :] = temp.T[0]

    return calculated_points


def load_transform_mat(path: str, var_name: str = 'HRadar0') -> NDArray:
    """
    Loads a transformation matrix from a MATLAB .mat file

    :param path: Path to the .mat file
    :param var_name: The variable name of the matrix
    :return: A numpy array representation of the transformation matrix
    """
    return scipy.io.loadmat(path)[var_name]


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


def polar_in_cartesian(range_grid, angle_grid, dim=(128, 128)):
    """
    Calculates the cartesian position of every point in a polar grid of dim(ension). Default dimension is (128 x 128)

    :param dim: Dimensions of the grid
    :return:
    """
    x_posn = np.empty(dim)
    y_posn = np.empty(dim)

    for i in range(dim[0]):
        for j in range(dim[1]):
            angle = angle_grid[j]
            rang = range_grid[i]
            x_posn[i, j] = np.sin(angle) * rang
            y_posn[i, j] = np.cos(angle) * rang

    return x_posn.flatten(), y_posn.flatten()


def img_to_radar_cartesian(img_bboxes: List[NDArray], H: NDArray) -> NDArray:
    """
    Converts the centroids of a list of image bboxes to their location on radar (in cartesian coordinate)

    :param img_bboxes: Image bounding boxes to process
    :param H: Conversion matrix
    :return: Centroids' location in cartesian radar plane
    """

    # Extract the images centroids to be a new [3 x n] numpy array
    print(img_bboxes)
    if len(img_bboxes) == 0:
        return np.asarray(img_bboxes)
    img_coordinates = img_bboxes[:, 1:4].copy()
    # Calculate the centroid
    img_coordinates[:, 1] += img_bboxes[:, 4] / 2
    img_coordinates[:, 2] = 1

    # Convert the coordinates to cartesian radar coordinates
    radar_centroids = transform(H, img_coordinates)

    # TODO: Assign bbox size manually for each radar centroids
    return np.concatenate((img_bboxes[:, 0:1], radar_centroids[:, 0:2]), axis=1)


def dim_ratio(bbox: NDArray) -> float:
    """
    Calculate the ratio between width and height of a bounding box. (w/h)

    :param bbox: Norfair bbox
    :return: Ratio in terms of w/h
    """
    return bbox[3] / bbox[4]


if __name__ == '__main__':
    src_path = "D:\\UWCR Data\\"
    seq_name = "2019_04_30_mlms001"

    date_pattern = re.compile("\d{4}_\d{2}_\d{2}")
    date = date_pattern.search(seq_name).group()
    seq_path = os.path.join(src_path, date, seq_name)

    if not os.path.exists(src_path):
        print("Path does not exist...")
        exit(1)

    img_bboxes = load_yolo(os.path.join(seq_path, "images_0", "YOLO"))


    trans_mat: NDArray = load_transform_mat("transform.mat")

    range_grid, angle_grid = calc_conversion_grid()
    x, y = polar_in_cartesian(range_grid, angle_grid)

    plt.ion()
    while True:
        # Take user input/exit gracefully
        try:
            f_num = int(input("Frame num?\t"))
        except EOFError:
            print("Exiting...")
            exit()
        # Clear previous drawings
        plt.clf()
        radar_posn = img_to_radar_cartesian(img_bboxes[f_num], trans_mat)

        z = get_radar_bg_intensity(f_num, seq_path)
        print(radar_posn)

        plt.scatter(x, y, c=z)
        for i in range(radar_posn.shape[0]):
            radar_bbox_dim: List[float] = label_conv(radar_posn[i, 0])
            if radar_posn[i, 0] != 0 and dim_ratio(img_bboxes[f_num][i, :]) < 1.8:
                temp: float = radar_bbox_dim[0]
                radar_bbox_dim[0] = radar_bbox_dim[1]
                radar_bbox_dim[1] = temp
            if radar_posn[i, 2] > MAX_RADAR_RADIUS:
                continue
            # fig.patches.extend([plt.Rectangle((0.25, 0.5), 0.25, 0.25, color='g', alpha=0.5, zorder=1000, transform=fig.transFigure, figure=fig)])
            plt.gca().add_patch(
                patches.Rectangle((radar_posn[i, 1] - radar_bbox_dim[0] / 2, radar_posn[i, 2] - radar_bbox_dim[1] / 2),
                                  radar_bbox_dim[0], radar_bbox_dim[1], fill=True, color='pink', alpha=0.35,
                                  zorder=100))

        plt.axis('square')
        plt.xlim((-MAX_RADAR_RADIUS, MAX_RADAR_RADIUS))
        plt.ylim((0, MAX_RADAR_RADIUS))
        # plt.scatter(radar_posn[:,1], radar_posn[:, 2], c='red', alpha = 0.8)
        plt.show()

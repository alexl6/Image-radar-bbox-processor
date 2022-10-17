import csv, os, re, glob

import numpy as np
import scipy.io

from numpy.typing import NDArray
from typing import List, Dict


def load_yolo(dir_path: str, scale:NDArray = np.array([1, 1440, 1080, 1440, 1080, 1]))-> (List[NDArray], List[Dict]):
    """
    Load all yolo formatted detections from a given directory where every detection is in file named '[frame_num].txt'
    Scales dimensions to 1440x1080. Expected file is space separated where each row represents a detection formatted as:
    Object_class, center_X, center_Y, width, height, UID

    :param dir_path: Directory to look for the detections
    :return: A list of 2D numpy arrays of YOLO detections. Each 2D array represents the detections in that given frame.
                Every row is a bounding box formatted as [Object_class, center_X, center_Y, width, height]
             Also returns a list of dictionaries (same length as the YOLO detections list). Each dict maps the UID of
                a detection to its row index in the 2D array for that given frame.
    """
    det_bbox = []
    uid_map = []
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
            mapping = {}
            reader = csv.reader(f, delimiter=' ')
            idx = 0 # Can't use enumerate here because it has problem with reading extra empty lines
            for line in reader:
                if len(line) == 0:
                    continue
                entry.append(line)
                mapping[line[-1]] = idx
                idx +=1
            det_bbox.append(np.asarray(entry, dtype=float))
            uid_map.append(mapping)

    # Scale bboxes (1440x1080 by default)
    for frame_data in det_bbox:
        if len(frame_data) > 0:
            frame_data *= scale

    return det_bbox, uid_map


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


def img_to_radar_cartesian(img_bboxes: NDArray, H: NDArray) -> NDArray:
    """
    Converts the centroids of image bboxes to their location on radar (in cartesian coordinate)

    :param img_bboxes: Image bounding boxes to process
    :param H: Conversion matrix
    :return: Centroids' location in cartesian radar plane
    """

    # Extract the images centroids to be a new [3 x n] numpy array
    if len(img_bboxes) == 0:
        return np.asarray(img_bboxes)
    img_coordinates = img_bboxes[:, 1:4].copy()
    # Calculate the centroid
    img_coordinates[:, 1] += img_bboxes[:, 4] / 2
    img_coordinates[:, 2] = 1

    # Convert the coordinates to cartesian radar coordinates
    radar_centroids = transform(H, img_coordinates)

    # TODO: Assign bbox size manually for each radar centroids
    ret_val: NDArray = np.concatenate((img_bboxes[:, 0:1], radar_centroids[:, 0:2]), axis=1)
    return ret_val
    # if len(ret_val.shape) == 2:
    #     return ret_val
    # return ret_val.reshape((1, len(ret_val)))


def swap(container: NDArray, idx0: int = 0, idx1: int = 1) -> None:
    """
    Swaps two elements in a list. Both idx0 and idx 1 should be > 0 and < len(container)

    :param container: List containing two elements
    :param idx0: Index to the first element
    :param idx1: Index to the second element
    """
    temp = container[idx0]
    container[idx0] = container[idx1]
    container[idx1] = temp


def dim_ratio(bbox: NDArray) -> float:
    """
    Calculate the ratio between width and height of a bounding box. (w/h)

    :param bbox: Norfair bbox
    :return: Ratio in terms of w/h
    """
    return bbox[3] / bbox[4]
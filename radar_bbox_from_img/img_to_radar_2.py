import csv
import re
import os
import math

import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from convert import calc_conversion_grid
from radar_utilities import load_yolo, load_mat, polar_in_cartesian, img_to_radar_cartesian, swap, dim_ratio
from radar_velocity import get_centroids, get_velocities, find_uid_idx

from typing import List, Dict, Tuple
from numpy.typing import NDArray

MAX_RADAR_RADIUS = 30
GRID_DIM_X = 280
GRID_DIM_Y = 140
GRID_RES = 0.2


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


def get_radar_intensity(f_num: int, seq_path: str):
    """
    Obtains the visual representation of a given radar frame

    :param f_num: frame number
    :param seq_path: path to the directory of radar detections
    :return: radar intensity in a flattened array
    """
    radar_dir = os.path.join(seq_path, "radar")

    frame_num = str(f_num).zfill(6) + ".npy"

    a = np.load(os.path.join(radar_dir, frame_num))
    b = a[:, :, 0] ** 2 + a[:, :, 1] ** 2

    return b.flatten()



def fnum_prompt(length: int):
    """
    Prompt the user for a frame number or exit gracefully if the user types EOF

    :param length: The length of the sequence (aka. maximum accepted frame number + 1)
    :return: parsed frame number
    """
    while True:
        # Take user input/exit gracefully
        try:
            f_num = int(input("Frame num?\t"))
        except EOFError:
            print("Exiting...")
            exit()
        # Retry if frame number out of range
        if f_num >= 0 or f_num < length:
            return f_num


def probe_frames_by_uid(uid: int, uid_map: List[map], curr: int, num: int, step: int = 1, backwards: bool = False):
    """
    Probe future or past frames to see if it contains a bounding box with the caller specified uid.
    Starts from the frame that's `num` frames away from the current frame until we find a match.

    :param uid: uid to search
    :param uid_map: List of maps containing a mapping from uid to the bounding box's index
    :param curr: current frame
    :param num: specifies the distance of the starting frame to the current frame
    :param step: size of the steps
    :param backwards: Probes past frames if True. Default is False (probe future frames).
    :return: The furthest frame (<=`num` frames away from the curr) containing a matching uid.
    """
    if backwards:
        for i in range(max(0, curr - num * step), curr, step):
            if uid in uid_map[i]:
                return i
    else:
        for i in range(min(len(uid_map) - 1, curr + 1 + num * step), curr + 1, -step):
            if uid in uid_map[i]:
                return i

    return curr


def matched_filter_image(seq_path: str, centroids: List[NDArray], x, y, uid_map: List[Dict], base_frame: int = 100,
                         uid='0'):
    """
    Runs match filter for all bounding boxes in a given frame
    :param seq_path:
    :param centroids:
    :param x:
    :param y:
    :param uid_map:
    :param base_frame:
    :param uid:
    :return:
    """
    # Matched filter parameters
    num_frames = 12  # Number of frames to use
    step_size = 2  # Spacing between frames

    # Check if we have the same number of radar files (background image) and trajectory centroid entries
    num_radar_files: int = len(os.listdir(os.path.join(seq_path, "radar")))
    assert (num_radar_files == len(centroids))

    # Get base frame's radar image in cartesian plane
    z = get_radar_intensity(base_frame, seq_path)
    base_image = scale_to_grid(x, y, z)

    # Determine the base location used to calculate position offset for other frames
    base_loc = centroids[base_frame][uid_map[base_frame][uid]][1:3]
    # print("Base centroids location: %s" % base_loc)

    # Try to find future frames containing matching uid
    future_frames = range(base_frame + step_size,
                          probe_frames_by_uid(uid, uid_map, base_frame, num_frames, step=step_size), step_size)
    # Find past frames with matching uid if there isn't enough future frames
    past_frames = range(
        probe_frames_by_uid(uid, uid_map, base_frame, num_frames - len(future_frames), backwards=True, step=step_size),
        base_frame, step_size)

    # Combine both future and past frames
    frames = list(past_frames) + list(future_frames)
    # print("Frames used: %s" % frames)

    # Predetermine images multiply dimensions based on num of frames available
    images_multiply = np.empty((len(frames) * GRID_DIM_X * GRID_DIM_Y)).reshape((len(frames), GRID_DIM_X, GRID_DIM_Y))
    assert (len(images_multiply.shape) == 3)

    accumulation_buffer = np.zeros((GRID_DIM_X, GRID_DIM_Y))  # Buffer for the sum of frames after multiplication
    img_to_multiply = base_image  # buffer for the last pair of neighboring images
    neighbor_sum_buffer = base_image  # buffer for storing the sum of neighboring images

    # Run matched filter on selected frames. Every pair of neighboring frames are added, then multiplied to its neighbor
    # before added to the accumulation_buffer. For example:  We perform element-wise operation
    # (frame #1 + frame #2) * (frame #2 + frame #3), then add this result to the accumulation buffer
    for i, f in enumerate(frames):
        # Convert image to cartesian plane
        z = get_radar_intensity(f, seq_path)
        image = scale_to_grid(x, y, z)

        # Shift the image by its bounding box offset from the base_frame
        try:
            obj_loc = centroids[f][uid_map[f][uid]][1:3]
        except KeyError:
            continue
        loc_diff = (base_loc - obj_loc) / GRID_RES
        image = scipy.ndimage.shift(image, shift=loc_diff, mode='constant')

        # TODO: Debug printouts
        # print("Obj location: %s"%obj_loc)
        # print("Scaled location diff: %s"%loc_diff)

        # Fuse with neighbor_img & copy to img_to_multiply for every other img
        if i % 2 == 0:
            # Add the current frame to neighbor sum buffer
            neighbor_sum_buffer += image
            if i != 0:  # Multiply with the last pair's sum, then add this result to accumulation buffer
                accumulation_buffer += neighbor_sum_buffer * img_to_multiply
        else:
            # Save last pair of neighbor's sum
            img_to_multiply = neighbor_sum_buffer
            # Reset the neighbor sum buffer to be the current frame
            neighbor_sum_buffer = image

    return accumulation_buffer


def scale_to_grid(x: NDArray, y: NDArray, z: NDArray):
    """
    Scales a 2D matrix of intensity values to a fixed size grid. Does not interpolate.
    Depends on global GRID dimensions.

    :param x: array of x coordinates
    :param y: array of y coordinates
    :param z: array of intensity at location (x,y) in the original grid
    :return: Scaled 2D matrix of intensities
    """
    grid = np.zeros((GRID_DIM_X, GRID_DIM_Y))
    grid_count = np.ones((GRID_DIM_X, GRID_DIM_Y))
    for i in range(len(z)):
        if grid[x[i], y[i]] != 0:
            grid_count[x[i], y[i]] += 1
        grid[x[i], y[i]] += z[i]

    # Divide each virtual `pixel` by the number of physical `pixel` behind it
    return grid / grid_count


def map_to_grid(x: NDArray, y: NDArray, precision: float = 0.25) -> Tuple[NDArray, NDArray]:
    scaled_x = x // precision
    scaled_y = y // precision
    scaled_x += int(GRID_DIM_X / 2)

    return scaled_x.astype(int), scaled_y.astype(int)


def batch_process_to_img(img_bboxes, seq_path, centroids, centroids_v, radar_label_path, x, y):
    # Process every frame in the sequence
    for f_num in range(len(img_bboxes)):
        if f_num % (len(img_bboxes) // 4) == 0:
            print("%i%% Done" % (100 * (f_num / len(img_bboxes))))
        z = get_radar_intensity(f_num, seq_path)
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


def matched_filter_interactive(img_bboxes:List[NDArray], seq_path:str, centroids:List[NDArray], centroids_v:List[NDArray],
                               x: NDArray, y:NDArray, uid_mapping: List[Dict])-> None:
    """
    Runs matched filter on a given sequence of data. Behaves differently from `matched_filter_batch()` since it does
    not use temporal data. Interactively presents converted radar bbox visualization on a frame by frame basis.

    :param img_bboxes: A list of image bounding boxes in YOLO
    :param seq_path: Path to the data sequence containing radar data
    :param centroids: A list of centroids by frame
    :param centroids_v: A list of centroid velocities by frame
    :param x: Array of possible x positions in radar data
    :param y: Array of possible y positions in radar data
    :param uid_mapping: A list of uid to index mapping for each frame.
    :return: Nothing
    """
    # Map x,y to a grid
    scaled_x, scaled_y = map_to_grid(x, y, GRID_RES)

    # Enable interactive mode
    plt.ion()

    while True:
        # Prompt use for frame number
        f_num = fnum_prompt(len(img_bboxes))

        # Clear previous drawings
        plt.clf()

        uids = list(uid_mapping[f_num].keys())

        # Display the original background
        z = get_radar_intensity(f_num, seq_path)
        image = scale_to_grid(scaled_x, scaled_y, z).T
        plt.ylim((0, GRID_DIM_Y))
        plt.imshow(image)

        # Run matched filter on every bounding box in the frame
        for uid in uids:
            # print("Match filter on type %s"%img_bboxes[f_num][uid_mapping[f_num][uid], 0])
            # Calculate matched filter image
            matched_img = matched_filter_image(seq_path, centroids, scaled_x, scaled_y, uid_mapping, f_num, uid)
            # Display the matched filter image as background
            # plt.imshow(matched_img.T, vmax=0.6)

            # Obtain the index of the selected bounding box
            bbox_idx = uid_mapping[f_num][uid]
            # Skip bboxes outside of radar range
            if centroids[f_num][bbox_idx, 2] > MAX_RADAR_RADIUS:
                continue

            # Get the bounding box dimensions from object type label
            # Assume object is 'vertical' by default
            radar_bbox_dim: NDArray = np.asarray(label_conv(img_bboxes[f_num][bbox_idx, 0]))

            # first check the velocity, prefer determining orientation by velocity
            if np.sum(np.abs(centroids_v[f_num][bbox_idx, 0:2] ** 2)) > 0.08:
                # If there's greater horizontal movement
                if abs(centroids_v[f_num][bbox_idx, 0]) / abs(centroids_v[f_num][bbox_idx, 0]) > 1.1:
                    swap(radar_bbox_dim)
            # Check box shape if the object is reasonably far away
            elif centroids[f_num][bbox_idx, 2] > 8 or (abs(centroids[f_num][bbox_idx, 1]) < 1 and centroids[f_num][bbox_idx, 2] > 5):
                # Use the bounding box aspect ratio to determine object orientation
                if dim_ratio(img_bboxes[f_num][bbox_idx, :]) > 2.3:
                    swap(radar_bbox_dim)

            # Pre-calculate scaled centroid & radar dim
            scaled_centroid = centroids[f_num][bbox_idx, 1:3] / GRID_RES
            scaled_radar_dim = radar_bbox_dim / GRID_RES

            # Shifts bounding box in the polar coordinate system to maximize the area sum
            shifted_centroids, _ = maximize_area_sum(scaled_centroid, scaled_radar_dim, matched_img)

            plt.gca().add_patch(patches.Rectangle(
                (scaled_centroid[0] - scaled_radar_dim[0] / 2 + GRID_DIM_X / 2,
                 scaled_centroid[1] - scaled_radar_dim[1] / 2),
                scaled_radar_dim[0], scaled_radar_dim[1], fill=True, color='green', alpha=0.25,
                zorder=100))

            plt.gca().add_patch(patches.Rectangle(
                (shifted_centroids[0] - scaled_radar_dim[0] / 2 + GRID_DIM_X / 2,
                 shifted_centroids[1] - scaled_radar_dim[1] / 2),
                scaled_radar_dim[0], scaled_radar_dim[1], fill=True, color='pink', alpha=0.3,
                zorder=100))


        # Use the same scaling for x,y-axis, configure bounds, display
        # plt.axis('square')
        # plt.xlim((-MAX_RADAR_RADIUS, MAX_RADAR_RADIUS))
        # plt.ylim((0, MAX_RADAR_RADIUS))
        # plt.scatter(radar_posn[:,1], radar_posn[:, 2], c='red', alpha = 0.8)
        # Debug plot: Show a dot for the radar's position
        plt.scatter(GRID_DIM_X/2, 0, color='blue')
        plt.show()


def maximize_area_sum(scaled_centroid: NDArray, scaled_radar_dim: NDArray, matched_img: NDArray, step_size: int = 2, init_shift = 0):
    # Experimental: Shifts bounding box in the polar coordinate system to maximize the area sum
    angle, dist = to_polar(scaled_centroid)

    # num_steps = 9 + max(0, math.ceil(dist - 3/GRID_RES) * 20)
    num_steps = 10

    if dist > 5/GRID_RES:
        num_steps *= 4
        step_size *= 2
    # Calculate area sum without shifting the bounding box
    prev_area_sum = radar_area_sum(scaled_centroid, scaled_radar_dim, matched_img)
    # print("Original: %3f"%prev_area_sum)

    # start with the initial shift amount
    dist += init_shift

    # Try to move the bounding box further/closer to the radar
    # Change distance by 0.2 m increment
    further_area_sum = radar_area_sum(to_cartesian(angle, dist + step_size), scaled_radar_dim, matched_img)
    closer_area_sum = radar_area_sum(to_cartesian(angle, dist - step_size), scaled_radar_dim, matched_img)

    # Check if the area sum is already maximized (locally, along the same fixed radar angle)
    if further_area_sum / prev_area_sum < 1 and closer_area_sum / prev_area_sum < 1:
        return scaled_centroid

    # Decide whether to move further away/closer
    if further_area_sum < closer_area_sum:
        step_size *= -1
        prev_area_sum = closer_area_sum
        # print("closer")
    else:
        prev_area_sum = further_area_sum
        # print("further")

    prev_centroid = to_cartesian(angle, dist + step_size)


    # Shifts until the area sum has peaked
    for i in range(2, num_steps):
        curr_centroid = to_cartesian(angle, dist + step_size * i)
        curr_area_sum = radar_area_sum(curr_centroid, scaled_radar_dim, matched_img)
        if curr_area_sum < prev_area_sum:
            # print("Shifted: %3f" % prev_area_sum)
            return prev_centroid, step_size * (i-1) + init_shift

        prev_area_sum = curr_area_sum
        prev_centroid = curr_centroid

    # print("Shifted: %3f" % curr_area_sum)
    return curr_centroid, step_size * (i-1) + init_shift



def radar_area_sum(scaled_centroid: NDArray, scaled_radar_dim: NDArray, matched_img)->float:
    """
    Calculate the radar intensity sum within the area bounded by a bbox of size `scaled_radar_dim`
    centered at `scaled_centroid`

    :param scaled_centroid:
    :param scaled_radar_dim:
    :param matched_img: An array representation of the image after running matched filter
    :return: Radar intensity sum within the bounded area.
    """
    lower_bounds = scaled_centroid - scaled_radar_dim / 2
    lower_bounds[0] += GRID_DIM_X / 2

    upper_bounds = lower_bounds + scaled_radar_dim

    lower_bounds = lower_bounds.astype(int)
    upper_bounds = upper_bounds.astype(int)

    sum = np.sum(matched_img[
                 np.clip(lower_bounds[0], 0, GRID_DIM_X):np.clip(upper_bounds[0], 0, GRID_DIM_X),
                 np.clip(lower_bounds[1], 0, GRID_DIM_X):np.clip(upper_bounds[1], 0, GRID_DIM_X)])
    # DEBUG points
    # print("Sum %3f"%sum)
    # plt.scatter(np.clip(upper_bounds[0], 0, GRID_DIM_X), np.clip(upper_bounds[1], 0, GRID_DIM_Y), color='pink')
    # plt.scatter(np.clip(lower_bounds[0], 0, GRID_DIM_X), np.clip(lower_bounds[1], 0, GRID_DIM_Y), color='pink')

    return sum


def to_polar(scaled_centroid: NDArray):
    """
    Converts a cartesian coordinate into polar coordinate system

    :param scaled_centroid: Array of x,y coordiantes
    :return: angle in radians, dist
    """

    angle = math.atan(scaled_centroid[0] / scaled_centroid[1])
    dist = math.sqrt(np.sum(scaled_centroid ** 2))

    # print("Angle: %5f\t Dist: %5f" % (angle, dist))
    return angle, dist


def to_cartesian(angle:float, dist:float)->NDArray:
    """
    Converts a polar coordinate into cartesian coordinate system.

    :param angle: Angle in radians
    :param dist: Distance
    :return: x, y coordinates in an array
    """
    x = math.sin(angle) * dist
    y = math.cos(angle) * dist

    # plt.scatter(x, y, color='red')
    return np.array([x, y])


def raw_bbox_interactive(img_bboxes:List[NDArray], seq_path:str, centroids:List[NDArray],
                         centroids_v:List[NDArray], x:NDArray, y:NDArray, in_cartesian: bool = False)->None:
    """
    Runs an interactive version of the processor without matched filter

    :param img_bboxes: A list of image bounding boxes in YOLO
    :param seq_path: Path to the data sequence containing radar data
    :param centroids: A list of centroids by frame
    :param centroids_v: A list of centroid velocities by frame
    :param x: Array of possible x positions in radar data
    :param y: Array of possible y positions in radar data
    :param in_cartesian: If true, scales result to cartesian coordinates, otherwise presents in the original polar coordinate system.
    :return: Nothing
    """

    # Map x,y to a grid
    scaled_x, scaled_y = map_to_grid(x, y, GRID_RES)

    # Enable interactive mode
    plt.ion()
    while True:
        # Prompt use for frame number
        f_num = fnum_prompt(len(img_bboxes))

        # Clear previous drawings
        plt.clf()
        if in_cartesian:  # Convert radar background to cartesian before drawing
            # Get radar background & scale to cartesian grid
            z = get_radar_intensity(f_num, seq_path)
            image = scale_to_grid(scaled_x, scaled_y, z).T
            plt.imshow(image)
            plt.ylim((0, GRID_DIM_Y))
        else:  # Draw raw radar background
            z = get_radar_intensity(f_num, seq_path)
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

            print("%i, %i" % (centroids[f_num][i, 1], centroids[f_num][i, 2]))

            if in_cartesian:
                plt.gca().add_patch(patches.Rectangle(
                    ((centroids[f_num][i, 1] - radar_bbox_dim[0] / 2) / GRID_RES + GRID_DIM_X / 2,
                     (centroids[f_num][i, 2] - radar_bbox_dim[1] / 2) / GRID_RES),
                    radar_bbox_dim[0] / GRID_RES, radar_bbox_dim[1] / GRID_RES, fill=True, color='pink', alpha=0.25,
                    zorder=100))
            else:
                plt.gca().add_patch(patches.Rectangle(
                    (centroids[f_num][i, 1] - radar_bbox_dim[0] / 2, centroids[f_num][i, 2] - radar_bbox_dim[1] / 2),
                    radar_bbox_dim[0], radar_bbox_dim[1], fill=True, color='pink', alpha=0.35,
                    zorder=100))
        if not in_cartesian:
            # Use the same scaling for x,y-axis, configure bounds, display
            plt.axis('square')
            plt.xlim((-MAX_RADAR_RADIUS, MAX_RADAR_RADIUS))
            plt.ylim((0, MAX_RADAR_RADIUS))

        plt.show()


def matched_filter_batch(img_bboxes: List[NDArray], seq_path:str, label_path:str, centroids:List[NDArray], centroids_v:List[NDArray], x:NDArray, y:NDArray, uid_mapping:List[Dict])->None:
    """
    Performs matched filter on a given data sequence. Optionally output text/visualized radar bbox data.

    :param img_bboxes: A list of image bounding boxes in YOLO
    :param seq_path: Path to the data sequence containing radar data
    :param label_path: Visual & text label output path, should be base directory for the given sequence
    :param centroids: A list of centroids by frame
    :param centroids_v: A list of centroid velocities by frame
    :param x: Array of possible x positions in radar data
    :param y: Array of possible y positions in radar data
    :param uid_mapping: A list of uid to index mapping for each frame.
    :return: None
    """
    # Map x,y to a grid
    scaled_x, scaled_y = map_to_grid(x, y, GRID_RES)

    cached_shift_amt = {}


    img_label_path = path_generator(seq_path, ['vis_labels'])
    text_label_path = path_generator(seq_path, ['vis_labels', 'text_label'])

    # Enable interactive mode
    plt.ion()
    for f_num in range(len(img_bboxes)):
        file = open(os.path.join(text_label_path, str(f_num).zfill(10) + '.csv'), mode='w', newline='')
        writer = csv.writer(file)
        if f_num % 10 == 0:
            print("%i\t/\t%i"%(f_num, len(img_bboxes)))

        # Clear previous drawings
        plt.clf()

        uids = list(uid_mapping[f_num].keys())

        # Display the original background
        z = get_radar_intensity(f_num, seq_path)
        image = scale_to_grid(scaled_x, scaled_y, z).T

        if False:
            plt.ylim((0, GRID_DIM_Y))
            plt.imshow(image)

        # Run matched filter on every bounding box in the frame
        for uid in uids:
            # Calculate matched filter image
            matched_img = matched_filter_image(seq_path, centroids, scaled_x, scaled_y, uid_mapping, f_num, uid)

            # Obtain the index of the selected bounding box
            bbox_idx = uid_mapping[f_num][uid]
            # Skip bboxes outside of radar range
            if centroids[f_num][bbox_idx, 2] > MAX_RADAR_RADIUS:
                continue

            # Get the bounding box dimensions from object type label
            # Assume object is 'vertical' by default
            radar_bbox_dim: NDArray = np.asarray(label_conv(img_bboxes[f_num][bbox_idx, 0]))

            # first check the velocity, prefer determining orientation by velocity
            if np.sum(np.abs(centroids_v[f_num][bbox_idx, 0:2] ** 2)) > 0.08:
                # If there's greater horizontal movement
                if abs(centroids_v[f_num][bbox_idx, 0]) / abs(centroids_v[f_num][bbox_idx, 0]) > 1.1:
                    swap(radar_bbox_dim)
            # Check box shape if the object is reasonably far away
            elif centroids[f_num][bbox_idx, 2] > 8 or (abs(centroids[f_num][bbox_idx, 1]) < 1 and centroids[f_num][bbox_idx, 2] > 5):
                # Use the bounding box aspect ratio to determine object orientation
                if dim_ratio(img_bboxes[f_num][bbox_idx, :]) > 2.3:
                    swap(radar_bbox_dim)

            # Pre-calculate scaled centroid & radar dim
            scaled_centroid = centroids[f_num][bbox_idx, 1:3] / GRID_RES
            scaled_radar_dim = radar_bbox_dim / GRID_RES

            shift_amt = cached_shift_amt.get(uid, 0)

            # Shifts bounding box in the polar coordinate system to maximize the area sum
            shifted_centroids, cached_shift_amt[uid] = maximize_area_sum(scaled_centroid, scaled_radar_dim, matched_img, shift_amt)

            if False:
                plt.gca().add_patch(patches.Rectangle(
                    (shifted_centroids[0] - scaled_radar_dim[0] / 2 + GRID_DIM_X / 2,
                     shifted_centroids[1] - scaled_radar_dim[1] / 2),
                    scaled_radar_dim[0], scaled_radar_dim[1], fill=True, color='pink', alpha=0.3,
                    zorder=100))

            restored_centroid = np.round(shifted_centroids * GRID_RES, 5)
            line = [uid, int(img_bboxes[f_num][bbox_idx, 0])]
            line += list(restored_centroid) + list(radar_bbox_dim)
            writer.writerow(line)
        file.close()

        if False:
            plt.scatter(GRID_DIM_X/2, 0, color='blue')
            plt.savefig(os.path.join(img_label_path, str(f_num).zfill(10) + '.png'), format='png')

def path_generator(root: str, sub: List[str])->str:
    p = os.path.join([root] + sub)
    if not os.path.exists(p):
        os.makedirs(p)
    return p

if __name__ == '__main__':
    # Build path to image & radar data files
    src_path = "D:\\UWCR Data\\"
    seq_name = input("Seq name?\t")

    if seq_name == '':
        seq_name = "2019_04_30_mlms001"

    date_pattern = re.compile("\d{4}_\d{2}_\d{2}")
    date = date_pattern.search(seq_name).group()
    seq_path = os.path.join(src_path, date, seq_name)

    if not os.path.exists(src_path):
        print("Path does not exist...")
        exit(1)

    # Load YOLO detection for images
    print("Loading img bboxes")
    img_bboxes, uid_mapping = load_yolo(os.path.join(seq_path, "images_0", "YOLO"))

    # Load camera ==> radar transformation matrix
    trans_mat: NDArray = load_mat("transform.mat")
    # Calculate polar to cartesian conversion
    range_grid, angle_grid = calc_conversion_grid()
    x, y = polar_in_cartesian(range_grid, angle_grid)

    print("Calculating radar positions & velocities")
    # Calculate image bbox centroids in cartesian radar plane and their velocities
    centroids = get_centroids(img_bboxes, trans_mat)
    centroids_v = get_velocities(centroids, uid_mapping)

    radar_label_path = os.path.join(seq_path, 'radar_label')
    vis_label_path = os.path.join(seq_path, 'vis_labels')
    if not os.path.exists(radar_label_path):
        os.makedirs(radar_label_path)
    if not os.path.exists(vis_label_path):
        os.makedirs(vis_label_path)


    # raw_bbox_interactive(img_bboxes, seq_path, centroids, centroids_v, x, y)
    matched_filter_interactive(img_bboxes, seq_path, centroids, centroids_v, x, y, uid_mapping)
    # matched_filter(img_bboxes, seq_path, vis_label_path, centroids, centroids_v, x, y, uid_mapping)

    # batch_process_to_img(img_bboxes, seq_path, centroids, centroids_v, radar_label_path, x, y)
    # plt.ion()
    # while True:
    #     f_num = int(input("F num?"))
    #     matched_filter_image(seq_path, centroids, scaled_x, scaled_y, uid_mapping, f_num)

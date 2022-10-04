import numpy as np
from norfair import Detection, Tracker
from utilities import iou, iou2, filter_cyclists
from tracklet import tracklet

from numpy.typing import NDArray
from typing import List
# Regex
import re

# File I/O
import glob
import os
import csv
import shutil

label_map = {0: '0',
               1: '0',
               2: '2',
               3: '0',
               5: '2',
               7: '2',
               80: '0',
               }

def label_conv(label: int) -> str:
    """
    Converts an object label into category labels (e.g. grouping cars, buses, and trucks into the same group)

    :param label: The original label of the object
    :return: The category label of the object
    """
    """
        mapping = {0: '0',
               1: '0',
               2: '1',
               3: '0',
               5: '2',
               7: '2',
               80: '0',
               }
    """
    try:
        return label_map[label]
    except KeyError:
        return '4'


def yolo2norfair(bbox: NDArray, dims=[1440, 1080]) -> Detection:
    """
    Converts a single norfair bbox to a norfair detection object

    :param bbox: A norfair bounding box (top-left & bottom-right)
    :param dims: Dimensions of the image in px (Optional, Defaults to [1440, 1080])
    :return: Constructed norfair detection object
    """
    new_bbox = bbox[1:5].reshape((2, 2)) * np.asarray(dims)
    score = np.asarray([bbox[5], bbox[5]])
    # label = str(int(bbox[0]))
    return Detection(new_bbox, score, data=int(bbox[0]), label=label_conv(int(bbox[0])))


def yolo2norfair_multi(bboxes: List[NDArray], dims=[1440, 1080]):
    """
    Convert a list of YOLO detections into a list of norfair detections

    :param bboxes: List of YOLO detection bounding boxes
    :param dims: Dimensions of the image in px (Optional, Default sot [1440, 1080])
    :return: List of norfair detections. Empty list if there are no input bboxes
    """
    norfair_dets = []
    bboxes = np.array(bboxes, dtype=float)
    # Handle scenarios when there's only a single bbox
    if len(bboxes) == 0:
        return norfair_dets
    elif len(bboxes.shape) == 1:
        bboxes = bboxes.reshape((1, bboxes.shape[0]))
    # Convert to norfair format
    bboxes[:, 1] -= bboxes[:, 3] / 2
    bboxes[:, 2] -= bboxes[:, 4] / 2
    bboxes[:, 3] += bboxes[:, 1]
    bboxes[:, 4] += bboxes[:, 2]
    # Change confidence to be between 0 and 1
    bboxes[:, 5] /= 100

    # TODO: Get rid of duplicates
    l1 = bboxes.shape[0]
    bboxes = filter_by_dim(bboxes)
    bboxes = remove_duplitcates(bboxes)
    bboxes = filter_cyclists(bboxes)

    for bbox in bboxes:
        if bbox[0] in label_map and (bbox[3] - bbox[1]) / (bbox[4] - bbox[2]) <= 5:  # Eliminate ultra-wide boxes
            norfair_dets.append(yolo2norfair(bbox, dims))
    return norfair_dets


def filter_by_dim(bboxes: NDArray) -> NDArray:
    keep_idx = np.ones(bboxes.shape[0])
    for i in range(bboxes.shape[0]):
        ratio = (bboxes[i, 3] - bboxes[i, 1]) / (bboxes[i, 4] - bboxes[i, 2])
        if ratio > 6 or (bboxes[i, 0] == 2 and (ratio > 3.5 or ratio < 0.45)):
            # print(ratio)
            keep_idx[i] = 0
    return bboxes[keep_idx.astype(bool), :]


def remove_duplitcates(bboxes: NDArray) -> NDArray:
    keep_idx = np.ones(bboxes.shape[0])
    for i in range(bboxes.shape[0]):
        for j in range(i + 1, bboxes.shape[0]):
            if iou2(bboxes[i, :], bboxes[j, :]) > 0.8:
                keep_idx[i if bboxes[i, 5] > bboxes[j, 5] else j] = 0
    return bboxes[keep_idx.astype(bool), :]


def load_yolo(dir_path: str) -> List[NDArray]:
    """
    Load all yolo formatted detections from a given directory where every detection is in file named '[frame_num].txt'
    Expected file is space separated where each row represents a detection formatted as:
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
                entry.append(line)
            raw_bbox.append(entry)
    return raw_bbox


def filter_tracklets(tracklets):
    """
    Filters a given list of tracklets & finalizes their object type

    :param tracklets: List of tracklets to be processed
    :return: Processed tracklets
    """
    new_tracklets = {}
    for k, v in tracklets.items():
        if not v.background:
            v.finalize_obj_type()
            v.obj_type.clear()  # frees the memory associated with the map
            new_tracklets[k] = v

    return new_tracklets


def output_yolo(results, tracklets, dir_path: str, starting_frame: int, ending_frame: int,
                print_uid: bool = True) -> None:
    """
    Outputs results into a subdirectory in YOLO format

    :param results: Detection results from do_tracking()
    :param tracklets: Auxiliary tracklet from do_tracking() that have been filtered whose object type has been finalized
    :param dir_path: Path to a location where a subdirectory named "YOLO" will be created
    :param starting_frame: Starting frame number, used to name the files
    :param ending_frame: Ending frame number used to name the files
    :param print_uid: Prints the uid for each tracklet if True, otherwise ht uid will not be printed (For compatibility with MakeSense.ai)
    """
    # Output result in YOLO format
    output_dir = os.path.join(dir_path, "YOLO")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    res = iter(results)

    # Iterate through every frame
    for i in range(starting_frame, ending_frame):

        # zero pad output file name
        fname = os.path.join(output_dir, str(i).zfill(10) + ".txt")
        # Set up csv writer
        with open(fname, 'w') as f:
            writer = csv.writer(f, delimiter=' ')
            res_by_frame = next(res)
            # Check every tracklet
            for r in res_by_frame:
                # Skip bboxes that don't have matching auxiliary tracklet object
                if r[-1] not in tracklets:
                    continue

                # Update object type
                r[0] = tracklets[r[-1]].final_type
                if print_uid:
                    writer.writerow(r)
                else:
                    writer.writerow(r[:-1])
    # Copies the label file to the output directory if not already present
    if not os.path.exists(os.path.join(dir_path, 'labels.txt')):
        src_path = os.path.join(os.path.dirname(__file__), "../extras", "labels.txt")
        shutil.copy2(src=src_path, dst=os.path.join(output_dir, 'labels.txt'))
    # print("Tracked YOLO files stored at " + output_dir)


def centroid_dist(det, pred) -> float:
    """
    Calculates the centroid distance between a detection and a prediction

    :param det: Detection object
    :param pred: Prediction object (estimate)
    :return: Centroid distance (possibly normalized)
    """
    DIST_LIM = 1080 / 20
    detection_centroid = np.sum(det.points, axis=0) / len(det.points)
    tracked_centroid = np.sum(pred.estimate, axis=0) / len(det.points)
    distances = np.linalg.norm(detection_centroid - tracked_centroid, axis=0)
    return distances / (DIST_LIM + distances)


def reversed_iou(det, pred) -> float:
    """
    Performs a 'reversed' IOU calculation. The returned value is 1-IOU of the detection and prediction

    :param det: Detection object
    :param pred: Prediction object (estimate)
    :return: The 'reversed' IOU. The equivalent of 1-IOU
    """
    det = det.points.flatten()
    pred = pred.estimate.flatten()
    return 1 - iou(det, pred)


def norfair2yolo(bbox: NDArray, dims=[1440, 1080], do_clip=True):
    """
    Converts a norfair bbox into a YOLO bbox for an image of size dims.

    :param bbox: Norfair bbox (x1, y1, x2, y2)
    :param dims: Dimensions of the image. Defaults to 1440x1080
    :param do_clip: The results will be constrained to between 0,1 if True.
                    This parameter is for compatibility with https://www.makesense.ai
    :return: np.array of a YOLO format bbox
    """

    norfair_bbox = bbox / np.array(dims)
    YOLO_bbox = np.concatenate([np.sum(norfair_bbox, axis=0) / 2, norfair_bbox[1, :] - norfair_bbox[0, :]], axis=0)
    YOLO_bbox = np.around(YOLO_bbox, decimals=6)
    # TODO: Potentially remove this clipping function if we aren't using https://www.makesense.ai for viz
    if do_clip:
        return np.clip(YOLO_bbox, 0, 1)
    return YOLO_bbox


def add_bbox_res(norfair_bbox, label, id):
    """
    Formats a bbox result into a list object containing a YOLO bbox

    :param norfair_bbox: A bbox in norfair format
    :param label: Category label
    :param id: Uid of the object
    :return: Formatted list containing the input data
    """
    YOLO_bbox = norfair2yolo(norfair_bbox).flatten()
    return [int(label)] + YOLO_bbox.tolist() + [id]


def do_tracking(input_dir):
    """
    Runs the tracking program on the input directory, then outputs the result to the output directory

    :param input_dir: The input directory to look for YOLO detections.
    :return: A list of lists containing tracked detections. Also returns a list of tracklet objects containing
            details about the detections.
    """

    # Load raw detection bboxes
    det_bbox = load_yolo(input_dir)
    results_list = []

    # Construct a norfair Tracker object
    tracker: Tracker = Tracker(
        distance_function=reversed_iou,
        distance_threshold=0.65,
        detection_threshold=0.4,
        hit_counter_max=12,
        # initialization_delay=0,
        initialization_delay=5,
        past_detections_length=6
    )

    # Auxiliary tracklet object to keep track of additional info
    tracklets = {}
    init_tracklets = {}

    # Iterate through the data, get all the detections by each frame
    i = 0
    for bbox_by_frame in det_bbox:
        # List to store all estimate (predicted) bboxes
        tracked_bbox = []
        # Obtain all YOLO detection for this frame and convert them to norfair detections
        dets_by_frame = yolo2norfair_multi(bbox_by_frame)
        # Update tracker, obtain a list of tracked objects
        curr_objs = tracker.update(dets_by_frame)
        # Add tracked objects to list of bbox
        for tracked_obj in curr_objs:
            if tracked_obj.hit_counter_is_positive and (tracked_obj.estimate[1, 0] > 0.05 or tracked_obj.age < 50):
                # If the current detection just finished init delay, go back and add the predictions made during init
                if not tracked_obj.is_initializing_flag and tracked_obj.id not in tracklets.keys():
                    # Move tracklet from init_tracklets to tracklets
                    tracklets[tracked_obj.id] = init_tracklets.pop(tracked_obj.initializing_id)
                    tracklets[tracked_obj.id].update_id(tracked_obj.id)
                    # Copy estimates made during initialization to results_list
                    past_det_idx = len(results_list) - len(tracklets[tracked_obj.id].estimates) - 1
                    for j in range(len(tracklets[tracked_obj.id].estimates)):
                        results_list[past_det_idx + j].append(tracklets[tracked_obj.id].estimates[j])
                    # Remove estimates stored in tracklet obj
                    tracklets[tracked_obj.id].clear_estimates()

                #     # Create new auxiliary tracklet object
                #     tracklets[tracked_obj.id] = tracklet(tracked_obj.id)
                #     # find the list in results_list that contains all detections for the frame where this tracked_obj started init
                #     past_det_idx = len(results_list)-tracked_obj.past_detections_length-1

                # Add detection for current frame
                entry = add_bbox_res(tracked_obj.estimate, tracked_obj.label, tracked_obj.id)
                # Update auxiliary tracklet data
                tracklets[tracked_obj.id].update(tracked_obj.last_detection.data, norfair2yolo(tracked_obj.estimate))
                tracked_bbox.append(entry)

        for obj in tracker.tracked_objects:
            # Ignore already processed Detections
            if obj in curr_objs:
                continue
            # Create & update auxiliary tracklet objects for initializing objects
            if obj.is_initializing_flag and obj.hit_counter_is_positive:
                if obj.initializing_id not in init_tracklets.keys():
                    init_tracklets[obj.initializing_id] = tracklet(obj.initializing_id)
                init_tracklets[obj.initializing_id].update(obj.last_detection.data, norfair2yolo(obj.estimate))
                init_tracklets[obj.initializing_id].add_estimate(
                    add_bbox_res(obj.estimate, obj.label, obj.initializing_id))

        results_list.append(tracked_bbox)
        i += 1

        if i % 50 == 0:
            # Update progress & cleanup unused tracklets
            new_init_tracklets = {}
            for obj in tracker.tracked_objects:
                if obj.initializing_id in init_tracklets.keys():
                    new_init_tracklets[obj.initializing_id] = init_tracklets.pop(obj.initializing_id)
            init_tracklets.clear()
            init_tracklets = new_init_tracklets
            # print("Processed frame %d"%i)
    return results_list, tracklets


def runner(path: str):
    res, tracklets_res = do_tracking(os.path.join(path))
    tracklets_res = filter_tracklets(tracklets_res)
    output_yolo(res, tracklets_res, path, 0, len(res))


if __name__ == "__main__":
    path = "D:\\UWCR Data2\\2019_04_09\\2019_04_09_pms1000\\images_0"
    runner(path)
    print("Done")

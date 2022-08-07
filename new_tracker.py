import numpy as np
from norfair import Detection, Tracker
from utilities import iou

# Regex
import re

# File I/O
import glob
import os
import csv
import shutil

def yolo2norfair(bbox, dims=[1440, 1080]) -> Detection:
    """
    Converts a single YOLO bbox to a norfair detection object

    :param bbox: A YOLO bounding box
    :param dims: Dimensions of the image in px (Optional, Defaults to [1440, 1080])
    :return: Constructed norfair detection object
    """
    new_bbox = np.asarray([[bbox[1] - bbox[3] / 2, bbox[2] - bbox[4] / 2],
            [bbox[1] + bbox[3] / 2, bbox[2] + bbox[4] / 2]])
    new_bbox = new_bbox * np.asarray(dims)
    score = np.asarray([bbox[5], bbox[5]])
    label = str(int(bbox[0]))
    return Detection(new_bbox, score, data=bbox[0], label=label)

def yolo2norfair_multi(bboxes, dims=[1440,1080]):
    """
    Convert a list of YOLO detections into a list of norfair detections

    :param bboxes: List of YOLO detection bounding boxes
    :param dims: Dimensions of the image in px (Optional, Default sot [1440, 1080])
    :return: List of norfair detections. Empty list if there are no input bboxes
    """
    norfair_dets = []
    for bbox in bboxes:
        bbox = np.array(bbox,dtype=float)
        norfair_dets.append(yolo2norfair(bbox, dims))
    return norfair_dets


def load_yolo(dir_path):
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


def output_yolo(results, dir_path: str, starting_frame: int, ending_frame: int):
    # Output result in YOLO format
    print("Printing YOLO output to a subdirectory")
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
                writer.writerow(r)
    # Copies the label file to the output directory if not already present
    if not os.path.exists(os.path.join(dir_path, 'labels.txt')):
        src_path = os.path.join(os.path.dirname(__file__), "extras", "labels.txt")
        shutil.copy2(src=src_path, dst=os.path.join(output_dir, 'labels.txt'))
    print("Tracked YOLO files stored at " + output_dir)


def centroid_dist(det, pred):
    DIST_LIM = 1080/20
    detection_centroid = np.sum(det.points, axis=0)/len(det.points)
    tracked_centroid = np.sum(pred.estimate, axis=0)/len(det.points)
    distances = np.linalg.norm(detection_centroid - tracked_centroid, axis=0)
    return distances / (DIST_LIM + distances)


def reversed_iou(det, pred):
    """
    Performs a 'reversed' IOU calculation. The returned value is 1-IOU of the detection and prediction
    :param det:
    :param pred:
    :return:
    """
    det = det.points.flatten()
    pred = pred.estimate.flatten()
    return 1-iou(det, pred)


def do_tracking(input_dir):
    """
    Runs the tracking program on the input directory, then outputs the result to the output directory
    :param input_dir:
    :param output_dir:
    :return:
    """
    # Load raw detection bboxes
    det_bbox = load_yolo(input_dir)
    results_list = []

    # Construct a norfair Tracker object
    tracker:Tracker = Tracker(
        distance_function=reversed_iou,
        distance_threshold=0.65,
        detection_threshold= 0.25,
        hit_counter_max=15,
        initialization_delay=0
    )


    # Iterate through the data, get all the detections by each frame
    i = 0
    for bbox_by_frame in det_bbox:
        # List to store all estimate (predicted) bboxes
        tracked_bbox = []
        # Obtain all YOLO detection for this frame and convert them to norfair detections
        dets_by_frame = yolo2norfair_multi(bbox_by_frame)
        tracked_objs = tracker.update(dets_by_frame)
        for tracklet in tracked_objs:
            if tracklet.hit_counter_is_positive and (tracklet.estimate[1, 0] > 0.05 or tracklet.age < 50):
                norfair_bbox = tracklet.estimate / np.array([1440, 1080])
                YOLO_bbox = np.concatenate([np.sum(norfair_bbox, axis=0) / 2, norfair_bbox[1,:]-norfair_bbox[0,:]], axis=0)
                YOLO_bbox = np.around(YOLO_bbox, decimals=6)
                # TODO: Potentially remove this clipping function if we aren't using https://www.makesense.ai for viz
                YOLO_bbox = np.clip(YOLO_bbox, 0, 1)

                entry = [int(tracklet.label)] + YOLO_bbox.flatten().tolist()
                tracked_bbox.append(entry)
        results_list.append(tracked_bbox)
        i += 1
        if i == 5:
            tracker.initialization_delay = 5
        if i % 100 == 0:
            print("Processed frame %d"%i)
    return results_list



if __name__ == "__main__":
    path = "D:\\UWCR Data\\2019_04_30\\2019_04_30_mlms001\\images_0\\"
    res = do_tracking(os.path.join(path))
    output_yolo(res, path, 0, len(res))
    # get_path_by_file("/Volumes/Untitled/VOC and test data/Image Tests/2021_06_11_lpkf_ococ009/")

import os
import json
import math
import random

import pandas as pd
import numpy as np
from utils import find_nearest, confmap2ra, labelmap2ra, cart2pol, pol2cart

from config import class_ids
from config import radar_configs, rodnet_configs

range_grid = confmap2ra(radar_configs, name='range')
angle_grid = confmap2ra(radar_configs, name='angle')
range_grid_label = labelmap2ra(radar_configs, name='range')
angle_grid_label = labelmap2ra(radar_configs, name='angle')

label_map = {0: 'person',
             1: 'bicycle',
             2: 'car',
             3: 'motorbike',
             5: 'bus',
             7: 'truck',
             80: 'cyclist',
             }


mapping = {0: [0.7, 0.7],
           1: [0.6, 1.7],
           2: [1.8, 4.5],
           3: [0.8, 2],
           5: [2.5, 12],
           7: [2.6, 15],
           80: [0.6, 1.7]
           }


def read_ra_labels_csv(seq_path):
    label_csv_name = os.path.join(seq_path, 'ramap_labels.csv')
    data = pd.read_csv(label_csv_name)
    n_row, n_col = data.shape
    obj_info_list = []
    cur_idx = -1

    for r in range(n_row):
        filename = data['filename'][r]
        frame_idx = int(filename.split('.')[0].split('_')[-1])
        if cur_idx == -1:
            obj_info = []
            if filename[0:10] == '2020_00_00' and frame_idx > 0:
                print('fill in the gaps of ', frame_idx, ' frames')
                for empty in range(frame_idx):
                    obj_info_list.append([])
            cur_idx = frame_idx
        if frame_idx > cur_idx:
            obj_info_list.append(obj_info)
            # deal with the csv doesn't have row for empty frame
            if frame_idx - cur_idx > 1:
                for empty in range(frame_idx-cur_idx-1):
                    obj_info_list.append([])
            obj_info = []
            cur_idx = frame_idx

        region_count = data['region_count'][r]
        region_id = data['region_id'][r]

        if region_count != 0:
            region_shape_attri = json.loads(data['region_shape_attributes'][r])
            region_attri = json.loads(data['region_attributes'][r])

            cx = region_shape_attri['cx']
            cy = region_shape_attri['cy']
            distance = range_grid_label[cy]
            angle = angle_grid_label[cx]
            if distance > rodnet_configs['rr_max'] or distance < rodnet_configs['rr_min']:
                continue
            if angle > rodnet_configs['ra_max'] or angle < rodnet_configs['ra_min']:
                continue
            rng_idx, _ = find_nearest(range_grid, distance)
            agl_idx, _ = find_nearest(angle_grid, angle)
            try:
                class_str = region_attri['class']
            except:
                print("missing class at row %d" % r)
                continue
            try:
                class_id = class_ids[class_str]
            except:
                if class_str == '':
                    print("no class label provided!")
                    raise ValueError
                else:
                    class_id = -1000
                    print(class_str)
                    print("Warning class not found! %s %010d" % (seq_path, frame_idx))
            obj_info.append([rng_idx, agl_idx, class_id])

    obj_info_list.append(obj_info)

    return obj_info_list


def find_match(bboxes, range, angle, class_match, used_list):
    closet_idx = None
    smallest_met = 1000
    for idx, bbox in enumerate(bboxes):
        range_bbox, angle_bbox = cart2pol(bbox[2], bbox[3])
        angle_bbox = math.degrees(angle_bbox)
        met = abs(range_bbox - range) * 0.5 + abs(angle_bbox - angle)
        if int(bbox[1]) in class_match and abs(angle_bbox - angle) < 20 and met < smallest_met and idx not in used_list:
            smallest_met = met
            closet_idx = idx

    return closet_idx


if __name__ == '__main__':
    src_path = "D:\center_correct"
    # seqs = ['2019_05_29_pcms005'] '2019_05_29_mlms006', '2019_05_29_bcms000', '2019_05_09_mlms003',
    # '2019_04_30_pcms001', '2019_04_30_mlms000', '2019_04_09_pms2000', '2019_04_30_mlms001', '2019_05_09_bm1s007'
    # '2019_05_09_cm1s003'
    seqs = ['2019_04_30_pbms002']

    for seq in seqs:
        seq_dir = os.path.join(src_path, seq)
        # read ramap labels
        center_gt = read_ra_labels_csv(seq_dir)

        # read the matched filter labels
        # match_label_dir = os.path.join(seq_dir, 'vis_labels\\text_label')
        match_label_dir = os.path.join(seq_dir, 'text_label')
        label_files = sorted(os.listdir(match_label_dir))

        # save folder
        # save_label_dir = os.path.join(seq_dir, 'vis_labels\\text_label_corrected')
        save_label_dir = os.path.join(seq_dir, 'text_label_corrected')
        if not os.path.exists(save_label_dir):
            # Create a new directory because it does not exist
            os.makedirs(save_label_dir)

        for file_id, file in enumerate(label_files):
            if file_id % 30 == 0:
                # extra tracking id
                extra_uid = {0: random.randrange(50, 300),
                             1: random.randrange(50, 300),
                             2: random.randrange(50, 300),
                             3: random.randrange(50, 300),
                             5: random.randrange(50, 300),
                             7: random.randrange(50, 300),
                             80: random.randrange(50, 300),
                            }

            f_path = os.path.join(match_label_dir, file)
            save_f_dir = os.path.join(save_label_dir, file)
            # Load radar bboxes
            bboxes = np.genfromtxt(f_path, dtype=float, delimiter=',')
            if len(bboxes.shape) == 1:
                bboxes = bboxes.reshape((1, bboxes.shape[0]))

            bboxes_correct = []

            frame_id = int(file.split('.')[0])
            used_list = []
            for center_label in center_gt[frame_id]:
                range = range_grid[center_label[0]]
                angle = angle_grid[center_label[1]]
                px, py = pol2cart(range, math.radians(angle))
                class_id = center_label[2]
                if class_id == 0:
                    class_match = [0]
                if class_id == 1:
                    class_match = [80, 1, 3]
                if class_id == 2:
                    class_match = [2, 3, 5, 7]
                if class_id == 4:
                    class_match = [7]

                closet_idx = find_match(bboxes, range, angle, class_match, used_list)
                if closet_idx is not None:
                    used_list.append(closet_idx)
                    info = bboxes[closet_idx]
                    bboxes_correct.append([info[0], info[1], px, py, info[4], info[5]])
                else:
                    bboxes_correct.append([extra_uid[class_match[0]], class_match[0], px, py,
                                           mapping[class_match[0]][0], mapping[class_match[0]][1]])

            bboxes_correct = np.asarray(bboxes_correct)
            np.savetxt(save_f_dir, bboxes_correct, fmt='%.2f', delimiter=',')

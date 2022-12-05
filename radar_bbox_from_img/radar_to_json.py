import os
import csv
import numpy as np
from numpy.typing import NDArray

out_dir = "D:\\UWCR Data\\2019_04_09\\2019_04_09_cms1000\\vis_labels\\"
seq_dir = os.path.join(out_dir, "text_label")

frames = os.listdir(seq_dir)

label_map = {0: 'person',
             1: 'bicycle',
             2: 'car',
             3: 'motorbike',
             5: 'bus',
             7: 'truck',
             80: 'cyclist',
             }

# Pixel offsets
x_width = 640
x_offset = 1

y_offset = 82
y_height = 317

y_dim = 28
x_dim = 56

img_resolution = [640, 480]

def scale_box_to_img(bbox):
    x = bbox[:, 2:3] - bbox[:, 4:5] / 2
    x = (x + x_dim/2) / x_dim
    x = x * x_width + x_offset

    y = bbox[:, 3:4] + bbox[:, 5:6] / 2
    y = (y_height - (y / y_dim * y_height)) + y_offset

    w = bbox[:, 4:5]/x_dim * x_width
    h = bbox[:, 5:6]/y_dim * y_height

    res = np.concatenate((x, y, w, h), 1)
    res = np.round(res)

    return res.astype(int)


"""
filename,file_size,file_attributes,region_count,region_id,region_shape_attributes,region_attributes
"""

out_path = os.path.join(out_dir, "raw_box.csv")

out_file = open(out_path, 'w', newline='\n')
writer = csv.writer(out_file)

# Initialize 2D list with the header row
all_rows = [['filename', 'file_size', 'file_attributes', 'region_count', 'region_id', 'region_shape_attributes',
             'region_attributes']]

# Process frame by frame
for f_name in frames:
    f_path = os.path.join(seq_dir, f_name)
    img_name = f_name[:-4] + ".png"
    # Set of the shared part of for all bboxes in this frame
    row_base = [img_name, os.path.getsize(os.path.join(out_dir, img_name)), "{}"]
    # Load radar bboxes
    bboxes: NDArray = np.genfromtxt(f_path, dtype=float, delimiter=',')

    # Determine the number of entries in this frame
    region_count = bboxes.shape[0]  # No. of bboxes = No. rows
    # TODO: Handle edge case with 0 or 1 bbox entries
    if region_count == 0:
        continue
    if len(bboxes.shape) == 1:
        bboxes = bboxes.reshape((1, bboxes.shape[0]))

    # Get uids, types, and bboxes
    uids = bboxes[:, 0:1].flatten().astype(int)
    types = bboxes[:, 1:2].flatten().astype(int)
    VGG_boxes = scale_box_to_img(bboxes)

    if type(uids) is int:
        uids = np.array([uids])
        types = np.array([types])

    for i in range(len(uids)):
        # Bbox shape attributes
        region_shape_attributes = "{\"name\":\"rect\",\"x\":%s,\"y\":%s,\"width\":%s,\"height\":%s}" % (VGG_boxes[i, 0],  VGG_boxes[i, 1], VGG_boxes[i, 2], VGG_boxes[i, 3])
        # Bbox type & uid
        region_attributes = "{\"type\":\"%s\",\"uid\":\"%s\"}" % (types[i], uids[i])
        # Build and append row to local frame cache
        temp_row = row_base + [region_count, i, region_shape_attributes, region_attributes]
        all_rows.append(temp_row)

writer.writerows(all_rows)

out_file.close()

import os
import csv
import numpy as np
from numpy.typing import NDArray

out_dir = "D:\\UWCR Data\\2019_04_09\\2019_04_09_cms1000\\vis_labels\\"
seq_dir = os.path.join(out_dir, "text_label")

frames = os.listdir(seq_dir)

uid_to_remove = {38}

# Initialize 2D list with the header row
all_rows = [['filename', 'file_size', 'file_attributes', 'region_count', 'region_id', 'region_shape_attributes',
             'region_attributes']]

# Process frame by frame
for f_name in frames:
    f_path = os.path.join(seq_dir, f_name)
    rows = []
    dirty = False

    with open(f_path, 'r') as in_file:
        reader = csv.reader(in_file)

        for line in reader:
            if int(line[0]) in uid_to_remove:
                dirty = True
                continue
            rows.append(line)

    if not dirty:
        continue

    with open(f_path, 'w', newline='\n') as out_file:
        writer = csv.writer(out_file)
        writer.writerows(rows)


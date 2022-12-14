# class settings
n_class = 3
class_table = {
    0: 'pedestrian',
    1: 'cyclist',
    2: 'car',
    # 3: 'van',
    4: 'truck',
}

class_ids = {
    'pedestrian': 0,
    'cyclist': 1,
    'car': 2,
    'truck': 4,  # TODO: due to detection model bug
    'train': 2,
    'noise': -1000,
}

confmap_sigmas = {
    'pedestrian': 15,
    'cyclist': 20,
    'car': 30,
    # 'van': 12,
    # 'truck': 20,
}

confmap_sigmas_interval = {
    'pedestrian': [5, 15],
    'cyclist': [8, 20],
    'car': [10, 30],
    # 'van': 12,
    # 'truck': 20,
}

confmap_length = {
    'pedestrian': 1,
    'cyclist': 2,
    'car': 3,
    # 'van': 12,
    # 'truck': 20,
}

radar_configs = {
    'ramap_rsize': 128,             # RAMap range size
    'ramap_asize': 128,             # RAMap angle size
    'ramap_vsize': 128,             # RAMap angle size
    'frame_rate': 30,
    'crop_num': 3,                  # crop some indices in range domain
    'n_chirps': 255,                # number of chirps in one frame
    'sample_freq': 4e6,
    'sweep_slope': 21.0017e12,
    'data_type': 'RISEP',           # 'RI': real + imaginary, 'AP': amplitude + phase
    'ramap_rsize_label': 122,       # TODO: to be updated
    'ramap_asize_label': 121,       # TODO: to be updated
    'ra_min_label': -60,            # min radar angle
    'ra_max_label': 60,             # max radar angle
    'rr_min': 1.0,                  # min radar range (fixed)
    'rr_max': 25.0,                 # max radar range (fixed)
    'ra_min': -90,                  # min radar angle (fixed)
    'ra_max': 90,                   # max radar angle (fixed)
    'ramap_folder': 'WIN_HEATMAP',
}

# network settings
rodnet_configs = {
    'data_folder': 'WIN_PROC_MAT_DATA',
    # 'label_folder': 'dets_3d',
    'label_folder': 'dets_refine',
    'n_epoch': 100,
    'batch_size': 3,
    'learning_rate': 1e-5,
    'lr_step': 5,       # lr will decrease 10 times after lr_step epoches
    'win_size': 16,
    'input_rsize': 128,
    'input_asize': 128,
    'rr_min': 1.0,                  # min radar range
    'rr_max': 24.0,                 # max radar range
    'ra_min': -90.0,                  # min radar angle
    'ra_max': 90.0,                   # max radar angle
    'rr_min_eval': 1.0,                  # min radar range
    'rr_max_eval': 20.0,                 # max radar range
    'ra_min_eval': -60.0,                  # min radar angle
    'ra_max_eval': 60.0,                   # max radar angle
    'max_dets': 20,
    'peak_thres': 0.2,
    'ols_thres': 0.2,
    'stacked_num': 2,
    'test_stride': 8,
}

import numpy as np
import matplotlib.pyplot as plt
import re
import os 
import scipy
import math


# radar_configs = {
#     'ramap_rsize': 128,             # RAMap range size
#     'ramap_asize': 128,             # RAMap angle size
#     'ramap_vsize': 128,             # RAMap angle size
#     'frame_rate': 30,
#     'crop_num': 3,                  # crop some indices in range domain
#     'n_chirps': 255,                # number of chirps in one frame
#     'sample_freq': 4e6,
#     'sweep_slope': 21.0017e12,
#     'data_type': 'RISEP',           # 'RI': real + imaginary, 'AP': amplitude + phase
#     'ramap_rsize_label': 122,       # TODO: to be updated
#     'ramap_asize_label': 121,       # TODO: to be updated
#     'ra_min_label': -60,            # min radar angle
#     'ra_max_label': 60,             # max radar angle
#     'rr_min': 1.0,                  # min radar range (fixed)
#     'rr_max': 25.0,                 # max radar range (fixed)
#     'ra_min': -90,                  # min radar angle (fixed)
#     'ra_max': 90,                   # max radar angle (fixed)
#     'ramap_folder': 'WIN_HEATMAP',
# }

# def confmap2ra(radar_configs, name, radordeg=None):
#     """
#     Map confidence map to range(m) and angle(deg): not uniformed angle
#     :param name: 'range' for range mapping, 'angle' for angle mapping
#     :return: mapping grids
#     """
#     # TODO: add more args for different network settings
#     Fs = radar_configs['sample_freq']
#     sweepSlope = radar_configs['sweep_slope']
#     num_crop = radar_configs['crop_num']
#     fft_Rang = radar_configs['ramap_rsize'] + 2*num_crop
#     fft_Ang = radar_configs['ramap_asize']
#     c = 3e08
#
#     if name == 'range':
#         freq_res = Fs / fft_Rang
#         freq_grid = np.arange(fft_Rang) * freq_res
#         rng_grid = freq_grid * c / sweepSlope / 2
#         rng_grid = rng_grid[num_crop:fft_Rang - num_crop]
#         return rng_grid
#
#     if name == 'angle':
#         # for [-90, 90], w will be [-1, 1]
#         w = np.linspace(math.sin(math.radians(radar_configs['ra_min'])),
#                         math.sin(math.radians(radar_configs['ra_max'])),
#                         radar_configs['ramap_asize'])
#         if radordeg is None or radordeg == 'deg':
#             agl_grid = np.degrees(np.arcsin(w))  # rad to deg
#         elif radordeg == 'rad':
#             agl_grid = np.arcsin(w)
#         else:
#             raise TypeError
#         return agl_grid


def prompt_frame_num(prev):
    try:
        frame_num = input("Enter frame num:\t")
    except EOFError:
        print()
        exit()
    if frame_num == '':
        frame_num = str(int(prev[:-4])+1)
    return frame_num.zfill(6) + ".npy"


def visualize(file, title):
    a = np.load(file)

    b = a[:, :, 0] ** 2 + a[:,:,1] ** 2
    plt.ylim(0,128)
    plt.title(title)
    plt.imshow(b)
    plt.show()


if __name__ == "__main__":
    # visualize("D:\\UWCR Data\\2019_04_09\\2019_04_09_cms1000\\radar\\000000.npy")
    # exit()

    # range_grid = confmap2ra(radar_configs, 'range')
    # angle_grid = confmap2ra(radar_configs, 'angle', 'deg')
    # print(angle_grid)
    # print(len(angle_grid))
    # exit()

    src_root = "D:\\UWCR Data\\"
    date_pattern = re.compile("\d{4}_\d{2}_\d{2}")

    # Prompt user for sequence name
    try:
        seq_name = input("Entry sequence name:\t")
    except EOFError:
        print()
        exit()
    date = date_pattern.search(seq_name).group()
    radar_dir = os.path.join(src_root, date, seq_name, "radar")

    while not os.path.exists(radar_dir):
        print('Invalid seq name. You entered: %s'%radar_dir)
        try:
            seq_name = input("Entry sequence name:\t")
        except EOFError:
            print()
            exit()
        date = date_pattern.search(seq_name).group()
        radar_dir = os.path.join(src_root, date, seq_name, "radar")

    plt.ion()
    frame_num = '000000.npy'
    while True:
        # Prompt user for frame number
        frame_num = prompt_frame_num(frame_num)
        radar_file = os.path.join(radar_dir, frame_num)
        while not os.path.exists(radar_file):
            frame_num = prompt_frame_num(frame_num)
            radar_file = os.path.join(radar_dir, frame_num)
        print(radar_file)
        # a = np.load("D:\\UWCR Data\\2019_04_09\\2019_04_09_cms1000\\radar\\000000.npy")
        visualize(radar_file, frame_num)


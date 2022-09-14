import csv
import numpy as np
import matplotlib.pyplot as plt
import re
import os 
import scipy
import math


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

def confmap2ra(radar_configs, name, radordeg=None):
    """
    Map confidence map to range(m) and angle(deg): not uniformed angle
    :param name: 'range' for range mapping, 'angle' for angle mapping
    :return: mapping grids
    """
    # TODO: add more args for different network settings
    Fs = radar_configs['sample_freq']
    sweepSlope = radar_configs['sweep_slope']
    num_crop = radar_configs['crop_num']
    fft_Rang = radar_configs['ramap_rsize'] + 2*num_crop
    fft_Ang = radar_configs['ramap_asize']
    c = 3e08

    if name == 'range':
        freq_res = Fs / fft_Rang
        freq_grid = np.arange(fft_Rang) * freq_res
        rng_grid = freq_grid * c / sweepSlope / 2
        rng_grid = rng_grid[num_crop:fft_Rang - num_crop]
        return rng_grid

    if name == 'angle':
        # for [-90, 90], w will be [-1, 1]
        w = np.linspace(math.sin(math.radians(radar_configs['ra_min'])),
                        math.sin(math.radians(radar_configs['ra_max'])),
                        radar_configs['ramap_asize'])
        if radordeg is None or radordeg == 'deg':
            agl_grid = np.degrees(np.arcsin(w))  # rad to deg
        elif radordeg == 'rad':
            agl_grid = np.arcsin(w)
        else:
            raise TypeError
        return agl_grid
    

def calc_conversion_grid():
    return confmap2ra(radar_configs, 'range'), confmap2ra(radar_configs, 'angle', 'rad')


if __name__ == '__main__':
    range_grid, angle_grid = calc_conversion_grid()
    new_data = []

    with open('person.csv', 'r') as f:
        raw_data = np.genfromtxt(f, dtype= int, delimiter=',')
    
    with open('output.csv', 'w') as f:
        for i in range(raw_data.shape[0]):
            rang = range_grid[raw_data[i, 1]]
            angle = angle_grid[raw_data[i, 0]]
            # print("%f %f"%(range, angle))
            x = np.sin(angle)*rang
            y = np.cos(angle)*rang
            print("%f %f"%(x, y))
            f.write("%f %f\n"%(x, y))


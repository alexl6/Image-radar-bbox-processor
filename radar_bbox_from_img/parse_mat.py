import numpy as np
from radar_utilities import load_mat
from numpy.typing import NDArray
from convert import calc_conversion_grid
from radar_utilities import polar_in_cartesian

import matplotlib.pyplot as plt

def parser(path: str):
    R_data = load_mat(path, 'R_data')
    b = R_data[:,:,0]
    c = np.fft.fft(b, 128, axis=1)
    c = np.fft.fftshift(c, 1)

    return np.power(np.abs(c), 2)

if __name__ == '__main__':
    a = parser("D:\\UWCR Data\\2019_04_09\\2019_04_09_cms1000\\WIN_R_MAT\\2019_04_09_cms1000_000002.mat")
    # This is equivalent to the output using visualization.py
    plt.imshow(a)
    plt.show()

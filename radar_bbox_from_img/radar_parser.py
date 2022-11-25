import numpy as np
from radar_utilities import load_mat
from numpy.typing import NDArray
from convert import calc_conversion_grid
from radar_utilities import polar_in_cartesian
import os
import matplotlib.pyplot as plt
import re

# Regex pattern for matching sequence name
seq_name_pattern = re.compile("\d{4}_\d{2}_\d{2}_[a-z0-9]+")


def mat_parser(f_num: int, seq_path: str) -> NDArray:
    seq_name = seq_name_pattern.search(seq_path).group()

    path = os.path.join(seq_path, 'WIN_R_MAT', seq_name + '_' + str(f_num).zfill(6) + '.mat')
    R_data = load_mat(path, 'R_data')
    b = R_data[:, :, 0]
    c = np.fft.fft(b, 128, axis=1)
    c = np.fft.fftshift(c, 1)

    return np.power(np.abs(c), 2)


def npy_parser(f_num: int, seq_path: str) -> NDArray:
    frame_num = str(f_num).zfill(6) + ".npy"
    a = np.load(os.path.join(seq_path, "radar", frame_num))
    b = a[:, :, 0] ** 2 + a[:, :, 1] ** 2
    return b.flatten()


def parse_radar_data(f_num: int, seq_path: str) -> NDArray:
    """
    Obtains the visual representation of a given radar frame

    :param f_num: frame number
    :param seq_path: path to the directory of radar detections
    :return: radar intensity in a flattened array
    """
    if os.path.exists(os.path.join(seq_path, "radar")):
        return npy_parser(f_num, seq_path)
    elif os.path.exists(os.path.join(seq_path, "WIN_R_MAT")):
        return mat_parser(f_num, seq_path)
    else:
        err_msg = "Unable to find valid radar file at %s"%seq_path
        raise FileNotFoundError(err_msg)


if __name__ == '__main__':
    # a = mat_parser("D:\\UWCR Data\\2019_04_09\\2019_04_09_cms1000\\WIN_R_MAT\\2019_04_09_cms1000_000002.mat")
    a = mat_parser(2, "D:\\UWCR Data\\2019_04_09\\2019_04_09_cms1000\\")

    # This is equivalent to the output using visualization.py
    plt.imshow(a)
    plt.show()

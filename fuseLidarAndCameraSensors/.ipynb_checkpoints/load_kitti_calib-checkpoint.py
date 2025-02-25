import numpy as np

def load_kitti_calib(calib_file):
    """Load KITTI calibration file into a dictionary."""
    data = {}
    with open(calib_file, 'r') as f:
        for line in f.readlines():
            key, value = line.split(":", 1)
            data[key] = np.array([float(x) for x in value.split()]).reshape(-1, 4 if 'P' in key else 3)
    return data
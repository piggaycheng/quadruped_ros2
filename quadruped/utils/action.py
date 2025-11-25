import numpy as np


def tanh_post_process(data: np.ndarray, limit: tuple[float, float]):
    # 將 (-1, 1) 的範圍縮放到目標範圍 [min, max]
    data_min, data_max = limit
    data_range = (data_max - data_min) / 2.0
    data_bias = (data_max + data_min) / 2.0
    scaled_data = data * data_range + data_bias
    return scaled_data

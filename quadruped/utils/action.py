import numpy as np


def tanh_process(data: np.ndarray, limit: tuple[float, float]):
    # 使用 tanh 將 data 從 (-inf, inf) 映射到 (-1, 1)
    tanh_data = np.tanh(data)
    # 將 (-1, 1) 的範圍縮放到目標範圍 [min, max]
    data_min, data_max = limit
    data_range = (data_max - data_min) / 2.0
    data_bias = (data_max + data_min) / 2.0
    scaled_data = tanh_data * data_range + data_bias
    return scaled_data

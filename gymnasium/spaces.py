"""最小 spaces 定义，覆盖本项目使用到的 Box/Discrete。"""

import numpy as np


class Discrete:
    def __init__(self, n):
        self.n = int(n)


class Box:
    def __init__(self, low, high, shape, dtype=np.float32):
        self.low = low
        self.high = high
        self.shape = tuple(shape)
        self.dtype = dtype


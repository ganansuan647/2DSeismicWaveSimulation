import bruges as bg
import numpy as np
from typing import Literal

def wavelet(wavelet_type: Literal["ricker", "berlage", "other"] = "ricker",
            f: float = 20.0,
            dt: float = 0.001,
            T: float = 2.0,
            ) -> tuple[np.ndarray, np.ndarray]:
    """
    生成震源子波。

    参数：
    -------
    wavelet_type : Literal["ricker", "berlage", "other"]
        子波类型，默认为 "ricker"。
    f : float
        子波中心频率（Hz）。
    dt : float
        时间步长（s）。
    T : float
        总时间长度（s）。

    返回：
    -------
    s : np.ndarray
        子波数值序列。
    tt : np.ndarray
        时间序列。
    """
    if wavelet_type == "ricker":
        s, tt = bg.filters.ricker(duration=T, dt=dt, f=f)
    elif wavelet_type == "berlage":
        s, tt = bg.filters.wavelets.berlage(duration=1.4*T, dt=dt, f=f)
    else:
        raise ValueError(f"Unsupported wavelet type: {wavelet_type}.")
    return s, tt

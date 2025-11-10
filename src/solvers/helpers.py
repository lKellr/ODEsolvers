from typing import Any, Callable
import numpy as np
from numpy.typing import NDArray

norm_hairer: Callable[[NDArray[np.floating]], float] = (
    lambda x: np.sum(1 / x.size * np.abs(x) ** 2) ** 0.5
)


controller_I: Callable[[float, float, float], float] = (
    lambda err_ratio, err_ratio_last, p: 1.0 / err_ratio ** (1 / p)
)
controller_PI: Callable[[float, float, float, float], float] = (
    lambda err_ratio, err_ratio_last, alpha, beta: 1.0
    / err_ratio**alpha
    * err_ratio_last**beta
)

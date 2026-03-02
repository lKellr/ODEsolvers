from typing import Any, Callable
import numpy as np
from numpy.typing import NDArray
from solvers.embedded import DP45
from modules.helpers import norm_hairer


def find_local_errors(
    x_dot: Callable[[NDArray[np.floating]], NDArray[np.floating]],
    t: NDArray[np.floating],
    x_computed: NDArray[np.floating],
    norm: Callable[[NDArray[np.floating]], float] = norm_hairer,
) -> NDArray[np.floating]:
    """computes the local error using DP45"""
    err_loc = np.empty(x_computed.shape[0])
    err_loc[0] = 0.0
    for ix_time in range(t.size - 1):
        _, x_analytic, _ = DP45(
            x_dot,
            x0=x_computed[ix_time],
            t0=t[ix_time],
            t_max=t[ix_time + 1],
            h_limits=(1e-20, np.inf),
            atol=1e-16,
            rtol=1e-9,
        )
        err_loc[ix_time + 1] = norm(x_computed[ix_time + 1] - x_analytic[-1])
    return err_loc

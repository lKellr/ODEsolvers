from typing import Any, Callable
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root

norm_hairer: Callable[[NDArray[np.floating]], float] = (
    lambda x: np.sum(1 / x.size * np.abs(x) ** 2) ** 0.5
)


def root_wrapped(
    fun: Callable[[NDArray[np.floating]], NDArray[np.floating]],
    x0: NDArray[np.floating],
    jac_fun: Callable[[NDArray[np.floating]], NDArray[np.floating]] | None,
    tol: float | NDArray[np.floating],
    method: str = "hybr",
) -> tuple[NDArray[np.floating], dict[str, Any]]:
    sol_result = root(fun, x0, tol=tol, jac=jac_fun, method=method)

    info: dict[str, Any] = dict(
        n_feval=sol_result.nfev, success=sol_result.success, n_jaceval=0, n_lu=0
    )
    return sol_result.x, info


def numerical_jacobian(x, f, delta):
    jac = np.empty((x.shape[0], x.shape[0]))
    for j in range(x.shape[0]):
        shift = np.zeros_like(x)
        shift[j] = delta
        jac[:, j] = (f(x + shift) - f(x)) / delta
    return jac


controller_I: Callable[[float, float, float], float] = (
    lambda err_ratio, err_ratio_last, p: err_ratio ** (-1 / p)
)
controller_PI: Callable[[float, float, float, float], float] = (
    lambda err_ratio, err_ratio_last, alpha, beta: err_ratio ** (-alpha)
    * err_ratio_last**beta
)

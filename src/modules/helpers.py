from typing import Any, Callable
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root

norm_hairer: Callable[[NDArray[np.floating]], float] = (
    lambda x: (np.sum(x**2) / x.size) ** 0.5
)

clip: Callable[[float, float, float], float] = lambda x, x_min, x_max: min(
    max(x, x_min), x_max
)  # runs faster then np.clip since we do  not deal with arrays


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


def numerical_jacobian(
    x: NDArray[np.floating],
    f: Callable[[NDArray[np.floating]], NDArray[np.floating]],
    delta: float,
) -> NDArray[np.floating]:
    jac = np.empty((x.shape[0], x.shape[0]))
    f_x = f(x)
    for j in range(x.shape[0]):
        shift = np.zeros_like(x)
        shift[j] = delta
        jac[:, j] = (f(x + shift) - f_x) / delta
    return jac


def numerical_jacobian_t(
    x: NDArray[np.floating],
    t: float,
    f: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
    delta: float,
) -> NDArray[np.floating]:
    jac = np.empty((x.shape[0], x.shape[0]))
    f_x = f(t, x)
    for j in range(x.shape[0]):
        shift = np.zeros_like(x)
        shift[j] = delta
        jac[:, j] = (f(t, x + shift) - f_x) / delta
    return jac

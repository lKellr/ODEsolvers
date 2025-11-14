import logging
from typing import Any, Callable
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import lu_factor, lu_solve

from solvers.helpers import norm_hairer, numerical_jacobian

logger = logging.getLogger(__name__)


def Secant_method(f: Callable[[float], float], a: float, b: float, tol: float) -> float:
    """searches for a root of the function f in the interval [a, b]"""
    x0 = a  # starting value 1
    x1 = b  # starting value 2

    f0 = f(x0)
    f1 = f(x1)
    dx = -f1 / (f1 - f0) * (x1 - x0)
    x1 = x1 + dx

    while abs(f1) > tol:
        f0 = f1
        f1 = f(x1)
        dx = -f1 / (f1 - f0) * dx
        x1 = x1 + dx

    return x1


def Bisection(f: Callable[[float], float], a: float, b: float, tol: float) -> float:
    """searches for a root of the function f in the interval [a, b]"""
    x0 = a
    x1 = b

    f0 = f(x0)
    f1 = f(x1)

    if np.sign(f0) == np.sign(f1):
        print("invalid starting values")
        return np.nan

    xm = 0.5 * (x0 + x1)

    while abs(f1) > tol:
        fm = f(xm)
        if np.sign(fm) == np.sign(f0):
            f0 = fm
        else:
            f1 = fm
        xm = 0.5 * (x0 + x1)

    return xm


def Newton(
    fun: Callable[[NDArray[np.floating]], NDArray[np.floating]],
    x0: NDArray[np.floating],
    jac_fun: Callable[[NDArray[np.floating]], NDArray[np.floating]] | None,
    tol: float | NDArray[np.floating],
    max_iter: int = 20,
    jac_freq: int = 5,
    norm: Callable[[NDArray[np.floating]], float] = norm_hairer,
) -> tuple[NDArray[np.floating], dict[str, Any]]:
    info: dict[str, Any] = dict(n_feval=0, n_jaceval=1, n_lu=1, success=True)

    if jac_fun is None:
        jac_fun = lambda x: numerical_jacobian(x, fun, delta=1e-8)
    else:
        jac_fun = jac_fun

    jac = jac_fun(x0)
    lu, piv = lu_factor(jac)

    x = np.copy(x0)
    delta_x = np.ones_like(x0) * np.inf
    iter = 0
    while norm(delta_x / (tol * x + tol)) > 1:
        delta_x_last = delta_x
        delta_x = lu_solve((lu, piv), -fun(x), overwrite_b=True, check_finite=False)
        x += delta_x

        conv_rate = norm(delta_x) / norm(delta_x_last)
        iter += 1
        info["n_feval"] += 1

        if conv_rate >= 1 or iter >= max_iter:
            info["success"] = False
            logger.warning(f"Ended Newton in step: {iter}, conv_rate: {conv_rate}")
            break

        if (
            iter % jac_freq == 0 or conv_rate >= 0.5
        ):  # TODO: is the second criterion a good idea? which value should i choose?
            jac = jac_fun(x)
            lu, piv = lu_factor(jac)
            info["n_jaceval"] += 1
            info["n_lu"] += 1
            logger.debug(
                f"Recomputing Jacobian in step: {iter}, conv_rate: {conv_rate}"
            )
    logger.debug(f"Finished Newton after: {iter} iterations")
    return x, info


def NewtonIRK(
    fun: Callable[[NDArray[np.floating]], NDArray[np.floating]],
    U0: NDArray[np.floating],
    jac_fun: Callable[[NDArray[np.floating]], NDArray[np.floating]] | None,
    tol: float | NDArray[np.floating],
    max_iter: int = 20,
    jac_freq: int = 0,
    norm: Callable[[NDArray[np.floating]], float] = norm_hairer,
) -> tuple[float, dict[str, Any]]:
    info: dict[str, Any] = dict(n_feval=0, n_jaceval=1, n_lu=1, success=True)

    if jac_fun is None:
        jac_fun = lambda x: numerical_jacobian(x, fun, delta=1e-8)
    else:
        jac_fun = jac_fun

    jac = jac_fun(U0)
    lu, piv = lu_factor(jac)

    U = np.copy(U0)
    delta_U = np.zeros_like(U0)
    eta = max(eta_old, U_rounding) ** 0.8
    iter = 0
    while eta * delta_U > kappa * tol:
        delta_U_last = delta_U
        delta_U = -lu_solve((lu, piv), fun(U), overwrite_b=True, check_finite=False)
        U += delta_U

        conv_rate = norm(delta_U) / norm(delta_U_last)
        eta = conv_rate / (1.0 - conv_rate)
        iter += 1
        info["n_feval"] += 1

        if conv_rate >= 1 or iter >= max_iter:
            info["success"] = False
            break

        if (
            iter % jac_freq == 0 or conv_rate >= 0.8
        ):  # TODO: is the second criterion a good idea?
            jac = jac_fun(U)
            lu, piv = lu_factor(jac)
            info["n_jaceval"] += 1
            info["n_lu"] += 1

    return U, info


# TODO: evaluate jacobian inside or outside of this function? -> probably give frequency

import logging
from typing import Any, Callable
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import lu_factor, lu_solve

from modules.helpers import norm_hairer, numerical_jacobian

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
    tol_iter: float,
    max_iter: int = 20,
    jac_freq: int = 5,
    norm: Callable[[NDArray[np.floating]], float] = norm_hairer,
    **kwargs,
) -> tuple[NDArray[np.floating], bool, dict[str, Any]]:
    success = True
    info: dict[str, Any] = dict(
        n_feval=0, n_jaceval=1, n_lu=1, stop_reason="", eta=None, error=None
    )

    if jac_fun is None:
        jac_fun = lambda x: numerical_jacobian(x, fun, delta=1e-8)
    else:
        jac_fun = jac_fun

    jac = jac_fun(x0)
    lu, piv = lu_factor(jac)

    x = np.copy(x0)
    norm_delta = np.inf
    norm_delta_last = np.inf
    iter = 0
    while norm_delta > tol_iter:
        delta = lu_solve((lu, piv), -fun(x), overwrite_b=True, check_finite=False)
        x += delta

        norm_delta = norm(delta)
        conv_rate = norm_delta / norm_delta_last
        if conv_rate >= 1:
            success = False
            logger.warning(
                f"Divergence of Newton in iteration {iter+1}, conv_rate = {conv_rate}"
            )
            info["stop_reason"] = "slow convergence"
            break
        elif iter >= max_iter:
            success = False
            logger.warning(f"Maximum number of steps {max_iter} reached")
            info["stop_reason"] = "max steps reached"
            break
        if (
            iter + 1
        ) % jac_freq == 0 or conv_rate >= 0.5:  # TODO: is the second criterion a good idea? which value should i choose?
            jac = jac_fun(x)
            lu, piv = lu_factor(jac)
            info["n_jaceval"] += 1
            info["n_lu"] += 1
            logger.debug(
                f"Recomputing Jacobian in iteration {iter+1}, conv_rate: {conv_rate}"
            )
        iter += 1
        norm_delta_last = norm_delta

    if success:
        info["stop_reason"] = "success"
        logger.debug(f"Finished Newton after {iter} iterations with success")
    info["n_feval"] = iter
    info["error"] = norm_delta
    return x, success, info


def NewtonODE(
    fun: Callable[[NDArray[np.floating]], NDArray[np.floating]],
    x0: NDArray[np.floating],
    jac_fun: Callable[[NDArray[np.floating]], NDArray[np.floating]] | None,
    tol_iter: float,  # iteration error; should be discretization error * 1e-1 - 1e-2
    eta_old: float,
    max_iter: int = 20,
    jac_freq: int = 100,
    norm: Callable[[NDArray[np.floating]], float] = norm_hairer,
) -> tuple[NDArray[np.floating], bool, dict[str, Any]]:
    """Version of Newtons method with stopping criteria adapted to ODE solvers"""

    success = True
    info: dict[str, Any] = dict(
        n_feval=0, n_jaceval=1, n_lu=1, stop_reason="", eta=None, error=None
    )

    if jac_fun is None:
        jac_fun = lambda x: numerical_jacobian(x, fun, delta=1e-8)
    else:
        jac_fun = jac_fun

    jac = jac_fun(x0)
    lu, piv = lu_factor(jac)

    x = np.copy(x0)
    norm_delta = np.nan
    norm_delta_last = np.nan
    eta = max(eta_old, np.finfo(x0.dtype).eps) ** 0.8
    conv_rate = 0.0
    iter = 0
    while True:
        delta = -lu_solve((lu, piv), fun(x), overwrite_b=True, check_finite=False)
        x += delta

        norm_delta = norm(delta)
        if iter > 0:
            conv_rate = norm_delta / norm_delta_last  # small values are better
            eta = conv_rate / (1.0 - conv_rate)

        if eta * norm_delta < tol_iter:  # normal convergence check
            success = True
            info["stop_reason"] = "success"
            logger.debug(f"Finished Newton after {iter+1} iterations with success")
            break
        elif conv_rate >= 1:  # check for divergence
            success = False
            info["stop_reason"] = "divergence"
            logger.warning(
                f"Divergence of Newton in iteration {iter+1}, conv_rate = {conv_rate}"
            )
            break
        elif (
            eta * conv_rate ** (max_iter - iter - 1) * norm_delta
            > tol_iter  # TODO: first iteration
        ):  # check for max iters, with estimate if convergence speed is fast enough to reach convergence in max iters
            success = False
            info["stop_reason"] = "slow convergence"
            logger.warning(
                f"Too slow convergence of Newton in iteration {iter+1} for k_max = {max_iter}, conv_rate = {conv_rate}"
            )
            break

        # recomputing the jacobian
        if (iter + 1) % jac_freq == 0 or (
            conv_rate >= 0.5 and iter > 0
        ):  # TODO: is the second criterion a good idea?
            jac = jac_fun(x)
            lu, piv = lu_factor(jac)
            info["n_jaceval"] += 1
            info["n_lu"] += 1
            logger.debug(
                f"Recomputing Jacobian in iteration {iter+1}, conv_rate = {conv_rate}"
            )
        iter += 1
        norm_delta_last = norm_delta
    info["n_feval"] = iter + 1
    info["eta"] = eta
    info["error"] = norm_delta
    return x, success, info

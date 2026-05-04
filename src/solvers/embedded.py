import logging
from numpy._typing._array_like import NDArray
from numpy import copy, floating
from typing import Any, Callable
import numpy as np
from numpy.typing import NDArray
from modules.helpers import norm_hairer
from modules.step_control import (
    get_default_PI_parameters,
    StepControllerPI,
    get_default_PI_parameters_rejected,
)

logger = logging.getLogger(__name__)


def DP54(
    ode_fun: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
    x0: NDArray[np.floating],
    t_max: float,
    t0: float = 0.0,
    h0: float | None = None,
    **step_controller_kwargs: dict[str, Any],
) -> tuple[NDArray[np.floating], NDArray[np.floating], dict[str, Any]]:
    """Dormand Prince 5(4) Method, MATLAB ode45 solver"""
    info: dict[str, Any] = dict(
        n_feval=0,
        n_jaceval=0,
        n_lu=0,
        n_restarts=0,
        local_errors=[],
    )
    if "control_params" not in step_controller_kwargs.keys():
        step_controller_kwargs["control_params"] = get_default_PI_parameters(5)
    if "control_params_rejected" not in step_controller_kwargs.keys():
        step_controller_kwargs["control_params_rejected"] = (
            get_default_PI_parameters_rejected(5)
        )
    step_controller = StepControllerPI(**step_controller_kwargs)

    if h0 is None:
        h = step_controller.get_initial_stepHW(ode_fun, x0, t0=t0, p=5)
    else:
        h = h0

    t = [t0]
    x = [x0]
    t_crit: list[float] = []
    x_crit: list[NDArray[np.floating]] = []

    ix_step = 0
    k1 = ode_fun(t0, x0)  # FSAL property
    while t[ix_step] < t_max:  # iterate until t_max is reached
        if t[ix_step] + h > t_max:  # shorten h if we would go further than necessary
            h = t_max - t[ix_step]
        t_pred = t[ix_step] + h

        # calulate k's
        # k1 = ode_fun(t[ix_step], x[ix_step])
        k2 = ode_fun(t[ix_step] + 1 / 5 * h, x[ix_step] + 1 / 5 * h * k1)
        k3 = ode_fun(t[ix_step] + 3 / 10 * h, x[ix_step] + h * (3 * k1 + 9 * k2) / 40)
        k4 = ode_fun(
            t[ix_step] + 4 / 5 * h,
            x[ix_step] + h * (44 / 45 * k1 - 56 / 15 * k2 + 32 / 9 * k3),
        )
        k5 = ode_fun(
            t[ix_step] + 8 / 9 * h,
            x[ix_step]
            + h
            * (
                19372 / 6561 * k1
                - 25360 / 2187 * k2
                + 64448 / 6561 * k3
                - 212 / 729 * k4
            ),
        )
        k6 = ode_fun(
            t[ix_step] + h,
            x[ix_step]
            + h
            * (
                9017 / 3168 * k1
                - 355 / 33 * k2
                + 46732 / 5247 * k3
                + 49 / 176 * k4
                - 5103 / 18656 * k5
            ),
        )

        x_pred = x[ix_step] + h * (
            35 / 384 * k1
            + 500 / 1113 * k3
            + 125 / 192 * k4
            - 2187 / 6784 * k5
            + 11 / 84 * k6
        )  # embedding, as ode_fun(x_pred)==k2
        k2 = ode_fun(
            t_pred, x_pred
        )  # reusing k2 instead of new variable k7, as it is not used again

        err = h * (
            71 / 57600 * k1
            - 71 / 16695 * k3
            + 71 / 1920 * k4
            - 17253 / 339200 * k5
            + 22 / 525 * k6
            - 1 / 40 * k2
        )

        h, accepted = step_controller.evaluate_step(h, err, x[ix_step], x_pred)

        if accepted:  # accept result if tolerance is met, or we cant decrease h anymore
            t.append(t_pred)  # NOTE: at this point t_pred != t[ix_step] + h
            x.append(x_pred)
            k1 = k2
            ix_step += 1
            info["local_errors"].append(step_controller.norm(err))
        else:
            t_crit.append(t_pred)
            x_crit.append(x_pred)
    info["n_feval"] = 1 + (len(t) - 1 + len(t_crit)) * 6
    info["n_restarts"] = len(t_crit)
    info["restarts"] = (t_crit, x_crit)
    print(
        "Finished\n"
        + "\n".join(
            [
                f"{k}: {v}"
                for k, v in info.items()
                if not isinstance(v, list) and not isinstance(v, tuple)
            ]
        )
        + "\n"
    )
    return np.array(t), np.array(x), info


def BS32(
    ode_fun: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
    x0: NDArray[np.floating],
    t_max: float,
    t0: float = 0.0,
    h0: float | None = None,
    **step_controller_kwargs: dict[str, Any],
) -> tuple[NDArray[np.floating], NDArray[np.floating], dict[str, Any]]:
    """Bogacki–Shampine Method, MATLAB ode23 solver"""

    info: dict[str, Any] = dict(
        n_feval=0,
        n_jaceval=0,
        n_lu=0,
        n_restarts=0,
        local_errors=[],
    )

    if "control_params" not in step_controller_kwargs.keys():
        step_controller_kwargs["control_params"] = get_default_PI_parameters(3)
    if "control_params_rejected" not in step_controller_kwargs.keys():
        step_controller_kwargs["control_params_rejected"] = (
            get_default_PI_parameters_rejected(3)
        )
    step_controller = StepControllerPI(**step_controller_kwargs)

    if h0 is None:
        h = step_controller.get_initial_stepHW(ode_fun, x0, t0=t0, p=3)
    else:
        h = h0

    t = [t0]
    x = [x0]
    t_crit: list[float] = []
    x_crit: list[NDArray[np.floating]] = []

    ix_step = 0
    k1 = ode_fun(t[0], x[0])  # FSAL
    while t[ix_step] < t_max:  # iterate until t_max is reached
        if t[ix_step] + h > t_max:  # shorten h if we would go further than necessary
            h = t_max - t[ix_step]

        t_pred = t[ix_step] + h
        # calulate k's
        k2 = ode_fun(t[ix_step] + 1 / 2 * h, x[ix_step] + 1 / 2 * h * k1)
        k3 = ode_fun(t[ix_step] + 3 / 4 * h, x[ix_step] + 3 / 4 * h * k2)
        x_pred = x[ix_step] + h * (2 * k1 + 3 * k2 + 4 * k3) / 9
        k4 = ode_fun(t_pred, x_pred)

        err = h * (-5 / 72 * k1 + 1 / 12 * k2 + 1 / 9 * k3 - 1 / 8 * k4)
        # h*np.linalg.norm(71/57600*k1-71/16695*k3+71/1920*k4-17253/339200*k5+22/525*k6-1/40*k2, ord=np.inf) #local error:||x_pred-X_pred7||

        h, accepted = step_controller.evaluate_step(h, err, x[ix_step], x_pred)

        if accepted:  # accept result if tolerance is met, or we cant decrease h anymore
            t.append(t_pred)
            x.append(x_pred)
            k1 = k4
            ix_step += 1
            info["local_errors"].append(step_controller.norm(err))
        else:
            t_crit.append(t_pred)
            x_crit.append(x_pred)
    info["n_feval"] = 1 + (len(t) - 1 + len(t_crit)) * 3
    info["n_restarts"] = len(t_crit)
    info["restarts"] = (t_crit, x_crit)
    return np.array(t), np.array(x), info


def RKX4(
    ode_fun: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
    x0: NDArray[np.floating],
    t_max: float,
    t0: float = 0.0,
    h0: float | None = None,
    extrap_step_ratio: int = 2,
    **step_controller_kwargs: dict[str, Any],
) -> tuple[NDArray[np.floating], NDArray[np.floating], dict[str, Any]]:
    """Classical Runge-Kutta Method with step control. Instead of using an embedded stage, this method approximates the error by a single step of Richardson extrapolation"""

    info: dict[str, Any] = dict(
        n_feval=0,
        n_jaceval=0,
        n_lu=0,
        n_restarts=0,
        local_errors=[],
    )

    if "control_params" not in step_controller_kwargs.keys():
        step_controller_kwargs["control_params"] = get_default_PI_parameters(5)
    if "control_params_rejected" not in step_controller_kwargs.keys():
        step_controller_kwargs["control_params_rejected"] = (
            get_default_PI_parameters_rejected(5)
        )
    step_controller = StepControllerPI(**step_controller_kwargs)

    if h0 is None:
        h = step_controller.get_initial_stepHW(ode_fun, x0, t0=t0, p=5)
    else:
        h = h0

    def do_RK4(
        x_i: NDArray[np.floating], t_i: float, h: float, f_i: NDArray[np.floating]
    ):
        k1 = f_i
        k2 = ode_fun(t_i + 0.5 * h, x_i + 0.5 * h * k1)
        k3 = ode_fun(t_i + 0.5 * h, x_i + 0.5 * h * k2)
        k4 = ode_fun(t_i + h, x_i + h * k3)
        return x_i + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    t = [t0]
    x = [x0]
    t_crit: list[float] = []
    x_crit: list[NDArray[np.floating]] = []

    ix_step = 0
    while t[ix_step] < t_max:  # iterate until t_max is reached
        if t[ix_step] + h > t_max:  # shorten h if we would go further than necessary
            h = t_max - t[ix_step]

        t_pred = t[ix_step] + h

        f_i = ode_fun(t[ix_step], x[ix_step])

        x_coarse = do_RK4(x[ix_step], t[ix_step], h, f_i)

        x_fine = do_RK4(x[ix_step], t[ix_step], h / extrap_step_ratio, f_i)
        for j in range(1, extrap_step_ratio):
            x_fine = do_RK4(
                x_fine,
                t[ix_step] + j * h / extrap_step_ratio,
                h / extrap_step_ratio,
                ode_fun(t[ix_step] + j * h / extrap_step_ratio, x_fine),
            )

        x_extrap = x_fine + (x_fine - x_coarse) / (extrap_step_ratio**4 - 1.0)

        err = (
            x_fine - x_extrap
        )  # NOTE: alternatively: err = x_fine - x_coarse (better error estimate for lower error, suggested by NR, Deuflhard and H&W prefer subdiagonal, thats why I am using it)
        x_pred = x_extrap  # NOTE: continue with higher order estimate at cost of worse error estimation (which is not that great anyway)
        h, accepted = step_controller.evaluate_step(h, err, x[ix_step], x_pred)

        if accepted:  # accept result if tolerance is met, or we cant decrease h anymore
            t.append(t_pred)
            x.append(x_pred)
            ix_step += 1
            info["local_errors"].append(step_controller.norm(err))
        else:
            t_crit.append(t_pred)
            x_crit.append(x_pred)
    info["n_feval"] = (len(t) - 1 + len(t_crit)) * 11
    info["n_restarts"] = len(t_crit)
    info["restarts"] = (t_crit, x_crit)
    return np.array(t), np.array(x), info

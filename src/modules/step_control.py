from typing import Any, Callable, NamedTuple
import numpy as np
from numpy.typing import NDArray
from modules.helpers import norm_hairer, clip
import logging

logger = logging.getLogger(__name__)


class ControllerParams(NamedTuple):
    coeff_i: float
    coeff_p: float = 0.0
    s_limits: tuple[float, float] = (0.2, 5.0)

    @property
    def alpha(self) -> float:
        return self.coeff_i + self.coeff_p

    @property
    def beta(self) -> float:
        return self.coeff_p


def get_PI_parameters(p: int) -> ControllerParams:
    return ControllerParams(coeff_i=0.3 / p, coeff_p=0.4 / p, s_limits=(0.2, 5.0))


def get_PI_parameters_rejected(p: int) -> ControllerParams:
    return ControllerParams(coeff_i=1.0 / p, coeff_p=0.0, s_limits=(0.2, 1.0))


def get_step_PI(err_ratio, err_ratio_last, control_params):
    return clip(
        (1.0 / err_ratio) ** control_params.alpha * err_ratio_last**control_params.beta,
        control_params.s_limits[0],
        control_params.s_limits[1],
    )


class StepController:
    """PI step size controller"""

    def __init__(
        self,
        control_params: ControllerParams,
        control_params_rejected: ControllerParams,
        atol: float | NDArray[np.floating] = 10**-5,
        rtol: float | NDArray[np.floating] = 10**-3,
        norm: Callable[[NDArray[np.floating]], float] = norm_hairer,
        safety_tol: float = (
            0.9  # is just a modifier for tolerance (after scaling by PI parameters)
        ),
        step_rejection_limit: float = 1.2,
        s_deadzone: tuple[float, float] = (
            1.0,
            1.0,
        ),
        h_limits: tuple[float, float] = (
            0,
            np.inf,
        ),
    ) -> None:
        self.atol = atol
        self.rtol = rtol
        self.norm = norm
        self.safety_tol = safety_tol

        self.control_params_accepted = control_params
        self.control_params_rejected = control_params_rejected

        self.step_rejection_limit = step_rejection_limit
        self.s_deadzone = s_deadzone
        self.h_limits = h_limits

        self.err_ratio_prev = 1.0
        self.is_retry = False
        self.prev_step_size = float("nan")

    def get_initial_stepFhty(
        self,
        ode_fun: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
        x0: NDArray[np.floating],
        t_max: float,
        p: int,
        t0: float = 0.0,
    ) -> float:
        """From the Flaherty lecture notes"""
        tol = self.atol + self.rtol * np.abs(x0)

        step_size0 = (
            self.norm(tol) / (1 / (t_max - t0) ** p + self.norm(ode_fun(t0, x0)) ** p)
        ) ** (1 / p)
        return step_size0

    def get_initial_stepHW(
        self,
        ode_fun: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
        x0: NDArray[np.floating],
        p: int,
        t0: float = 0.0,
    ) -> float:
        """From Hairer & Wanner eq. 4.14"""
        tol = self.atol + self.rtol * np.abs(x0)

        f0 = ode_fun(t0, x0)
        d0 = self.norm(x0 / tol)
        d1 = self.norm(f0 / tol)

        h0: float
        if d0 < 1e-5 or d1 < 1e-5:
            h0 = 1e-6
        else:
            h0 = 0.01 * (d0 / d1)

        U1_Eul = x0 + h0 * ode_fun(t0, x0)
        d2 = self.norm((ode_fun(t0 + h0, U1_Eul) - f0) / tol) / h0

        h1 = (0.01 / max(d1, d2)) ** (1 / (p + 1))
        if max(d1, d2) <= 1e-15:
            h1 = max(1e-6, h0 * 1e-3)

        return min(100 * h0, h1)

    def _get_error_ratio(
        self,
        error: NDArray[np.floating],
        x_prev: NDArray[np.floating],
        x_pred: NDArray[np.floating],
    ) -> float:
        # if (
        #     error == 0.0
        # ).all():  # for linear solutions, estimated error can be exactly zero; NOTE: this check is really expensive
        #     return float(np.finfo(error.dtype).max)

        tol = self.atol + self.rtol * np.maximum(np.abs(x_prev), np.abs(x_pred))
        err_ratio = self.norm(error / tol) / self.safety_tol
        return err_ratio

    def evaluate_step(
        self,
        tried_step_size: float,
        error: NDArray[np.floating],
        x_prev: NDArray[np.floating],
        x_pred: NDArray[np.floating],
    ) -> tuple[float, bool]:
        err_ratio = self._get_error_ratio(error, x_prev, x_pred)

        accepted: bool = (
            err_ratio <= self.step_rejection_limit
            or tried_step_size <= self.h_limits[0]
        )
        if accepted and err_ratio > self.step_rejection_limit:
            logger.warning(
                f"Accepting step with too large error {err_ratio} since further step size decrease from h = {tried_step_size} is not possible."
            )

        step_fac: float
        if accepted:
            logger.debug(
                msg=f"Accepting step"
                + (" with retry correction" if self.is_retry else "")
            )
            step_fac = get_step_PI(
                err_ratio, self.err_ratio_prev, self.control_params_accepted
            )
            # correction if the previous step has been rejected: multiply by ratio of tried step to last succesful step (Gustafsson1991)
            if self.is_retry:
                step_fac *= (  # TODO: first step self.prev_step_size is not initialized!
                    tried_step_size / self.prev_step_size
                )
                self.is_retry = False
            self.err_ratio_prev = err_ratio
            self.prev_step_size = tried_step_size * step_fac # NOTE: without deadzone and clipping, this should not be problematic since we use it just for improving rejected estiamtes
        else:
            logger.debug(
                msg=f"Rejecting step with error {err_ratio}, h = {tried_step_size}"
            )
            step_fac = get_step_PI(
                err_ratio, self.err_ratio_prev, self.control_params_rejected
            )
            self.is_retry = True

        next_step_size: float
        if (
            step_fac > self.s_deadzone[0] and step_fac < self.s_deadzone[1]
        ):  # use a deadzone
            next_step_size = tried_step_size
        else:
            next_step_size = clip(tried_step_size * step_fac, self.h_limits[0], self.h_limits[1])

        return next_step_size, accepted

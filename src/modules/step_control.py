from typing import Any, Callable
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root
from modules.helpers import norm_hairer


class step_controller:
    def __init__(
        self,
        atol: float | NDArray[np.floating] = 10**-5,
        rtol: float | NDArray[np.floating] = 10**-3,
        norm: Callable[[NDArray[np.floating]], float] = norm_hairer,
        safety: float = 0.8,
        s_limits: tuple[float, float] = (0.2, 5.0),
        s_deadzone: tuple[float, float] = (0.95, 1.05),
        h_limits: tuple[float, float] = (0, np.inf),
    ) -> None:
        self.atol = atol
        self.rtol = rtol
        self.norm = norm
        self.safety = safety
        self.s_limits = s_limits
        self.s_deadzone = s_deadzone
        self.h_limits = h_limits

        self.err_ratio_last = (
            1.0  # TODO: this is not sufficient to force I-behaviour in first step
        )

    def get_initial_step(self) -> float:
        tol = atol + rtol * np.abs(x0)

        h = (
            norm(tol) / (1 / (t_max - t0) ** (1 / 5) + norm(f(t0, x0)) ** (1 / 5))
        ) ** (
            1 / 5
        )  # TODO: check exponents, step_controller.h0
        return h

    def get_step(self, error) -> tuple[float, bool]:
        tol = atol + rtol * np.maximum(np.abs(x[iter]), np.abs(x_pred))
        err_ratio = norm(err / tol)  # TODO: h required?
        # TODO: divide by zero errors

        # TODO: init err_ratio
        # TODO: beta

        s = step_safety * step_controller(err_ratio, err_ratio_last)

        s = np.clip(
            s,
            s_limits[0],
            s_limits[1],
        )  # s gets clipped to prevent to extreme changes of h, also err might become zero
        if s > s_deadzone[0] and s < s_deadzone[1]:  # use a deadzone
            s = 1.0

        accepted = err_ratio <= 1 or next_step_size <= h_limits[0]
        if err_ratio > 1:
            logger.warn(
                f"Accepting step with too large error {err_ratio} since further step size decrease from h = {h} is not possible."
            )
        next_step_size = np.clip(current_step_size * s, h_limits[0], h_limits[1])

        if accepted:
            err_ratio_last = err_ratio
        else:
            # TODO: cahnge parametes for next step
            logger.debug(msg=f"Rejecting step {iter} with error {err_ratio}")
        return next_step_size, accepted


controller_I: Callable[[float, float, float], float] = (
    lambda err_ratio, err_ratio_last, p: err_ratio ** (-1 / p)
)
controller_PI: Callable[[float, float, float, float], float] = (
    lambda err_ratio, err_ratio_last, alpha, beta: err_ratio ** (-alpha)
    * err_ratio_last**beta
)

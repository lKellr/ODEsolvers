from abc import ABC, abstractmethod
from csv import Error
from typing import Any, Callable, Literal, NamedTuple, override
import numpy as np
from numpy.typing import DTypeLike, NDArray
from modules.helpers import norm_hairer, clip
import logging

logger = logging.getLogger(__name__)

contr_ext_state_type = Literal[
    "accepted", "continue", "too_slow_convergence", "divergence"
]


class ControllerPIParams(NamedTuple):
    coeff_i: float
    coeff_p: float = 0.0
    s_limits: tuple[float, float] = (0.2, 5.0)

    @property
    def alpha(self) -> float:
        return self.coeff_i + self.coeff_p

    @property
    def beta(self) -> float:
        return self.coeff_p


def get_default_PI_parameters(p: int) -> ControllerPIParams:
    return ControllerPIParams(coeff_i=0.3 / p, coeff_p=0.4 / p, s_limits=(0.2, 5.0))


def get_default_PI_parameters_rejected(p: int) -> ControllerPIParams:
    return ControllerPIParams(coeff_i=1.0 / p, coeff_p=0.0, s_limits=(0.2, 1.0))


def get_step_PI(
    err_ratio: float, err_ratio_last: float, control_params: ControllerPIParams
) -> float:
    return clip(
        (1.0 / err_ratio) ** control_params.alpha * err_ratio_last**control_params.beta,
        control_params.s_limits[0],
        control_params.s_limits[1],
    )


class StepController(ABC):
    def __init__(
        self,
        atol: float | NDArray[np.floating],
        rtol: float | NDArray[np.floating],
        norm: Callable[[NDArray[np.floating]], float],
        safety_tol: float,  # is just a modifier for tolerance (after scaling by PI parameters)
    ):
        self.atol = atol
        self.rtol = rtol
        self.norm = norm
        self.safety_tol = safety_tol

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
        x_curr: NDArray[np.floating],
        x_pred: NDArray[np.floating],
    ) -> float:
        # if (
        #     error == 0.0
        # ).all():  # for linear solutions, estimated error can be exactly zero; NOTE: this check is really expensive
        #     return float(np.finfo(error.dtype).max)

        tol = self.atol + self.rtol * np.maximum(np.abs(x_curr), np.abs(x_pred))
        err_ratio = self.norm(error / tol) / self.safety_tol
        return err_ratio


class StepControllerPI(StepController):
    """PI step size controller"""

    def __init__(
        self,
        control_params: ControllerPIParams,
        control_params_rejected: ControllerPIParams,
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
        super().__init__(atol, rtol, norm, safety_tol)

        self.control_params_accepted = control_params
        self.control_params_rejected = control_params_rejected

        self.step_rejection_limit = step_rejection_limit
        self.s_deadzone = s_deadzone
        self.h_limits = h_limits

        self.err_ratio_prev = 1.0
        self.prev_step_size = float("nan")
        self.is_retry = False

    def evaluate_step(
        self,
        tried_step_size: float,
        error: NDArray[np.floating],
        x_curr: NDArray[np.floating],
        x_pred: NDArray[np.floating],
    ) -> tuple[float, bool]:
        err_ratio = self._get_error_ratio(error, x_curr, x_pred)

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
                step_fac *= tried_step_size / self.prev_step_size
                self.is_retry = False
            self.err_ratio_prev = err_ratio
            self.prev_step_size = (
                tried_step_size * step_fac
            )  # NOTE: without deadzone and clipping, this should not be problematic since we use it just for improving rejected estimates
        else:
            logger.debug(
                msg=f"Rejecting step with error {err_ratio}, h = {tried_step_size}"
            )
            step_fac = get_step_PI(
                err_ratio, self.err_ratio_prev, self.control_params_rejected
            )
            self.is_retry = True
            if np.isnan(
                self.prev_step_size
            ):  # first step self.prev_step_size is not initialized!
                self.prev_step_size = tried_step_size

        next_step_size: float
        if (
            step_fac > self.s_deadzone[0] and step_fac < self.s_deadzone[1]
        ):  # use a deadzone
            next_step_size = tried_step_size
        else:
            next_step_size = clip(
                tried_step_size * step_fac, self.h_limits[0], self.h_limits[1]
            )

        return next_step_size, accepted


class StepControllerExtrap(StepController, ABC):

    def __init__(
        self,
        atol: float | NDArray[np.floating] = 10**-8,
        rtol: float | NDArray[np.floating] = 10**-5,
        norm: Callable[[NDArray[np.floating]], float] = norm_hairer,
        safety_unscaled: float = (0.94),
        safety_tol: float = (0.65),
        s_limits_scaled: tuple[float, float] = (0.02, 4.0),
        step_multiplier_divergence: float = 0.5,
        dtype: DTypeLike = np.double,
    ) -> None:
        super().__init__(atol, rtol, norm, safety_tol)

        self.safety_unscaled = safety_unscaled
        self.s_limits_scaled = s_limits_scaled
        self.step_multiplier_divergence = step_multiplier_divergence

        self.dtype = dtype

    def initialize_scheme(
        self,
        is_symmetric: bool,
        table_size: int,
        err_reduction_at_step: NDArray[np.floating],
        total_feval_cost_for_k: NDArray[np.floating],
        check_window: tuple[int, int],
    ):
        self.is_symmetric = is_symmetric
        self.table_size = table_size
        self.err_reduction_at_step = err_reduction_at_step
        self.total_feval_cost_for_k = total_feval_cost_for_k
        self.check_window = check_window

    def get_initial_ktarget(self) -> int:
        """very rough estimate from numerical recipes, can be motivated for example from Hairer&Wanner Fig.9.5"""
        if self.is_symmetric:
            log_fact = -max(-12.0, np.log10(self.rtol)) * 0.6 + 0.5
        else:
            log_fact = -np.log10(self.rtol + self.atol) * 0.6 + 0.5
        k_target = max(1, min(self.table_size - 1, int(log_fact)))
        return k_target

    def _get_step_mult_opt(self, err_ratio_k: float, next_k: int) -> float:
        """returns the optimal factor by which the step should be multiplied to reach the prescribed tolerance levels"""
        order_exponent = (
            1 / (2 * next_k + 1) if self.is_symmetric else next_k + 1  #
        )  # NOTE: 2*k+1 since k starts at zero. Hairer&Wanner use 2*k-1 for k starting with 1
        temp_s_limit_descaled = self.s_limits_scaled[0] ** order_exponent

        if err_ratio_k == 0:
            s_opt = 1 / temp_s_limit_descaled
        else:
            s_opt = (
                self.safety_unscaled * (self.safety_tol / err_ratio_k) ** order_exponent
            )
            s_opt = clip(
                s_opt,
                temp_s_limit_descaled / self.s_limits_scaled[1],
                1 / temp_s_limit_descaled,
            )
        return s_opt

    @abstractmethod
    def evaluate_step(
        self,
        error: NDArray[np.floating],
        x_curr: NDArray[np.floating],
        x_pred: NDArray[np.floating],
        k_curr: int,
        k_target: int,
        allow_early_check: bool = False,
    ) -> contr_ext_state_type:
        raise NotImplementedError()

    @abstractmethod
    def get_most_efficient_parameters(
        self,
        k_final: int,
        allow_order_increase: bool,
    ) -> tuple[int, float]:
        raise NotImplementedError()


class StepControllerExtrapKH(StepControllerExtrap):
    """Combined order and step size (k-h) controller for extrapolation methods. Following the strategy layed out in Hairer&Wanner and modified in Numerical Recipes. Originally proposed by Deulfhard"""

    def __init__(
        self,
        atol: float | NDArray[np.floating] = 10**-8,
        rtol: float | NDArray[np.floating] = 10**-5,
        norm: Callable[[NDArray[np.floating]], float] = norm_hairer,
        safety_unscaled: float = (0.94),
        safety_tol: float = (0.65),
        s_limits_scaled: tuple[float, float] = (0.02, 4.0),
        step_multiplier_divergence: float = 0.5,
        dtype: DTypeLike = np.double,
        work_order_limits: tuple[float, float] = (0.8, 0.9),
    ) -> None:
        super().__init__(
            atol,
            rtol,
            norm,
            safety_unscaled,
            safety_tol,
            s_limits_scaled,
            step_multiplier_divergence,
            dtype,
        )

        assert (
            1.0 / work_order_limits[1] > work_order_limits[0]
        ), f"Invalid work order limits {work_order_limits}!"
        self.work_order_limits = work_order_limits

    def initialize_scheme(
        self,
        is_symmetric: bool,
        table_size: int,
        err_reduction_at_step: NDArray[np.floating],
        total_feval_cost_for_k: NDArray[np.floating],
        check_window: tuple[int, int] = (1, 1),
    ):
        super().initialize_scheme(
            is_symmetric, table_size, err_reduction_at_step, total_feval_cost_for_k, check_window
        )
        self.k_min = 1
        self.k_max: int = table_size - 1  # NOTE: Hairer & Wanner use table_size - 2

        self.error_ratios_k = np.empty((table_size - 1,), self.dtype)

    @override
    def evaluate_step(
        self,
        error: NDArray[np.floating],
        x_curr: NDArray[np.floating],
        x_pred: NDArray[np.floating],
        k_curr: int,
        k_target: int,
        allow_early_check: bool = False,
    ) -> contr_ext_state_type:

        state: contr_ext_state_type = "continue"

        error_ratio: float = self._get_error_ratio(error, x_curr, x_pred)
        self.error_ratios_k[k_curr - 1] = (
            error_ratio  # cache this as error computation is expensive
        )

        if (
            k_curr >= 2 and error_ratio >= self.error_ratios_k[k_curr - 1]
        ):  # Hairer & Wanner divergence monitor a), does not have to be run for explicit schemes
            state = "divergence"
        elif k_curr >= k_target - self.check_window[0] or allow_early_check:
            if error_ratio <= 1.0:  # Convergence in line k_target − 1; or  k_target
                state = "accepted"
            elif error_ratio < np.prod(
                self.err_reduction_at_step[k_curr : k_target + self.check_window[1]]
            ):  # Convergence monitor: can we expect convergence in until k_target + reduction_window?
                state = "continue"
            else:
                state = "too_slow_convergence"
        logger.debug(f"Evaluating step {k_curr}, error ratio: {error_ratio}, {state}")
        return state

    # @override
    # def evaluate_step(
    #     self,
    #     error: NDArray[np.floating],
    #     x_curr: NDArray[np.floating],
    #     x_pred: NDArray[np.floating],
    #     k_curr: int,
    #     k_target: int,
    # ) -> contr_ext_state_type:
    #     """ Optimized variant that skips error computation if it is not necessary for step selection. Not applicable to implicit schemes as divergence check is not performed and for early cheking"""

    #     state: contr_ext_state_type = "continue"

    #     if k_curr >= k_target - self.check_window[0] - 1: # for NR variant, H&W step selection can use k_target - self.check_window[0]
    #         error_ratio = self._get_error_ratio(error, x_curr, x_pred)
    #         self.error_ratios_k[k_curr - k_target + self.check_window[0] + 1] = error_ratio # cache this as error computation is expensive
    #         logger.debug(f"Evaluating step {k_curr}, error ratio: {error_ratio}")

    #         if k_curr >= k_target - self.check_window[0]:
    #             if (
    #                 error_ratio <= 1.0
    #             ):  # a) Convergence in line k_target − 1; or  k_target
    #                 state = "accepted"
    #             elif error_ratio <= np.prod(
    #                 self.err_reduction_at_step[
    #                     k_curr : k_target + self.check_window[1]
    #                 ]
    #             ):  # Convergence monitor: can we expect convergence in until k_target + reduction_window?
    #                 state = "continue"
    #             else:
    #                 state = "too_slow_convergence"
    #     return state

    @override
    def get_most_efficient_parameters(
        self,
        k_final: int,
        allow_order_increase: bool,
    ) -> tuple[int, float]:
        next_ktarget = -1
        next_step_mult = -1.0

        assert (
            k_final >= self.k_min and k_final <= self.k_max
        ), "Values outside the allowed range should not be checked"

        if k_final < k_target + self.check_window[1]:
            s_decreased = self._get_step_mult_opt(
                self.error_ratios_k[k_final - 2], k_final - 1
            )  # NOTE: unavailable in the first step (k=1)
            s_same = self._get_step_mult_opt(self.error_ratios_k[k_final - 1], k_final)

            work_decreased = self.total_feval_cost_for_k[k_final - 1] / s_decreased
            work_same = (
                self.total_feval_cost_for_k[k_final] / s_same
            )  # NOTE: since the same step length is used with the multiplier, we can calculate the relative work just from the multipliers

            next_ktarget: int
            next_step_mult: float
            if (
                work_decreased < self.work_order_limits[0] * work_same
                and k_final - 1 >= self.k_min
            ):  # NOTE: this possible double decrease in k-1 appears in Numerical Recipes but not in Hairer&Wanner
                next_ktarget = k_final - 1 # order decrease/target double decrease (start of window)
                next_step_mult = s_decreased
            elif (
                work_same < self.work_order_limits[1] * work_decreased
                and allow_order_increase
                and k_final + 1 <= self.k_max
            ):  # NOTE: this work check is "out of phase" with the increase, since the work for order increase is unknown. We have assumed work_increased = work_check to find s_increased, so work_increased can not be computed from s_increased to perform the check
                next_ktarget = k_final + 1  # order increase / target constant (start of window)
                next_step_mult = (
                    s_same
                    * self.total_feval_cost_for_k[k_final + 1]
                    / self.total_feval_cost_for_k[k_final]
                )
            else:
                next_ktarget = k_final  # order constant / target decrease (start of window)
                next_step_mult = s_same

        else:  # different variant at edge of check window to allow for reduction down by two
            s_decreased = self._get_step_mult_opt(
                self.error_ratios_k[k_final - 3], k_final - 2
            )  # NOTE: might not be initialized
            s_target = self._get_step_mult_opt(
                self.error_ratios_k[k_final - 2], k_final - 1
            )
            s_last = self._get_step_mult_opt(self.error_ratios_k[k_final - 1], k_final)

            work_decreased = self.total_feval_cost_for_k[k_final - 2] / s_decreased
            work_target = self.total_feval_cost_for_k[k_final - 2] / s_target
            work_last = self.total_feval_cost_for_k[k_final] / s_last

            next_ktarget = k_final-1
            next_step_mult = s_target
            work_temp_faster = work_target
            if (
                work_decreased < self.work_order_limits[0] * work_target
                and k_final - 1 >= self.k_min
            ):
                next_ktarget = k_final - 2
                next_step_mult = s_decreased
                work_temp_faster = work_decreased

            if (
                work_last < self.work_order_limits[1] * work_temp_faster
                and k_final + 1 <= self.k_max # TODO: and allow_order_increase if using the strict variant
            ):
                next_ktarget = k_final
                next_step_mult = s_last  # NOTE: Numerical recipes instead gives (probably an oversight, their implementation is different): s_b*self.total_feval_cost_for_k[iterator_table + 1]/self.total_feval_cost_for_k[iterator_table]
        return next_ktarget, next_step_mult


class StepControllerExtrapKH_Deuflhard(StepControllerExtrapKH):
    """Combined order and step size (k-h) controller for extrapolation methods. Following the strategy by Deulfhard, "Order and Stepsize Control in Extrapolation Methods", 1983.
    The main difference to the strategy in the StepControllerExtrapKH is that its always possible to reduce the order down to k_min, instead of staying in the check window """

    def __init__(
        self,
        is_greedy: bool = True,
        atol: float | NDArray[np.floating] = 10**-8,
        rtol: float | NDArray[np.floating] = 10**-5,
        norm: Callable[[NDArray[np.floating]], float] = norm_hairer,
        safety_unscaled: float = (0.94),
        safety_tol: float = (0.65),
        s_limits_scaled: tuple[float, float] = (0.02, 4.0),
        step_multiplier_divergence: float = 0.5,
        dtype: DTypeLike = np.double,
        work_order_limits: tuple[float, float] = (0.8, 0.9),
    ) -> None:
        super().__init__(
            atol,
            rtol,
            norm,
            safety_unscaled,
            safety_tol,
            s_limits_scaled,
            step_multiplier_divergence,
            dtype,
        )
        self.is_greedy = is_greedy
        self.work_order_limits = work_order_limits

    def get_most_efficient_params_fullred_optimal(
        self,
        k_final: int,
        allow_order_increase: bool,
    ) -> tuple[int, float]:
        # TODO: order increase

        assert (
            k_final >= self.k_min and k_final <= self.k_max
        ), "Values outside the allowed range should not be checked"

        allow_order_increase &= k_final + 1 <= self.k_max

            work_opt = np.inf
            s_opt = np.inf
            k_opt = 0
            for k_ in reversed(
                range(self.k_min, k_final + 1)
            ):  # for NR version: range(k_final-1, ...)
                # for H&W version: range(k_target - self.check_window[0], ...)
                s_ = self._get_step_mult_opt(self.error_ratios_k[k_ - 1], k_)
                w_ = (
                    self.total_feval_cost_for_k / s_
                )  # NOTE: since the same step length is used with the multiplier, we can calculate the relative work just from the multipliers

                if (allow_order_increase and k_ == k_final-1): # Interjection by additional check once the two required work/step quantities are available
                    # TODO: these two variants are not necessary
                    if self.is_greedy:
                        if (work_opt < self.work_order_limits[1] * w_):
                            k_opt = k_final + 1
                            s_opt = s_opt * self.total_feval_cost_for_k[k_final + 1]/ self.total_feval_cost_for_k[k_final]
                            break
                    else:
                        work_inc = work_opt * (1. + 1./self.work_order_limits[1]) - w_
                        if work_inc < work_opt:
                            k_opt = k_final + 1
                            work_opt = work_inc
                            s_opt = s_opt * self.total_feval_cost_for_k[k_final + 1]/ self.total_feval_cost_for_k[k_final]


                if (
                    w_ < self.work_order_limits[0] * work_opt
                ):  # NOTE: threshold favors keeping the order constant (high)
                    k_opt = k_
                    work_opt = w_
                    s_opt = s_
                elif self.is_greedy:
                    break
        return k_opt, s_opt

    # def get_most_efficient_params_fullred_greedy(
    #     self,
    #     k_final: int,
    #     allow_order_increase: bool,
    # ) -> tuple[int, float]:
    #     assert (
    #         k_final >= self.k_min and k_final <= self.k_max
    #     ), "Values outside the allowed range should not be checked"

    #     s_decreased = self._get_step_mult_opt(
    #         self.error_ratios_k[k_final - 2], k_final - 1
    #     )  # NOTE: unavailable in the first step (k=1)
    #     s_check = self._get_step_mult_opt(self.error_ratios_k[k_final - 1], k_final)

    #     work_decreased = self.total_feval_cost_for_k[k_final - 1] / s_decreased
    #     work_target = (
    #         self.total_feval_cost_for_k[k_final] / s_check
    #     )  # NOTE: since the same step length is used with the multiplier, we can calculate the relative work just from the multipliers

    #     # check for possible order increase
    #     if (
    #         work_target < self.work_order_limits[1] * work_decreased
    #         and allow_order_increase
    #         and k_final + 1 <= self.k_max
    #     ):  # NOTE: this work check is "out of phase" with the increase, since the work for order increase is unknown. We have assumed work_increased = work_check to find s_increased. so work_increased can not be computed from s_increased to perform the check
    #         next_ktarget = (
    #             k_final + 1
    #         )  # order increase / target constant (start of window)
    #         next_step_mult = (
    #             s_check
    #             * self.total_feval_cost_for_k[k_final + 1]
    #             / self.total_feval_cost_for_k[k_final]
    #         )
    #     else:  # decrease k_check until optimum has been found, iteration works if there is just a single minimum, toherwise all work_k have to be computed and the minimum selected
    #         while work_decreased < self.work_order_limits[0] * work_target: # NOTE: threshold favors keeping the order constant
    #             k_final -= 1
    #             if (
    #                 k_final - 1
    #                 < self.k_min  # NOTE: selecting different limits here will cause this to behave like the H&W/NR versions
    #                 # k_final - 1 < k_target - self.check_window[0] # H&W version
    #                 # k_final - 1 < k_final_initial - 1 # NR version
    #             ):
    #                 break  # early exit to prevent the computation of the multiplier/access of error_ratio of k_min-1

    #             s_check = s_decreased
    #             s_decreased = self._get_step_mult_opt(
    #                 self.error_ratios_k[k_final - 2], k_final - 1
    #             )  # NOTE: if k_final=1, this will not exist
    #             work_target = work_decreased
    #             work_decreased = self.total_feval_cost_for_k[k_final - 1] / s_decreased
    #         next_ktarget = k_final
    #         next_step_mult = s_check
    #     return next_ktarget, next_step_mult

class StepControllerExtrapH(StepControllerExtrap):
    """Step size controller with constant order for extrapolation methods. Step size is controlled by an unsophisticated I-controller.
    Default check_window is (0, 0). By setting a lower check-window, the controller exits early without computing all orders up to k_target if convergene is not to be expected"""

    def __init__(
        self,
        k_target: int,
        atol: float | NDArray[np.floating] = 10**-8,
        rtol: float | NDArray[np.floating] = 10**-5,
        norm: Callable[[NDArray[np.floating]], float] = norm_hairer,
        safety_unscaled: float = (0.94),
        safety_tol: float = (0.65),
        s_limits_scaled: tuple[float, float] = (0.02, 4.0),
        step_multiplier_divergence: float = 0.5,
        dtype: DTypeLike = np.double,
    ) -> None:
        super().__init__(
            atol,
            rtol,
            norm,
            safety_unscaled,
            safety_tol,
            s_limits_scaled,
            step_multiplier_divergence,
            dtype,
        )
        self.error_ratio = np.nan

    def initialize_scheme(
        self,
        is_symmetric: bool,
        table_size: int,
        err_reduction_at_step: NDArray[np.floating],
        total_feval_cost_for_k: NDArray[np.floating],
        check_window: tuple[int, int] = (0, 0),
    ):
        super().initialize_scheme(
            is_symmetric,
            table_size,
            err_reduction_at_step,
            total_feval_cost_for_k,
            check_window=(0, 0),
        )

    def evaluate_step(
        self,
        error: NDArray[np.floating],
        x_curr: NDArray[np.floating],
        x_pred: NDArray[np.floating],
        k_curr: int,
        k_target: int,
        allow_early_check: bool = False,
    ) -> contr_ext_state_type:

        state: contr_ext_state_type = "continue"

        error_ratio = self._get_error_ratio(error, x_curr, x_pred)

        if (
            k_curr >= 2 and error_ratio >= self.error_ratio
        ):  # Hairer & Wanner divergence monitor a), does not have to be run for explicit schemes
            state = "divergence"
        elif k_curr >= k_target - self.check_window[0] or allow_early_check:
            if error_ratio <= 1.0:  # Convergence in line k_target − 1; or  k_target
                state = "accepted"
            else:
                state = "too_slow_convergence"
        self.error_ratio = error_ratio
        logger.debug(f"Evaluating step {k_curr}, error ratio: {error_ratio}, {state}")
        return state

    @override
    def get_most_efficient_parameters(
        self,
        k_final: int,
        allow_order_increase: bool,
    ) -> tuple[int, float]:
        next_ktarget = k_final 
        next_step_mult = self._get_step_mult_opt(self.error_ratio, k_final)

        return next_ktarget, next_step_mult

class StepControllerExtrapK(StepControllerExtrap):
    """Step size controller with constant step size for extrapolation methods. The order is adapted to fulfill the desired error tolerance"""

    def __init__(
        self,
        atol: float | NDArray[np.floating] = 10**-8,
        rtol: float | NDArray[np.floating] = 10**-5,
        norm: Callable[[NDArray[np.floating]], float] = norm_hairer,
        safety_unscaled: float = (0.94),
        safety_tol: float = (0.65),
        s_limits_scaled: tuple[float, float] = (0.02, 4.0),
        step_multiplier_divergence: float = 0.5,
        dtype: DTypeLike = np.double,
    ) -> None:
        super().__init__(
            atol,
            rtol,
            norm,
            safety_unscaled,
            safety_tol,
            s_limits_scaled,
            step_multiplier_divergence,
            dtype,
        )
        self.error_ratio = np.nan

    def initialize_scheme(
        self,
        is_symmetric: bool,
        table_size: int,
        err_reduction_at_step: NDArray[np.floating],
        total_feval_cost_for_k: NDArray[np.floating],
        check_window: tuple[int, int] = None,
    ):
        super().initialize_scheme(
            is_symmetric,
            table_size,
            err_reduction_at_step,
            total_feval_cost_for_k,
            check_window=(self.table_size - 1, self.table_size - 1),
        )

    @override
    def evaluate_step(
        self,
        error: NDArray[np.floating],
        x_curr: NDArray[np.floating],
        x_pred: NDArray[np.floating],
        k_curr: int,
        k_target: int,
        allow_early_check: bool = False,
    ) -> contr_ext_state_type:
        state: contr_ext_state_type = "continue"

        error_ratio: float = self._get_error_ratio(error, x_curr, x_pred)

        if error_ratio <= 1.0:
            state = "accepted"
        elif error_ratio > np.prod(
            self.err_reduction_at_step[k_curr : ]
        ):  # Convergence monitor: can we expect convergence in later steps?
            if k_curr < self.table_size - 1:
                state = "continue"
                logger.warning(
                    f"Error tolerance will probably not be met until the end of the table. Continuing anyway."
                )
            else:
                state = "accepted"
                logger.critical(
                    f"Error tolerance can't be met with the current step and table size. Continuing anyway."
                )

        logger.debug(f"Evaluating step {k_curr}, error ratio: {error_ratio}, {state}")
        return state

    @override
    def get_most_efficient_parameters(
        self,
        k_final: int,
        allow_order_increase: bool,
    ) -> tuple[int, float]:
        next_step_mult = 1.0
        next_ktarget = self.table_size-1 # this stays unused

        return next_ktarget, next_step_mult


class StepControllerExtrapDummy(StepControllerExtrap):
    """Step size controller that keeps step size and order constant (i.e. does nothing). Used for testing purposes"""

    def __init__(
        self,
        atol: float | NDArray[np.floating] = 10**-8,
        rtol: float | NDArray[np.floating] = 10**-5,
        norm: Callable[[NDArray[np.floating]], float] = norm_hairer,
        safety_unscaled: float = (0.94),
        safety_tol: float = (0.65),
        s_limits_scaled: tuple[float, float] = (0.02, 4.0),
        step_multiplier_divergence: float = 0.5,
        dtype: DTypeLike = np.double,
    ) -> None:
        super().__init__(
            atol,
            rtol,
            norm,
            safety_unscaled,
            safety_tol,
            s_limits_scaled,
            step_multiplier_divergence,
            dtype,
        )

    def initialize_scheme(
        self,
        is_symmetric: bool,
        table_size: int,
        err_reduction_at_step: NDArray[np.floating],
        total_feval_cost_for_k: NDArray[np.floating],
        check_window: tuple[int, int] = (0, 0),
    ):
        super().initialize_scheme(
            is_symmetric,table_size, err_reduction_at_step, total_feval_cost_for_k, check_window
        )

    @override
    def evaluate_step(
        self,
        error: NDArray[np.floating],
        x_curr: NDArray[np.floating],
        x_pred: NDArray[np.floating],
        k_curr: int,
        k_target: int,
        allow_early_check: bool = False,
    ) -> contr_ext_state_type:
        state: contr_ext_state_type = "continue"

        error_ratio = -1.0
        if logger.isEnabledFor(logging.DEBUG):
            error_ratio = self._get_error_ratio(error, x_curr, x_pred)

        if k_curr >= k_target:
            state = "accepted"
        else:
            state = "continue"
        logger.debug(f"Evaluating step {k_curr}, error ratio: {error_ratio}, {state}")
        return state

    @override
    def get_most_efficient_parameters(
        self,
        k_final: int,
        allow_order_increase: bool,
    ) -> tuple[int, float]:
        return k_final, 1.0

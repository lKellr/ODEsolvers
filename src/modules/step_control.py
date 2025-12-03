from abc import ABC
from typing import Any, Callable, Literal, NamedTuple
import numpy as np
from numpy.typing import NDArray
from modules.helpers import norm_hairer, clip
import logging

logger = logging.getLogger(__name__)

tab_state_type = Literal["accepted", "continue", "retry"]

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


def get_step_PI(err_ratio, err_ratio_last, control_params):
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
        self.is_retry = False
        self.prev_step_size = float("nan")

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
            )  # NOTE: without deadzone and clipping, this should not be problematic since we use it just for improving rejected estiamtes
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


class StepControllerExtrapKH(StepController):
    """Combined order and step size (k-h) controller for extrapolation methods. Following the strategy layed out in Hairer&Wanner and modified in Numerical Recipes. Originally proposed by Deulfhard"""

    def __init__(
        self,
        table_size: int,
        step_seq:NDArray[np.integer],
        atol: float | NDArray[np.floating] = 10**-8,
        rtol: float | NDArray[np.floating] = 10**-5,
        norm: Callable[[NDArray[np.floating]], float] = norm_hairer,
        safety_unscaled: float = (0.94),
        safety_tol: float = (0.65),
        s_limits_scaled: tuple[float, float] = (0.02, 4.0),
        work_order_limits: tuple[float, float] = (0.8, 0.9),
        step_multiplier_divergence: float = 0.5,
        rel_jac_cost: float = num_odes,  # TODO: put into named tuple
        rel_lu_cost: float = 0.5,
        rel_backsub_cost: float = 0.0,
    ) -> None:
        super().__init__(atol, rtol, norm, safety_tol)

        self.table_size = table_size

        self.err_inc_fac = np.array(
            [
                (step_seq[k] / step_seq[0]) ** (1 + is_symmetric)
                for k in range(table_size - 1)
            ]
        )

        feval_cost_per_kstep = (
            step_seq
            + is_symmetric
            + is_implicit * (rel_lu_cost + step_seq * rel_backsub_cost)
        )
        self.total_feval_cost_at_kstep = (
            np.cumsum(feval_cost_per_kstep)
            + is_implicit * rel_jac_cost
        )

        self.safety_unscaled = safety_unscaled
        self.s_limits_scaled = s_limits_scaled
        self.work_order_limits = work_order_limits
        self.step_multiplier_divergence = step_multiplier_divergence

    def get_initial_ktarget(self) -> int:
      """very rough estimate from numerical recipes, can be taken for example from Hairer&Wanner Fig.9.5"""
      log_fact = -max(-12.0, np.log10(self.rtol)) * 0.6 + 0.5
      k_target = max(
            1, min(self.table_size - 1, int(log_fact))
        )
      return k_target

    def _get_step_mult_opt(self, err_ratio_k: float, next_k: int) -> float:
        s_opt = self.safety_unscaled * (self.safety_tol / err_ratio_k) ** (
            1 / (2 * next_k + 1)
        )
        temp_s_limit_descaled = self.s_limits_scaled[0] ** (1 / (2 * next_k + 1))
        s_opt = clip(
            s_opt,
            temp_s_limit_descaled / self.s_limits_scaled[1],
            1 / temp_s_limit_descaled,
        )
        return s_opt

    def _get_params_next_step(
        self, err_ratio: float, err_ratio_prev_col: float, k_check: int
    ) -> tuple[int, float]:
        s_same = self._get_step_mult_opt(err_ratio, k_check)
        s_decreased = self._get_step_mult_opt(err_ratio_prev_col, k_check - 1)
        work_same = self.total_feval_cost_at_kstep[k_check] / s_same # NOTE: since the same step length is used with the multiplier, we can calcualte the relative work just from the multipliers
        work_decreased = self.total_feval_cost_at_kstep[k_check - 1] / s_decreased

        k_next: int
        s_next: float
        if (
            work_decreased < self.work_order_limits[0] * work_same and k_check > 2
        ):  # NOTE: this possible double decrease in k-1 appears in Numerical Recipes but not in Hairer&Wanner
            k_next = k_check - 1  # order decrease
            s_next = s_decreased
        elif (
            work_same < self.work_order_limits[1] * work_decreased
            and k_check + 1 < self.table_size
        ):  # NOTE: this work check is "out of phase" with the increase, since the work for order increase is unknown. This is probably still accurate since we are close to the optimal value, wheere the work costs are alsomst equal?
            k_next = k_check + 1  # order increase
            s_next = (
                s_same
                * self.total_feval_cost_at_kstep[k_check + 1]
                / self.total_feval_cost_at_kstep[k_check]
            )
        else:
            k_next = k_check
            s_next = s_same
        return k_next, s_next

    def evaluate_step(self, table_col_ix:int, k_target: int,
        error: NDArray[np.floating],
        error_prev_col: NDArray[np.floating],
        x_curr: NDArray[np.floating],
        x_table: NDArray[np.floating],
) -> tuple[int, float, tab_state_type]:
      state: tab_state_type = "continue"
      next_k = -1
      next_step_mult = -1.

      err_ratio = self._get_error_ratio(error, x_curr, x_table)

      if(table_col_ix <= k_target):
          if(err_ratio <= 1.): # a) Convergence in line k − 1iterator_target-1
              err_ratio_prev_col = self._get_error_ratio(error_prev_col, x_curr, x_prevtable) # TODO: this can be cached
              next_k, step_mult = self._get_params_next_step(err_ratio, err_ratio_prev_col, table_col_ix)
              next_step_mult = step_mult
              state = "accepted"
          elif(err_ratio > np.prod([self.err_inc_fac[k] for k in range(table_col_ix, k_target+1)])): # b) Convergence monitor: can we expect convergence in later steps?
              k_opt, step_mult = self._get_params_next_step(err_ratio, err_ratio_prev_col, table_col_ix)
              next_k = min(k_target, k_opt)
              next_step_mult = min(1., step_mult)
              state = "retry"
            # otherwise continue with the next table row
            # else:
            #   state = "continue"
      else: # table_col_ix == iterator_target+1
        if(err_ratio <= 1.): # in this case, get_params_next_step can not be used
            # TODO: write this more elegantly
            k_target, step_mult = ?self._get_params_next_step(err_ratio, iterator_table)

            s_same = self._get_step_mult_opt(err_ratio, iterator_table)
            s_decreased = self._get_step_mult_opt(!err_ratio, iterator_table-1)
            s_increased = self._get_step_mult_opt(!err_ratio, iterator_table+1)
            work_same = self.total_feval_cost_at_kstep[iterator_table]/s_same
            work_decreased = self.total_feval_cost_at_kstep[iterator_table-1]/s_decreased
            work_increased = self.total_feval_cost_at_kstep[iterator_table+1]/s_increased

            next_k = k_target
            if(work_decreased < self.work_order_limits[0]*work_same):
                next_k = iterator_table - 1
                next_step_mult = self._get_step_mult_opt(!err_ratio, k_check)
            if(work_increased < self.work_order_limits[1]*work_new):
                next_k = iterator_table + 1
                next_step_mult = self._get_step_mult_opt(!err_ratio, k_target)*self.total_feval_cost_at_kstep[iterator_table + 1]/self.total_feval_cost_at_kstep[iterator_table]
            else:
                next_step_mult = self._get_step_mult_opt(err_ratio, k_target)
            state = "accepted"
        else: # convergence not reached even at higher order, retry
            k_opt, step_mult = self._get_params_next_step(err_ratio_prev_col, err_ratio_2prev_col, k_target)
            next_k = min(k_target, k_opt)
            next_step_mult = min(1., step_mult)
            state = "retry"
      return next_k, next_step_mult, state

class StepControllerExtrapH(StepController):
    """Step size controller with constant order for extrapolation methods. Step size is controlled by an unsophisticated I-controller"""

    def __init__(
        self,
        table_size: int,
        order: int,
        step_seq:NDArray[np.integer],
        atol: float | NDArray[np.floating] = 10**-8,
        rtol: float | NDArray[np.floating] = 10**-5,
        norm: Callable[[NDArray[np.floating]], float] = norm_hairer,
        safety_unscaled: float = (0.94),
        safety_tol: float = (0.65),
        s_limits_scaled: tuple[float, float] = (0.02, 4.0),
        step_multiplier_divergence: float = 0.5,
    ) -> None:
        super().__init__(atol, rtol, norm, safety_tol)

        self.table_size = table_size

        self.order = order

        self.safety_unscaled = safety_unscaled
        self.s_limits_scaled = s_limits_scaled
        self.step_multiplier_divergence = step_multiplier_divergence

    def _get_step_mult_opt(self, err_ratio_k: float, next_k: int) -> float:
        s_opt = self.safety_unscaled * (self.safety_tol / err_ratio_k) ** (
            1 / (2 * next_k + 1)
        )
        temp_s_limit_descaled = self.s_limits_scaled[0] ** (1 / (2 * next_k + 1))
        s_opt = clip(
            s_opt,
            temp_s_limit_descaled / self.s_limits_scaled[1],
            1 / temp_s_limit_descaled,
        )
        return s_opt

    def evaluate_step(self, table_col_ix:int, k_target: int,
        error: NDArray[np.floating],
        error_prev_col: NDArray[np.floating],
        x_curr: NDArray[np.floating],
        x_table: NDArray[np.floating],
) -> tuple[int, float, tab_state_type]:
      next_k = self.order+1 # first check at next_k-1
      state: tab_state_type = "continue"
      err_ratio = self._get_error_ratio(error, x_curr, x_table)

      if(err_ratio <= 1.):
        next_step_mult = self._get_step_mult_opt(err_ratio, self.order)
        state = "accepted"
      else:
        next_step_mult = self._get_step_mult_opt(err_ratio, self.order)
        state = "retry"

      return next_k, next_step_mult, state

class StepControllerExtrapP(StepController):
    """Step size controller with constant step size for extrapolation methods. The order is adapted to fulfill the desired error tolerance"""

    def __init__(
        self,
        table_size: int,
        step_seq:NDArray[np.integer],
        atol: float | NDArray[np.floating] = 10**-8,
        rtol: float | NDArray[np.floating] = 10**-5,
        norm: Callable[[NDArray[np.floating]], float] = norm_hairer,
        safety_unscaled: float = (0.94),
        safety_tol: float = (0.65),
        s_limits_scaled: tuple[float, float] = (0.02, 4.0),
        step_multiplier_divergence: float = 0.5,
    ) -> None:
        super().__init__(atol, rtol, norm, safety_tol)

        self.table_size = table_size
        self.err_inc_fac = np.array(
            [
                (step_seq[k] / step_seq[0]) ** (1 + is_symmetric)
                for k in range(table_size - 1)
            ]
        )

        self.safety_unscaled = safety_unscaled
        self.s_limits_scaled = s_limits_scaled
        self.step_multiplier_divergence = step_multiplier_divergence

    def _get_step_mult_opt(self, err_ratio_k: float, next_k: int) -> float:
        s_opt = self.safety_unscaled * (self.safety_tol / err_ratio_k) ** (
            1 / (2 * next_k + 1)
        )
        temp_s_limit_descaled = self.s_limits_scaled[0] ** (1 / (2 * next_k + 1))
        s_opt = clip(
            s_opt,
            temp_s_limit_descaled / self.s_limits_scaled[1],
            1 / temp_s_limit_descaled,
        )
        return s_opt

    def evaluate_step(self, table_col_ix:int, k_target: int,
        error: NDArray[np.floating],
        error_prev_col: NDArray[np.floating],
        x_curr: NDArray[np.floating],
        x_table: NDArray[np.floating],
) -> tuple[int, float, tab_state_type]:
      next_step_mult = 1.0 # we do not want to change the step size
      next_k = 1 # this trips convergence checks after in each tableau column
      state: tab_state_type = "continue"
      err_ratio = self._get_error_ratio(error, x_curr, x_table)

      if(err_ratio <= 1.):
        state = "accepted"
      elif(err_ratio > np.prod([self.err_inc_fac[k] for k in range(table_col_ix, self.table_size+1)])): # b) Convergence monitor: can we expect convergence in later steps?
          state = "retry"

      return next_k, next_step_mult, state

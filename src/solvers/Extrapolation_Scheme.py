from abc import ABC, abstractmethod
from numpy._typing._array_like import NDArray
from numpy._typing._array_like import NDArray
from numpy._typing._array_like import NDArray
from numpy import float64, floating, integer
from typing import Any, Callable, Literal, NamedTuple, TypeVar, override
import numpy as np
from numpy.typing import DTypeLike, NDArray
import logging

from scipy.linalg import lu_factor, lu_solve

from modules.helpers import clip, numerical_jacobian, norm_hairer, numerical_jacobian_t

from modules.step_control import (
    StepControllerExtrap,
    StepControllerExtrapKH,
    contr_ext_state_type,
)

logger = logging.getLogger(__name__)


class ImplicitRelCosts(NamedTuple):
    rel_jac_cost: float
    rel_lu_cost: float
    rel_backsub_cost: float


class ExtrapolationSolver(ABC):
    def __init__(
        self,
        ode_fun: Callable[
            [float, NDArray[np.floating]], NDArray[np.floating]
        ],  # TODO: does this have to be set here?
        step_seq: (
            NDArray[np.integer]
            | Literal["harmonic", "Romberg", "Bulirsch", "harmonic2", "fours", "SODEX"]
        ),
        is_symmetric: bool,
        require_jacobian: bool,
        fevals_per_step: NDArray[np.integer],
        table_size: int,
        num_odes: int,  # TODO: this is only required for implicit_rel_costs and mass matrix creation
        jac_fun: (
            Callable[[float, NDArray[np.floating]], NDArray[np.floating]] | None
        ) = None,
        mass_matrix: NDArray[np.floating] | None = None,
        step_controller: StepControllerExtrap | None = None,
        implicit_rel_costs: ImplicitRelCosts | None = None,
        dtype: DTypeLike = np.double,
    ):
        # TODO: allow direct specification of the array
        if step_seq == "harmonic":
            step_seq = np.array(range(1, table_size + 1))
        elif step_seq == "Romberg":
            step_seq = np.array([i**2 for i in range(table_size + 1)])
        elif step_seq == "Bulirsch":
            step_seq = np.array(
                [
                    2 ** (k // 2) if k == 1 or k % 2 == 0 else 1.5 * 2 ** (k // 2)
                    for k in range(1, table_size + 1)
                ]
            )
        elif step_seq == "harmonic2":
            step_seq = np.array(range(2, table_size + 2))
        elif (
            step_seq == "fours"
        ):  # this sequence would allow for dense output, form Numerical Recipes
            step_seq = np.array(range(2, 4 * table_size, 4))
        elif step_seq == "SODEX":
            assert (
                table_size <= 7
            ), "table sizes larger than 7 are not implemented when using the step sequence SODEX"
            step_seq = np.array(
                [2, 6, 10, 14, 22, 34, 50][:table_size]
            )  # NOTE: i am not sure if a formula exists for these; they have to be multiples of even numbers, according to Bader&Deulfhard1983, the ratio of subsequent entries must lie between 1 and 5/7
        else:
            raise ValueError(f"step sequence of type {step_seq} is not available.")

        self.table_size: int = table_size
        self.dtype = dtype

        self.require_jacobian = require_jacobian  # TODO: is this required?
        self.step_seq = step_seq
        # not all entries are needed, only the lower? triangular part and only beginning from j=1, but i cant index a list, so this has to be a padded array
        self.coeffs_Aitken = np.array(
            [
                [
                    (
                        (
                            1.0
                            / (self.step_seq[j] / self.step_seq[j - k])
                            ** (2 if is_symmetric else 1)
                            - 1.0
                        )
                        if k <= j - 1
                        else 0.0
                    )
                    for k in range(table_size - 1)
                ]
                for j in range(table_size - 1)
            ],
            dtype,
        )

        self.fevals_per_step = fevals_per_step

        if step_controller is None:
            step_controller = StepControllerExtrapKH(
                dtype=dtype,
            )

        self.step_controller = step_controller

        # if implicit_rel_costs is None:
        #     implicit_rel_costs = ImplicitRelCosts(  # TODO: tune these
        #         rel_jac_cost=num_odes + 1 if jac_fun is None else 2,
        #         rel_lu_cost=1.0,
        #         rel_backsub_cost=0.0,
        #     )
        # total_feval_cost_for_kstep: NDArray[np.floating] = np.cumsum(
        #     fevals_per_step
        # )  # pyright: ignore[reportAssignmentType]
        # if self.is_implicit:
        #     total_feval_cost_for_kstep += np.cumsum(
        #         implicit_rel_costs.rel_lu_cost
        #         + self.step_seq * implicit_rel_costs.rel_backsub_cost
        #     )
        #     total_feval_cost_for_kstep += implicit_rel_costs.rel_jac_cost

        # err_reduction_at_step = np.array(
        #     [
        #         (self.step_seq[k] / self.step_seq[0]) ** (1 + is_symmetric)
        #         for k in range(self.table_size - 1)
        #     ]
        # )

        # self.step_controller.initialize_scheme(
        #     self.table_size, err_reduction_at_step, total_feval_cost_for_kstep
        # )

        self.ode_fun = ode_fun
        if jac_fun is None:
            self.jac_fun: Callable[
                [float, NDArray[np.floating]], NDArray[np.floating]
            ] = lambda t, x: numerical_jacobian_t(t, x, ode_fun, delta=1e-8)
        else:
            self.jac_fun = jac_fun
        self.mass_matrix: NDArray[np.floating] = (
            mass_matrix if mass_matrix is not None else np.identity(num_odes)
        )

    def init_controller(
        self,
        implicit_rel_costs: ImplicitRelCosts | None = None,
    ):
        if (
            implicit_rel_costs
            is None  # TODO: required for implicit schemes, or explicit with mass
        ):  # TODO: this depends on the equation, not the scheme
            implicit_rel_costs = ImplicitRelCosts(  # TODO: tune these
                rel_jac_cost=num_odes + 1 if jac_fun is None else 2,
                rel_lu_cost=1.0,
                rel_backsub_cost=0.0,
            )

        # TODO: following block should be moved to the scheme!
        # either a lambda in the child, or provide all params(n_feval, require backsub, n_LU, use_smootinh) to the parent
        feval_cost_per_base_solve = (
            lambda n_steps: (n_steps + use_smoothing)
            * n_feval
            * (1 + solve_linear * implicit_rel_costs.rel_backsub_cost)
            + implicit_rel_costs.rel_lu_cost * n_LU
        )
        total_feval_cost_for_kstep: NDArray[np.floating] = np.cumsum(
            feval_cost_per_base_solve
        )
        if self.require_jacobian:
            total_feval_cost_for_k += implicit_rel_costs.rel_jac_cost

        err_reduction_at_step = np.array(
            [
                (self.step_seq[k] / self.step_seq[0]) ** (1 + is_symmetric)
                for k in range(self.table_size - 1)
            ]
        )

        self.step_controller.initialize_scheme(
            self.table_size, err_reduction_at_step, total_feval_cost_for_k
        )

    def fill_extrapolation_table(
        self, T_fine_first_order, T_table_k, n_columns
    ) -> None:
        """Increases the accuracy of the estimate for x0 by one order in the stepsize with the help of Richardson extrapolation.
        For this, the number of steps has to be increased over the previous order. Approximations of all orders lower than the target order are computed with this number of steps.
        [[Target_k is the target_order - 1, which is again shfted by one relative to the literature because y indexing begins at zero.]]
        The function fills a table of the computed approximations to reuse in the next order increasing step.
        """
        T_extrap = T_fine_first_order

        # perform repeated Richardson extrapolation until the target order has been reached, T_table_k contains lower resolution approximations from previous extrapolation step
        for col in range(0, n_columns):
            T_coarselow = T_table_k[col]
            T_finelow = T_extrap
            T_extrap = (
                T_finelow
                + (T_finelow - T_coarselow) * self.coeffs_Aitken[n_columns, col]
            )
            T_table_k[col] = T_finelow
        T_table_k[n_columns] = T_extrap

    def extrapolation_step(
        self,
        t_curr: float,
        x_curr: NDArray[np.floating],
        k_target: int,
        step_size: float,
        allow_full_order_variation: bool = False,
    ) -> tuple[NDArray[np.floating], int, float, bool, dict[str, Any]]:
        """Performs an extrapolation step of x0 until t + step_size"""
        err = np.empty_like(x_curr)
        err_prev = np.empty_like(x_curr)  # required for overflow remedy

        step_info: dict[str, Any] = dict(
            n_feval=0,
            n_jaceval=0,
            n_lu=0,
            local_error=np.nan,
            max_substeps=np.nan,
        )
        # calculate initial jacobian, will be reused at the start of each extrapolation step
        jac0 = None
        if self.require_jacobian:
            jac0 = self.jac_fun(t_curr, x_curr.astype(self.dtype))

        # this is allocated with max size, alternative would be to extend the size each loop iteration, not sure if this would be smart in terms of repeated allocation performance cost
        T_table_k = np.empty((self.table_size, x_curr.shape[0]), self.dtype)
        T_table_k[0], is_diverging = self.base_scheme(
            x_curr,
            t_curr,
            t_max=t_curr + step_size,
            n_steps=self.step_seq[0],
            jac0=jac0,
        )

        state: contr_ext_state_type | Literal["divergence"] = (
            "continue" if not is_diverging else "divergence"
        )
        next_k = -1
        next_step_mult = -1.0
        iterator_table = 0
        while state == "continue":
            iterator_table += 1
            # Basic operations: compute with more steps, then fill row in tableau
            T_fine_first_order, is_diverging = self.base_scheme(
                x_curr,
                t_curr,
                t_max=step_size,
                n_steps=self.step_seq[iterator_table],
                jac0=jac0,
            )
            self.fill_extrapolation_table(T_fine_first_order, T_table_k, iterator_table)
            err_prev = err
            err = np.abs(T_table_k[iterator_table - 1] - T_table_k[iterator_table])

            logger.debug(f"Stage reached: {iterator_table}, error: {err}")

            # exit conditions
            if iterator_table >= k_target - 1 or allow_full_order_variation:
                next_k, next_step_mult, state = self.step_controller.evaluate_step(
                    iterator_table,
                    k_target,
                    err,
                    x_curr,
                    T_table_k[iterator_table],
                )

            # divergence monitor, TODO: only required for implicit methods
            if self.require_jacobian and (
                is_diverging or (iterator_table >= 2 and np.any(err >= err_prev))
            ):  # Hairer & Wanner divergence monitor a)
                next_k = k_target  # not sure if this is the correct way to do this, maybe k should even be decreased?
                next_step_mult = self.step_controller.step_multiplier_divergence
                self.step_controller.is_retry = True  # set retry flag so that order and step size are not allowed to increase during retry
                state = "divergence"

        step_info["stop_reason"] = state
        logger.debug(step_info["stop_reason"])
        step_info["n_feval"] = np.sum(
            self.fevals_per_step[: iterator_table + 1]
        )  # TODO: check this
        step_info["n_lu"] = iterator_table
        step_info["n_jaceval"] = 1
        step_info["local_error"] = err
        step_info["max_substeps"] = self.step_seq[iterator_table]
        return (
            T_table_k[iterator_table],
            next_k,
            next_step_mult * step_size,
            state == "accepted",
            step_info,
        )

    def solve(
        self,
        x0: NDArray[np.floating],
        t_max: float,
        t0: float = 0,
        params_step0: tuple[int, float] | None = None,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], dict[str, Any]]:

        solve_info: dict[str, Any] = dict(
            n_feval=0,
            n_jaceval=0,
            n_lu=0,
            n_restarts=0,
        )

        k_target: int
        step: float
        if params_step0 is None:
            k_target = self.step_controller.get_initial_ktarget()
            step = self.step_controller.get_initial_stepHW(
                self.ode_fun, x0, t0=t0, p=k_target
            )
        else:
            k_target = params_step0[0]
            step = params_step0[1]

        time = [t0]
        solution = [x0]

        current_time = t0
        while current_time < t_max:
            if (
                current_time + step > t_max
            ):  # shorten h if we would go further than necessary
                step = t_max - current_time
            logger.debug(f"Starting step at time {current_time} of {t_max}")

            # do step
            new_solution, k_target, step, accepted, step_info = self.extrapolation_step(
                current_time,
                solution[-1],
                k_target,
                step,
                allow_full_order_variation=(current_time <= t0)
                or (
                    current_time + step >= t_max
                ),  # allow quick order variation for first and last steps when target order is not optimal
            )
            # info
            solve_info["n_feval"] += step_info["n_feval"]
            solve_info["n_jaceval"] += step_info["n_jaceval"]
            solve_info["n_lu"] += step_info["n_lu"]
            logger.debug(
                "Finished step\n"
                + "\n".join([f"{k}: {v}" for k, v in step_info.items()])
                + "\n"
            )

            if accepted:
                solution.append(new_solution)
                time.append(current_time)
                current_time += step
            else:
                logger.debug(f"Retrying step with h = {step}")

        # finished
        logger.debug(
            "Finished\n"
            + "\n".join([f"{k}: {v}" for k, v in solve_info.items()])
            + "\n"
        )

        return np.array(time, self.dtype), np.array(solution, self.dtype), solve_info

    @abstractmethod
    def base_scheme(
        self,
        x0: NDArray[np.floating],
        t0: float,
        t_max: float,
        n_steps: int,
        jac0: NDArray[np.floating] | None,
    ) -> tuple[NDArray[np.floating], bool]:
        raise NotImplementedError()


class EulerExtrapolation(ExtrapolationSolver):
    """Extrapolation with Euler's method as the underlying scheme. With default config similar to the EULEX code of Hairer&Wanner"""

    def __init__(
        self,
        ode_fun: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
        num_odes: int,
        table_size: int = 10,
        step_controller: StepControllerExtrap | None = None,
        step_seq: (
            NDArray[np.integer]
            | Literal["harmonic", "Romberg", "Bulirsch", "harmonic2", "fours", "SODEX"]
        ) = "harmonic",
        mass_matrix: NDArray[np.floating] | None = None,
        dtype: DTypeLike = np.double,
    ):
        super().__init__(
            ode_fun=ode_fun,
            step_seq=step_seq,
            is_symmetric=False,
            require_jacobian=False,
            fevals_per_step=step_seq,
            table_size=table_size,
            num_odes=num_odes,
            jac_fun=None,
            mass_matrix=mass_matrix,
            step_controller=step_controller,
            implicit_rel_costs=None,
            dtype=dtype,
        )

    @override
    def base_scheme(
        self,
        x0: NDArray[np.floating],
        t0: float,
        t_max: float,
        n_steps: int,
        jac0: NDArray[np.floating] | None = None,
    ) -> tuple[NDArray[np.floating], bool]:
        """Forward Euler scheme"""
        delta_t = (t_max - t0) / n_steps

        x_n = x0
        t_n = t0
        for _ in range(n_steps):
            delta_x = delta_t * self.ode_fun(t_n, x_n)
            x_n = x_n + delta_x
            t_n += delta_t
        return x_n, False


class ModMidpointExtrapolation(ExtrapolationSolver):
    """This extrapolation method is based around the modified midpoint method. As it is symmetric, the order increases by afactor of 2 in each extapolation step.
    With default config similar to the ODEX code of Hairer&Wanner"""

    def __init__(
        self,
        ode_fun: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
        num_odes: int,
        table_size: int = 10,
        step_controller: StepControllerExtrap | None = None,
        use_smoothing: bool = False,
        step_seq: (
            NDArray[np.integer]
            | Literal["harmonic", "Romberg", "Bulirsch", "harmonic2", "fours", "SODEX"]
        ) = "harmonic",
        mass_matrix: NDArray[np.floating] | None = None,
        dtype: DTypeLike = np.double,
    ):
        self.use_smoothing = use_smoothing
        super().__init__(
            ode_fun=ode_fun,
            step_seq=step_seq,
            is_symmetric=True,
            require_jacobian=False,
            fevals_per_step=step_seq + 1 * use_smoothing,
            table_size=table_size,
            num_odes=num_odes,
            jac_fun=None,
            mass_matrix=mass_matrix,
            step_controller=step_controller,
            implicit_rel_costs=None,
            dtype=dtype,
        )

    @override
    def base_scheme(
        self,
        x0: NDArray[np.floating],
        t0: float,
        t_max: float,
        n_steps: int,
        jac0: NDArray[np.floating] | None = None,
    ) -> tuple[NDArray[np.floating], bool]:
        """modified midpoint method with the posibility of Gragg's smoothing"""
        delta_t = (t_max - t0) / n_steps
        x_2prev = x0
        x_prev = x0
        x_n = x0 + delta_t * self.ode_fun(t0, x_prev)  # start with an Euler step
        t_n = t0 + delta_t
        for _ in range(1, n_steps):
            x_prev = x_n
            delta_x = 2 * delta_t * self.ode_fun(t_n, x_prev)
            x_n = x_2prev + delta_x
            x_2prev = x_prev
            t_n += delta_t
        if self.use_smoothing:
            x_n = 0.5 * (x_n + x_prev + delta_t * self.ode_fun(t_n, x_n))
        return x_n, False


class LimplicitEulerExtrapolation(ExtrapolationSolver):
    """Extrapolation with the linearly-implicit Euler method as the underlying scheme and therefore usable for stiff problems. With default config similar to the SEULEX code of Hairer&Wanner"""

    def __init__(
        self,
        ode_fun: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
        num_odes: int,
        table_size: int = 10,
        jac_fun: (
            Callable[[float, NDArray[np.floating]], NDArray[np.floating]] | None
        ) = None,
        step_controller: StepControllerExtrap | None = None,
        step_seq: (
            NDArray[np.integer]
            | Literal["harmonic", "Romberg", "Bulirsch", "harmonic2", "fours", "SODEX"]
        ) = "harmonic2",
        mass_matrix: NDArray[np.floating] | None = None,
        implicit_rel_costs: ImplicitRelCosts | None = None,
        dtype: DTypeLike = np.double,
    ):
        super().__init__(
            ode_fun=ode_fun,
            step_seq=step_seq,
            is_symmetric=False,
            require_jacobian=True,
            fevals_per_step=step_seq,
            table_size=table_size,
            num_odes=num_odes,
            jac_fun=jac_fun,
            mass_matrix=mass_matrix,
            step_controller=step_controller,
            implicit_rel_costs=implicit_rel_costs,
            dtype=dtype,
        )
        self.norm = self.step_controller.norm

    @override
    def base_scheme(
        self,
        x0: NDArray[np.floating],
        t0: float,
        t_max: float,
        n_steps: int,
        jac0: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], bool]:
        r"""calculates the specified number of steps with the linearly-implicit euler scheme (Rosenbrock-like) (I - \Delta t J) x^{n+1} = \Delta t f(x^n) with a constant jacobian evaluated at x0"""
        delta_t = (t_max - t0) / n_steps
        lu_and_piv = lu_factor(self.mass_matrix - delta_t * jac0)

        x_n = x0
        t_n = t0
        delta_x_0: NDArray[np.floating]
        for n in range(n_steps):
            rhs = delta_t * self.ode_fun(t_n, x_n)
            delta_x: NDArray[np.floating] = lu_solve(
                lu_and_piv, rhs, overwrite_b=False, check_finite=False
            )  # TODO: i can not overwrite b here because of the convergence check
            x_n = x_n + delta_x
            t_n += delta_t

            if n == 0:
                delta_x_0 = delta_x
            elif (
                n == 1
                and self.norm(
                    lu_solve(
                        lu_and_piv,
                        b=rhs
                        - delta_x_0,  # pyright: ignore[reportPossiblyUnboundVariable]
                        overwrite_b=True,
                        check_finite=False,
                    )
                    / delta_x_0  # pyright: ignore[reportPossiblyUnboundVariable]
                )
                > 1.0
            ):  # stability check
                return x_n, True
            # delta_x_prev = delta_x
        return x_n, False


class LimplicitMidpointExtrapolation(ExtrapolationSolver):
    """Extrapolation method using the linearly-implicit midpoint scheme as its base method. Applicable to stiff problems and corresponds to the SODEX scheme of Hairer&Wanner"""

    def __init__(
        self,
        ode_fun: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
        num_odes: int,
        table_size: int = 10,
        jac_fun: (
            Callable[[float, NDArray[np.floating]], NDArray[np.floating]] | None
        ) = None,
        step_controller: StepControllerExtrap | None = None,
        use_smoothing: bool = False,
        step_seq: NDArray[np.integer] | Literal["fours", "SODEX"] = "SODEX",
        mass_matrix: NDArray[np.floating] | None = None,
        implicit_rel_costs: ImplicitRelCosts | None = None,
        dtype: DTypeLike = np.double,
    ):
        self.use_smoothing = use_smoothing
        super().__init__(
            ode_fun=ode_fun,
            step_seq=step_seq,
            is_symmetric=True,
            require_jacobian=True,
            fevals_per_step=step_seq + 1 * use_smoothing,
            table_size=table_size,
            num_odes=num_odes,
            jac_fun=jac_fun,
            mass_matrix=mass_matrix,
            step_controller=step_controller,
            implicit_rel_costs=implicit_rel_costs,
            dtype=dtype,
        )
        self.norm = self.step_controller.norm

    @override
    def base_scheme(
        self,
        x0: NDArray[np.floating],
        t0: float,
        t_max: float,
        n_steps: int,
        jac0: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], bool]:
        """linearly implicit midpoint scheme with optional Gragg-smoothing"""
        delta_t = (t_max - t0) / n_steps
        lu_and_piv = lu_factor(self.mass_matrix - delta_t * jac0)

        # start with a linearized-implicit euler step
        rhs = delta_t * self.ode_fun(t0, x0)
        delta_x: NDArray[np.floating] = lu_solve(
            lu_and_piv, rhs, overwrite_b=True, check_finite=False
        )
        delta_x_0 = delta_x
        x_n = x0 + delta_x
        t_n = t0 + delta_t

        # continue with linearly implicit midpoint
        for n in range(1, n_steps):
            rhs = 2 * delta_t * (self.ode_fun(t_n, x_n) - self.mass_matrix * delta_x)
            delta_x = delta_x + lu_solve(
                lu_and_piv, rhs, overwrite_b=False, check_finite=False
            )  # TODO: i can not overwrite b here because of the convergence check
            x_n = x_n + delta_x
            t_n += delta_t

            if (
                n == 1 and self.norm(0.5 * (delta_x - delta_x_0) / delta_x_0) > 1.0
            ):  # stability check
                return x_n, True

        if (
            self.use_smoothing
        ):  # Gragg's smoothing, requires one additional step before which we save the previous value of x
            rhs = 2 * delta_t * (self.ode_fun(t_n, x_n) - self.mass_matrix * delta_x)
            delta_x = delta_x + lu_solve(
                lu_and_piv, rhs, overwrite_b=True, check_finite=False
            )
            x_n = x_n + 0.5 * delta_x

        return x_n, False

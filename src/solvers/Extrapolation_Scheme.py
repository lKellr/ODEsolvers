from abc import ABC, abstractmethod
from numpy._typing._array_like import NDArray
from typing import (
    Any,
    Callable,
    Literal,
    NamedTuple,
    override,
)
import numpy as np
from numpy.typing import DTypeLike, NDArray
import logging

from scipy.linalg import lu_factor, lu_solve

from modules.helpers import numerical_jacobian_t

from modules.step_control import (
    StepControllerExtrap,
    StepControllerExtrapKH,
    contr_ext_state_type,
)

logger = logging.getLogger(__name__)


class ImplicitRelCosts(NamedTuple):
    rel_jac_cost: float = 2.0
    rel_lu_cost: float = 1.0
    rel_backsub_cost: float = 0.0
    norm_cost: float = 0.5


class ExtrapolationSolver(ABC):
    """
    Base class from which concrete extrapolation schemes can be implemented by providing a suitable base scheme.
    Additionally, child classes must initialize the controller
    """

    # these will be initialized in _init_implicit()
    mass_matrix: NDArray[np.floating]
    implicit_rel_costs: ImplicitRelCosts
    jac_fun: Callable[[float, NDArray[np.floating]], NDArray[np.floating]]

    def __init__(
        self,
        ode_fun: Callable[
            [float, NDArray[np.floating]], NDArray[np.floating]
        ],  # TODO: does this have to be set here?
        substep_seq: (
            NDArray[np.integer]
            | list[int]
            | Literal["harmonic", "Romberg", "Bulirsch", "harmonic2", "fours", "SODEX"]
        ),
        is_symmetric: bool,
        table_size: int,
        step_controller: StepControllerExtrap | None = None,
        dtype: DTypeLike = np.double,
    ):
        if hasattr(substep_seq, "__len__") and all(
            isinstance(item, int) for item in substep_seq
        ):
            assert (
                len(substep_seq) >= table_size
            ), f"substep_sequence must contain at least as many entries as the table size k={table_size}, current size: {len(substep_seq)}"
            substep_seq = np.array(substep_seq, dtype=int)
        elif substep_seq == "harmonic":
            substep_seq = np.array(range(1, table_size + 1), dtype=int)
            if is_symmetric:
                substep_seq *= 2
        elif substep_seq == "Romberg":
            substep_seq = np.array([2**i for i in range(table_size)], dtype=int)
            if is_symmetric:
                substep_seq *= 2
        elif substep_seq == "Bulirsch":
            substep_seq = np.array(
                [
                    2 ** (k // 2) if k % 2 == 0 else 1.5 * 2 ** (k // 2)
                    for k in range(1, table_size + 1)
                ],
                dtype=int,
            )
            if is_symmetric:
                substep_seq *= 2
        elif substep_seq == "harmonic2":
            substep_seq = np.array(range(2, table_size + 2), dtype=int)
        elif (
            substep_seq == "fours"
        ):  # this sequence would allow for dense output, form Numerical Recipes
            substep_seq = np.array(range(2, 4 * table_size, 4), dtype=int)
        elif substep_seq == "SODEX":
            # according to Bader&Deulfhard1983, the ratio of subsequent entries must be greater
            # than 7/5 (empirical value) and they should all be even (+ a stronger restriction n_i = 2*(2*i+1))
            alpha = 5 / 7
            n_list = [2]
            j = 1
            while len(n_list) < table_size:
                candidate = 4 * j + 2
                if n_list[-1] / candidate <= alpha:
                    n_list.append(candidate)
                j += 1
            substep_seq = np.array(n_list, dtype=int)
        else:
            raise ValueError(f"step sequence of type {substep_seq} is not available.")

        if is_symmetric:
            assert (
                substep_seq % 2 == 0
            ).all(), "step sequence for symmetric methods must be even to reach expected convergence rates"
        self.substep_seq = substep_seq
        self.is_symmetric = is_symmetric
        self.order_exponent = 2 if is_symmetric else 1
        self.table_size: int = table_size
        self.dtype = dtype

        # not all entries are needed, only the lower? triangular part and only beginning from j=1, but i cant index a list, so this has to be a padded array
        self.coeffs_Aitken = np.array(
            [
                [
                    (
                        (
                            1.0
                            / (
                                (self.substep_seq[k] / self.substep_seq[k - j])
                                ** self.order_exponent
                                - 1.0
                            )
                        )
                        if j <= k
                        else None  # will be cast to NaN
                    )
                    for j in range(1, table_size)
                ]
                for k in range(1, table_size)
            ],
            dtype,
        )
        self.n_fevals = np.cumsum(
            [self._fevals_per_base_solve(n_ss) for n_ss in substep_seq], dtype=int
        )  # cached for solve_info

        self.impl_base_scheme = False

        if step_controller is None:
            step_controller = StepControllerExtrapKH(
                dtype=dtype,
            )
        self.step_controller = step_controller

        self.ode_fun = ode_fun

    def _init_implicit(
        self,
        num_odes: int,
        require_jacobian: bool,
        jac_fun: (
            Callable[[float, NDArray[np.floating]], NDArray[np.floating]] | None
        ) = None,
        mass_matrix: NDArray[np.floating] | None = None,
        implicit_rel_costs: ImplicitRelCosts | None = None,
    ):
        if implicit_rel_costs is None:
            self.implicit_rel_costs = ImplicitRelCosts()
        else:
            self.implicit_rel_costs = implicit_rel_costs

        self.impl_base_scheme = require_jacobian
        if jac_fun is None and self.impl_base_scheme:
            self.jac_fun = lambda t, x: numerical_jacobian_t(
                t, x, self.ode_fun, delta=1e-12
            )
            if implicit_rel_costs is None:
                self.implicit_rel_costs = ImplicitRelCosts(rel_jac_cost=num_odes + 1)
        elif self.impl_base_scheme:
            assert jac_fun is not None
            self.jac_fun = jac_fun

        if mass_matrix is None:
            self.mass_matrix = np.identity(num_odes, dtype=self.dtype)
        else:
            mm_shape = mass_matrix.shape
            assert (
                len(mm_shape) == 2
                and (mm_shape[0] == mm_shape[1])
                and (mm_shape[0] == num_odes)
            ), "mass matrix must be square and dimensions must match num_odes"
            self.mass_matrix = mass_matrix

    def _init_controller(self, total_feval_cost_for_k: NDArray[np.floating]):
        err_reduction_at_step = np.array(
            [
                (self.substep_seq[k] / self.substep_seq[0]) ** self.order_exponent
                for k in range(1, self.table_size)
            ],
            dtype=self.dtype,
        )  # NOTE: first entry is never used

        self.step_controller.initialize_scheme(
            self.is_symmetric,
            self.table_size,
            err_reduction_at_step,
            total_feval_cost_for_k,
        )

    def fill_extrapolation_table(
        self,
        T_fine_first_order: NDArray[np.floating],
        T_table_k: NDArray[np.floating],
        k: int,
    ) -> None:
        """Increases the accuracy of the estimate for x0 by one order in the stepsize with the help of Richardson extrapolation.
        For this, the number of steps has to be increased over the previous order. Approximations of all orders lower than the target order are computed with this number of steps.
        The function fills a table of the computed approximations to reuse in the next order-increasing step.
        """
        T_extrap = T_fine_first_order

        # perform repeated Richardson extrapolation until the target order has been reached,
        # T_table_k starts with lower resolution approximations from a previous extrapolation
        # step and is progressively filled with extrapolated values (and the low order solver result)
        for j in range(0, k):
            # extraction from array for readability:
            T_coarselow = T_table_k[j]
            T_finelow = T_extrap

            T_extrap = (
                T_finelow + (T_finelow - T_coarselow) * self.coeffs_Aitken[k - 1, j]
            )
            T_table_k[j] = T_finelow
        T_table_k[k] = T_extrap

    def extrapolation_step(
        self,
        t_curr: float,
        x_curr: NDArray[np.floating],
        k_target: int,
        step_size: float,
        allow_early_check: bool = False,
    ) -> tuple[NDArray[np.floating], contr_ext_state_type, int, dict[str, Any]]:
        """Performs an extrapolation step of x0 until t + step_size"""
        error: NDArray[np.floating] = np.empty_like(x_curr)

        step_info: dict[str, Any] = dict(
            stop_reason = "continue",
            n_feval=0,
            n_jaceval=0,
            n_lu=0,
            local_error=np.nan,
            local_order=-1,
            max_substeps=np.nan,
        )
        # calculate initial jacobian, will be reused at the start of each extrapolation step
        jac0 = None
        if (
            self.impl_base_scheme
        ):  # TODO: recompute only if theta is above some tolerance, reuse if it is a retry!
            jac0 = self.jac_fun(t_curr, x_curr)

        # this is allocated with max size, alternative would be to extend the size each loop iteration, not sure if this would be smart in terms of repeated allocation performance cost
        T_table_k = np.empty((self.table_size, x_curr.shape[0]), self.dtype)
        T_table_k[0], is_diverging = self.base_scheme(
            x_curr,
            t_curr,
            t_max=t_curr + step_size,
            n_steps=self.substep_seq[0],
            jac0=jac0,
        )

        state: contr_ext_state_type = "continue" if not is_diverging else "divergence"
        k_curr = 0
        while state == "continue":
            k_curr += 1
            # Basic operations: compute with more steps, then fill row in tableau
            T_fine_base_order, is_diverging = self.base_scheme(
                x_curr,
                t_curr,
                t_max=t_curr + step_size,
                n_steps=self.substep_seq[k_curr],
                jac0=jac0,
            )
            if(is_diverging): # early exit: we don't have to calculate the next result and check the error if we are already diverging
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"Early exit in stage {k_curr} due to divergence in the solver"
                    )
                state = "divergence"
                break

            # last_table_diag= T_table_k[k_curr-1] # needs to be cached for advanced error computation
            self.fill_extrapolation_table(T_fine_base_order, T_table_k, k_curr)

            error = np.abs(T_table_k[k_curr - 1] - T_table_k[k_curr]) # subdiagonal
            # error_d = np.abs(last_table_diag - T_table_k[k_curr]) # diagonal
            # error_de = np.abs(last_table_diag - T_table_k[k_curr])/self.step_controller.err_reduction_at_step[k_curr-1] # diagonal-extrapolated

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Stage reached: {k_curr}, error: {error}")

            # exit conditions
            state = self.step_controller.evaluate_step(
                error, x_curr, T_table_k[k_curr], k_curr, k_target, allow_early_check
            )

        step_info["stop_reason"] = state
        step_info["n_feval"] = self.n_fevals[k_curr]
        step_info["n_lu"] = self.impl_base_scheme * (
            k_curr + 1
        )  # When a Jacobian is present, i also perform a LU factorization
        step_info["n_jaceval"] = 1 * self.impl_base_scheme
        step_info["local_error"] = error
        step_info["local_order"] = (k_curr + (not is_diverging))*self.order_exponent
        step_info["max_substeps"] = self.substep_seq[k_curr]

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Finished step\n"
                + "\n".join([f"{k}: {v}" for k, v in step_info.items()])
                + "\n"
            )

        return (
            T_table_k[k_curr],
            state,
            k_curr,
            step_info,
        )

    def solve(
        self,
        x0: NDArray[np.floating],
        t_max: float,
        t0: float = 0,
        k_initial: int | None = None,
        h_initial: float | None = None,
        log_period: float | None = None,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], dict[str, Any]]:
        """Solve the ODE with initial condition x0 from time t0 to t_max. If initial step size and order are not specified, heuristics provided by the step controller are used"""

        solve_info: dict[str, Any] = dict(
            n_steps=0,
            n_feval=0,
            n_jaceval=0,
            n_lu=0,  # initial one for implicit ODEs is not counted
            n_restarts=0,
            local_errors=[],
            local_orders=[],
        )

        k_target: int
        step: float

        if k_initial is None:
            k_target = self.step_controller.get_initial_ktarget()
        else:
            k_target = k_initial

        if h_initial is None:
            step = self.step_controller.get_initial_stepHW(
                self.ode_fun, x0, t0=t0, p=k_target + 1
            )
        else:
            step = h_initial

        assert step > 0, f"invalid initial step size {step}"
        assert (
            k_target >= self.step_controller.k_min
            and k_target <= self.step_controller.k_max
        ), f"invalid initial target order {k_target}"

        if log_period is None:
            log_period = step / (t_max - t0) * 50

        logger.info(f"Beginning solve.")

        time = [t0]
        solution = [x0.astype(self.dtype)]

        allow_early_check = True  # allow quick order variation for first and last steps when target order is not optimal
        current_time = t0
        t_last_log = t0
        while current_time < t_max:
            if (
                current_time + step > t_max
            ):  # shorten h if we would go further than necessary # TODO: this is inefficient since order selection is suboptimal
                step = t_max - current_time
                allow_early_check = True
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Starting step at time {current_time:.3f} of {t_max} with k_target = {k_target}, h = {step:.2E}"
                )

            # do step
            new_solution, state, k_final, step_info = self.extrapolation_step(
                current_time,
                solution[-1],
                k_target,
                step,
                allow_early_check,
            )
            # reset full order variation
            if allow_early_check:
                allow_early_check = False
            if (
                state == "divergence"
            ):  # smaller step sizes might lead to earlier convergence, therefore we should check earlier
                allow_early_check = True

            # update info
            solve_info["n_steps"] += 1
            solve_info["n_feval"] += step_info["n_feval"]
            solve_info["n_jaceval"] += step_info["n_jaceval"]
            solve_info["n_lu"] += step_info["n_lu"]

            # check for acceptance
            if state == "accepted":
                # save solution
                current_time += step
                solution.append(new_solution)
                time.append(current_time)
                solve_info["local_errors"].append(
                    self.step_controller.norm(step_info["local_error"])
                )
                solve_info["local_orders"].append(step_info["local_order"])
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Accepted step, continuing.")
            else:
                solve_info["n_restarts"] += 1
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Rejected step, restarting.")
            # finding parameters for the next step
            k_target, next_step_multiplier = (
                self.step_controller.get_next_step_parameters(k_final, k_target, state)
            )
            step *= next_step_multiplier

            if current_time >= t_last_log + log_period:
                logger.info(
                    f"Current time {current_time:.3f} of {t_max}.\n steps = {solve_info['n_steps']}, restarts = {solve_info['n_restarts']}, stages: {k_target}"
                )
                t_last_log = current_time

        # finished
        logger.info(
            "Finished\n"
            + "\n".join(
                [f"{k}: {v}" for k, v in solve_info.items() if not isinstance(v, list)]
            )
            + "\n"
        )

        return np.array(time, self.dtype), np.array(solution, self.dtype), solve_info

    def _fevals_per_base_solve(self, n_substeps: int) -> int:
        return n_substeps

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
    """Extrapolation with Euler's method as the underlying scheme. With default config similar to the EULEX code of Deuflhard1983 and Hairer1992"""

    def __init__(
        self,
        ode_fun: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
        table_size: int = 10,
        step_controller: StepControllerExtrap | None = None,
        substep_seq: (
            NDArray[np.integer]
            | Literal["harmonic", "Romberg", "Bulirsch", "harmonic2", "fours", "SODEX"]
        ) = "harmonic",
        dtype: DTypeLike = np.double,
    ):
        super().__init__(
            ode_fun=ode_fun,
            substep_seq=substep_seq,
            is_symmetric=False,
            table_size=table_size,
            step_controller=step_controller,
            dtype=dtype,
        )
        total_feval_cost_for_k = np.cumsum(self.substep_seq * 1.0, dtype=self.dtype)
        self._init_controller(total_feval_cost_for_k)

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


class EulerExtrapolationMass(ExtrapolationSolver):
    """Extrapolation with Euler's method as the underlying scheme. With default config similar to the EULEX code of Deuflhard1983 and Hairer1992.
    This version allows for the specification of a mass matrix for solving implicit ODEs
    """

    def __init__(
        self,
        ode_fun: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
        mass_matrix: NDArray[np.floating],
        table_size: int = 10,
        step_controller: StepControllerExtrap | None = None,
        substep_seq: (
            NDArray[np.integer]
            | Literal["harmonic", "Romberg", "Bulirsch", "harmonic2", "fours", "SODEX"]
        ) = "harmonic",
        implicit_rel_costs: ImplicitRelCosts | None = None,
        dtype: DTypeLike = np.double,
    ):
        super().__init__(
            ode_fun=ode_fun,
            substep_seq=substep_seq,
            is_symmetric=False,
            table_size=table_size,
            step_controller=step_controller,
            dtype=dtype,
        )
        self._init_implicit(
            num_odes=mass_matrix.shape[0],
            require_jacobian=False,
            mass_matrix=mass_matrix,
            implicit_rel_costs=implicit_rel_costs,
        )
        total_feval_cost_for_k = np.cumsum(
            self.substep_seq * (1.0 + self.implicit_rel_costs.rel_backsub_cost),
            dtype=self.dtype,
        )
        self._init_controller(total_feval_cost_for_k)

        self.lu_and_piv_mass = lu_factor(mass_matrix)

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
            delta_x = lu_solve(
                self.lu_and_piv_mass,
                delta_t * self.ode_fun(t_n, x_n),
                overwrite_b=True,
                check_finite=False,
            )

            x_n = x_n + delta_x
            t_n += delta_t
        return x_n, False


class EulerExtrapolationRational(EulerExtrapolation):
    """Extrapolation with Euler's method as the underlying scheme. This modification uses rational instead of polynomial extrapolation"""

    def __init__(
        self,
        ode_fun: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
        table_size: int = 10,
        step_controller: StepControllerExtrap | None = None,
        substep_seq: (
            NDArray[np.integer]
            | Literal["harmonic", "Romberg", "Bulirsch", "harmonic2", "fours", "SODEX"]
        ) = "harmonic",
        dtype: DTypeLike = np.double,
    ):
        super().__init__(
            ode_fun=ode_fun,
            table_size=table_size,
            step_controller=step_controller,
            substep_seq=substep_seq,
            dtype=dtype,
        )
        self.coeffs_extrap = np.array(
            [
                [
                    (
                        (
                            (self.substep_seq[k] / self.substep_seq[k - j])
                            ** self.order_exponent
                            - 1.0
                        )
                        if j <= k
                        else None  # will be cast to NaN
                    )
                    for j in range(1, table_size)
                ]
                for k in range(1, table_size)
            ],
            dtype,
        )

    @override
    def fill_extrapolation_table(
        self,
        T_fine_first_order: NDArray[np.floating],
        T_table_k: NDArray[np.floating],
        k: int,
    ) -> None:
        """Increases the accuracy of the estimate for x0 by one order in the stepsize with the help of Richardson extrapolation.
        For this, the number of steps has to be increased over the previous order. Approximations of all orders lower than the target order are computed with this number of steps.
        The function fills a table of the computed approximations to reuse in the next order-increasing step.

        This variant uses rational instead of polynomial extraplation
        """
        T_extrap = T_fine_first_order
        T_coarselow = 0.0

        # perform repeated Richardson extrapolation until the target order has been reached,
        # T_table_k starts with lower resolution approximations from a previous extrapolation
        # step and is progressively filled with extrapolated values (and the low order solver result)
        for j in range(0, k):
            # extraction from array for readability:
            T_coarselowlow = T_coarselow
            T_coarselow = T_table_k[j]
            T_finelow = T_extrap

            T_extrap = T_finelow + (T_finelow - T_coarselow) / (
                self.coeffs_extrap[k - 1, j]
                * (1.0 - (T_finelow - T_coarselow) / (T_finelow - T_coarselowlow))
                - 1.0
            )
            T_table_k[j] = T_finelow
        T_table_k[k] = T_extrap


class ModMidpointExtrapolation(ExtrapolationSolver):
    """This extrapolation method is based around the modified midpoint method. As it is symmetric, the order increases by a factor of 2 in each extrapolation step.
    For this, the "harmonic", "Romberg" and "Bulirsch" substep sequences are made even by doubling each entry.
    With default config similar to the ODEX code of Hairer1992 or DIFEX1 of Deuflhard1983. Also similar to the Bulirsch-Stoer method but without rational extrapolation.
    """

    def __init__(
        self,
        ode_fun: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
        table_size: int = 10,
        step_controller: StepControllerExtrap | None = None,
        use_smoothing: bool = False,
        substep_seq: (
            NDArray[np.integer]
            | Literal["harmonic", "Romberg", "Bulirsch", "harmonic2", "fours", "SODEX"]
        ) = "harmonic",
        dtype: DTypeLike = np.double,
    ):
        self.use_smoothing = use_smoothing
        super().__init__(
            ode_fun=ode_fun,
            substep_seq=substep_seq,
            is_symmetric=True,
            table_size=table_size,
            step_controller=step_controller,
            dtype=dtype,
        )
        total_feval_cost_for_k = np.cumsum(
            (self.substep_seq + self.use_smoothing) * 1.0, dtype=self.dtype
        )
        self._init_controller(total_feval_cost_for_k)

    @override
    def _fevals_per_base_solve(self, n_substeps: int) -> int:
        return n_substeps + self.use_smoothing

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


class ModMidpointExtrapolationMass(ExtrapolationSolver):
    """This extrapolation method is based around the modified midpoint method. As it is symmetric, the order increases by a factor of 2 in each extrapolation step.
    For this, the "harmonic", "Romberg" and "Bulirsch" substep sequences are made even by doubling each entry.
    With default config a Bulirsch-Stoer method similar to the ODEX code of Hairer1992 or DIFEX1 of Deuflhard1983. Also similar to the Bulirsch-Stoer method but without rational extrapolation.
    This version allows for the specification of a mass matrix for solving implicit ODEs
    """

    def __init__(
        self,
        ode_fun: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
        mass_matrix: NDArray[np.floating],
        table_size: int = 10,
        step_controller: StepControllerExtrap | None = None,
        use_smoothing: bool = False,
        substep_seq: (
            NDArray[np.integer]
            | Literal["harmonic", "Romberg", "Bulirsch", "harmonic2", "fours", "SODEX"]
        ) = "harmonic",
        implicit_rel_costs: ImplicitRelCosts | None = None,
        dtype: DTypeLike = np.double,
    ):
        self.use_smoothing = use_smoothing
        super().__init__(
            ode_fun=ode_fun,
            substep_seq=substep_seq,
            is_symmetric=True,
            table_size=table_size,
            step_controller=step_controller,
            dtype=dtype,
        )
        self._init_implicit(
            num_odes=mass_matrix.shape[0],
            require_jacobian=False,
            mass_matrix=mass_matrix,
            implicit_rel_costs=implicit_rel_costs,
        )
        total_feval_cost_for_k = np.cumsum(
            (self.substep_seq + self.use_smoothing)
            * (1.0 + self.implicit_rel_costs.rel_backsub_cost),
            dtype=self.dtype,
        )
        self._init_controller(total_feval_cost_for_k)

        self.lu_and_piv_mass = lu_factor(mass_matrix)

    @override
    def _fevals_per_base_solve(self, n_substeps: int) -> int:
        return n_substeps + self.use_smoothing

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
        x_n = x0 + lu_solve(
            self.lu_and_piv_mass,
            delta_t * self.ode_fun(t0, x_prev),
            overwrite_b=True,
            check_finite=False,
        )  # start with an Euler step
        t_n = t0 + delta_t
        for _ in range(1, n_steps):
            x_prev = x_n
            delta_x = lu_solve(
                self.lu_and_piv_mass,
                2 * delta_t * self.ode_fun(t_n, x_n),
                overwrite_b=True,
                check_finite=False,
            )
            x_n = x_2prev + delta_x
            x_2prev = x_prev
            t_n += delta_t
        if self.use_smoothing:
            x_n = 0.5 * (
                x_n
                + x_2prev
                + lu_solve(
                    self.lu_and_piv_mass,
                    delta_t * self.ode_fun(t_n, x_n),
                    overwrite_b=True,
                    check_finite=False,
                )
            )
        return x_n, False


class LimplicitEulerExtrapolation(ExtrapolationSolver):
    """Extrapolation with the linearly-implicit Euler method as the underlying scheme and therefore usable for stiff problems.
    With default config similar to the SEULEX code of Hairer1992.
    """

    def __init__(
        self,
        ode_fun: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
        table_size: int = 10,
        jac_fun: (
            Callable[[float, NDArray[np.floating]], NDArray[np.floating]] | None
        ) = None,
        step_controller: StepControllerExtrap | None = None,
        substep_seq: (
            NDArray[np.integer]
            | Literal["harmonic", "Romberg", "Bulirsch", "harmonic2", "fours", "SODEX"]
        ) = "harmonic2",
        mass_matrix: NDArray[np.floating] | None = None,
        num_odes: int | None = None,
        implicit_rel_costs: ImplicitRelCosts | None = None,
        dtype: DTypeLike = np.double,
    ):
        super().__init__(
            ode_fun=ode_fun,
            substep_seq=substep_seq,
            is_symmetric=False,
            table_size=table_size,
            step_controller=step_controller,
            dtype=dtype,
        )
        self.norm = self.step_controller.norm

        if num_odes is None:
            assert (
                mass_matrix is not None
            ), "either mass matrix or the number of ODEs has to be specified"
            num_odes = mass_matrix.shape[0]

        self._init_implicit(
            num_odes=num_odes,  # type: ignore
            require_jacobian=True,
            jac_fun=jac_fun,
            mass_matrix=mass_matrix,
            implicit_rel_costs=implicit_rel_costs,
        )

        feval_cost_per_k = (
            self.substep_seq * (1.0 + self.implicit_rel_costs.rel_backsub_cost)
            + self.implicit_rel_costs.rel_lu_cost
            + (
                self.implicit_rel_costs.rel_backsub_cost
                + self.implicit_rel_costs.norm_cost
            )
            * (self.substep_seq >= 2)  # cost of stability check (lu_backsub + norm)
        )
        total_feval_cost_for_k = (
            np.cumsum(feval_cost_per_k, dtype=self.dtype)
            + self.implicit_rel_costs.rel_jac_cost
        )
        self._init_controller(total_feval_cost_for_k)

    @override
    def base_scheme(
        self,
        x0: NDArray[np.floating],
        t0: float,
        t_max: float,
        n_steps: int,
        jac0: NDArray[np.floating] | None,
    ) -> tuple[NDArray[np.floating], bool]:
        r"""calculates the specified number of steps with the linearly-implicit euler scheme (Rosenbrock-like) (I - \Delta t J) x^{n+1} = \Delta t f(x^n) with a constant jacobian evaluated at x0"""
        assert jac0 is not None

        delta_t = (t_max - t0) / n_steps
        lu_and_piv = lu_factor(self.mass_matrix - delta_t * jac0)

        x_n = x0
        t_n = t0
        delta_x_0: NDArray[np.floating]
        for n in range(n_steps):
            rhs = delta_t * self.ode_fun(t_n, x_n)
            delta_x: NDArray[np.floating] = lu_solve(
                lu_and_piv, rhs, overwrite_b=False, check_finite=False
            )  # NOTE: i can not overwrite b here because of the convergence check
            x_n += delta_x
            t_n += delta_t

            if n == 0:
                delta_x_0 = delta_x # cache for stability check
            elif (n == 1):  # stability check
                theta = lu_solve( # NOTE: I calculate the norm after component-wise division instead of the ratio of the norms
                        lu_and_piv,
                        b=rhs - delta_x_0,  # pyright: ignore[reportPossiblyUnboundVariable]
                        overwrite_b=True,
                        check_finite=False,
                    ) / delta_x_0  # pyright: ignore[reportPossiblyUnboundVariable]
                if self.norm(theta) > 1.0:
                    return x_n, True
            # delta_x_prev = delta_x
        return x_n, False


class LimplicitMidpointExtrapolation(ExtrapolationSolver):
    """Extrapolation method using the linearly-implicit midpoint scheme as its base method.
    Applicable to stiff problems and corresponds to the SODEX scheme of Hairer1992 orMETAn1 of Deuflhard1983.

    The "harmonic", "Romberg" and "Bulirsch" substep sequences are made even by doubling each entry.
    """

    def __init__(
        self,
        ode_fun: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
        table_size: int = 7,
        jac_fun: (
            Callable[[float, NDArray[np.floating]], NDArray[np.floating]] | None
        ) = None,
        step_controller: StepControllerExtrap | None = None,
        use_smoothing: bool = False,
        substep_seq: (
            NDArray[np.integer]
            | Literal["harmonic", "Romberg", "Bulirsch", "fours", "SODEX"]
        ) = "SODEX",
        mass_matrix: NDArray[np.floating] | None = None,
        num_odes: int | None = None,
        implicit_rel_costs: ImplicitRelCosts | None = None,
        dtype: DTypeLike = np.double,
    ):
        self.use_smoothing = use_smoothing
        super().__init__(
            ode_fun=ode_fun,
            substep_seq=substep_seq,
            is_symmetric=True,
            table_size=table_size,
            step_controller=step_controller,
            dtype=dtype,
        )
        self.norm = self.step_controller.norm

        if num_odes is None:
            assert (
                mass_matrix is not None
            ), "either mass matrix or the number of ODEs has to be specified"
            num_odes = mass_matrix.shape[0]
        self._init_implicit(
            num_odes=num_odes,  # type: ignore
            require_jacobian=True,
            jac_fun=jac_fun,
            mass_matrix=mass_matrix,
            implicit_rel_costs=implicit_rel_costs,
        )

        feval_cost_per_k = (
            (self.substep_seq + self.use_smoothing)
            * (1.0 + self.implicit_rel_costs.rel_backsub_cost)
            + self.implicit_rel_costs.rel_lu_cost
            + self.implicit_rel_costs.norm_cost
            * (self.substep_seq >= 2)  # cost of stability check (norm)
        )
        total_feval_cost_for_k = (
            np.cumsum(feval_cost_per_k, dtype=self.dtype)
            + self.implicit_rel_costs.rel_jac_cost
        )
        self._init_controller(total_feval_cost_for_k)

    @override
    def base_scheme(
        self,
        x0: NDArray[np.floating],
        t0: float,
        t_max: float,
        n_steps: int,
        jac0: NDArray[np.floating] | None,
    ) -> tuple[NDArray[np.floating], bool]:
        """linearly implicit midpoint scheme with optional Gragg-smoothing"""
        assert jac0 is not None

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
            rhs = 2 * delta_t * (self.ode_fun(t_n, x_n) - self.mass_matrix @ delta_x)
            delta_x = delta_x + lu_solve(
                lu_and_piv, rhs, overwrite_b=False, check_finite=False
            )  # NOTE: i can not overwrite b here because of the convergence check
            x_n += delta_x
            t_n += delta_t

            if (n == 1): # stability check
                theta = 0.5 * (delta_x - delta_x_0) / delta_x_0 # NOTE: I calculate the norm after component-wise division instead of the ratio of the norms
                if self.norm(theta) > 1.0:  
                    return x_n, True

        if (
            self.use_smoothing
        ):  # Gragg's smoothing, requires one additional step before which we save the previous value of x
            rhs = 2 * delta_t * (self.ode_fun(t_n, x_n) - self.mass_matrix @ delta_x)
            delta_x = delta_x + lu_solve(
                lu_and_piv, rhs, overwrite_b=True, check_finite=False
            )
            x_n = x_n + 0.5 * delta_x

        return x_n, False

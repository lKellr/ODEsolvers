from typing import Any, Callable, Literal, override
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
import logging

from scipy.linalg import lu_factor, lu_solve

from modules import step_control
from modules.helpers import clip, numerical_jacobian, norm_hairer, numerical_jacobian_t
from modules.step_control import (
    ImplicitRelCosts,
    StepControllerExtrap,
    StepControllerExtrapKH,
    tab_state_type,
)

logger = logging.getLogger(__name__)


class Extrapolation_Solver(ABC):
    def __init__(
        self,
        step_seq: (
            NDArray[np.integer]
            | Literal["harmonic", "Romberg", "Bulirsch", "harmonic2", "fours", "SODEX"]
        ),
        is_symmetric: bool,
        is_implicit: bool,
        ode_fun: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
        num_odes: int,
        jac_fun: Callable[[float, NDArray[np.floating]], NDArray[np.floating]] | None,
        table_size: int,
        fevals_per_step: (
            NDArray[np.integer] | None
        ),  # most of the time = step_seq, possible one more (Gragg smoothing) for symmetric methods
        mass_matrix: NDArray[np.floating] | None,
        step_controller: StepControllerExtrap | None,
        dtype: np.floating = np.double,
    ):
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
            )  # NOTE: i am not sure if a formula exists for these; they have to be multiples of even numbers, according to Bader&Deulfhard1983, the ratio iof subsequent entries must lie between 1 and 5/7
        else:
            raise ValueError(f"step sequence of type {step_seq} is not available.")

        self.table_size: int = table_size
        self.num_odes = num_odes
        self.dtype = dtype
        self.ode_fun = ode_fun
        if jac_fun is None:
            self.jac_fun = lambda t, x: numerical_jacobian_t(t, x, ode_fun, delta=1e-8)
        else:
            self.jac_fun = jac_fun

        self.mass_matrix = (
            mass_matrix if mass_matrix is not None else np.identity(num_odes, dtype)
        )

        self.is_symmetric = is_symmetric
        self.is_implicit = is_implicit
        # self.base_order = base_order
        self.step_seq = step_seq
        # not all entries are needed, only the lower? triangular part and only beginning from j=1, but i cant index a list, so this has to be a padded array
        self.coeffs_Aitken = np.array(
            [
                [
                    (
                        (
                            1.0
                            / (self.step_seq[j] / self.step_seq[j - k])
                            ** (2.0 if is_symmetric else 1.0)
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

        if fevals_per_step is None:
            fevals_per_step = step_seq  # + is_symmetric # symmetric methods usually use Gragg smoothing
        self.fevals_per_step = fevals_per_step

        if step_controller is None:
            implicit_rel_costs = None
            if self.is_implicit:
                implicit_rel_costs = ImplicitRelCosts(  # TODO: tune these
                    rel_jac_cost=self.num_odes + 1,
                    rel_lu_cost=1.0,
                    rel_backsub_cost=0.0,
                )
            elif mass_matrix is not None:
                implicit_rel_costs = ImplicitRelCosts(
                    rel_jac_cost=0.0, rel_lu_cost=1.0, rel_backsub_cost=0.0
                )

            self.step_controller = StepControllerExtrapKH(
                table_size,
                step_seq,
                is_symmetric,
                fevals_per_step,
                dtype,
                implicit_rel_costs=implicit_rel_costs,
            )
        else:
            self.step_controller = step_controller

    @abstractmethod
    def base_scheme(
        self,
        U0: NDArray[np.floating],
        t0: float,
        t_max: float,
        n_steps: int,
        jac0: NDArray[np.floating] | None,
    ) -> NDArray[np.floating]:
        raise NotImplementedError()

    def fill_extrapolation_table(
        self, T_fine_first_order, T_table_k, n_columns
    ) -> None:
        """Increases the accuracy of the estimate for U0 by one order in the stepsize with the help of Richardson extrapolation.
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
        """Performs an extrapolation step of U0 until t + step_size"""
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
        if self.is_implicit:
            jac0 = self.jac_fun(t_curr, x_curr.astype(self.dtype))

        # this is allocated with max size, alternative would be to extend the size each loop iteration, not sure if this would be smart in terms of repeated allocation performance cost
        T_table_k = np.empty((self.table_size, self.num_odes), self.dtype)
        T_table_k[0] = self.base_scheme(
            x_curr,
            t_curr,
            t_max=t_curr + step_size,
            n_steps=self.step_seq[0],
            jac0=jac0,
        )

        state: tab_state_type = "continue"
        next_k = -1
        next_step_mult = -1.0
        iterator_table = 0
        while state == "continue":
            iterator_table += 1
            # Basic operations: compute with more steps, then fill row in tableau
            T_fine_first_order = self.base_scheme(
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
            if (
                self.is_implicit and iterator_table >= 2 and np.any(err >= err_prev)
            ):  # Hairer & Wanner divergence monitor a)
                next_k = k_target  # not sure if this is the correct way to do this, maybe k should even be decreased?
                next_step_mult = self.step_controller.step_multiplier_divergence
                self.step_controller.is_retry = True  # set retry flag so that order and step size are not allowed to increase during retry
                state = "retry"

        step_info["stop_reason"] = (  # TODO:distinguish slow convergence and divergence
            "success" if state == "accepted" else "nonconvergence"
        )
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
        U0: NDArray[np.floating],
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
                self.ode_fun, U0, t0=t0, p=k_target
            )
        else:
            k_target = params_step0[0]
            step = params_step0[1]

        time = [t0]
        solution = [U0]

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


class EULEX(Extrapolation_Solver):
    def __init__(
        self,
        ode_fun: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
        num_odes: int,
        jac_fun: (
            Callable[[float, NDArray[np.floating]], NDArray[np.floating]] | None
        ) = None,
        mass_matrix: NDArray[np.floating] | None = None,
        atol: float = 1e-11,
        rtol: float = 1e-5,
        table_size: int = 8,
        step_seq: (
            NDArray[np.integer]
            | Literal["harmonic", "Romberg", "Bulirsch", "harmonic2", "fours", "SODEX"]
        ) = "harmonic",
        step_controller: StepControllerExtrap | None = None,
        dtype: np.floating = np.double,
    ):

        if mass_matrix is not None:
            self.lu_and_piv_mass = lu_factor(self.mass_matrix)
            self.base_scheme = self.base_scheme_mass  # TODO: this is dangerous

        super().__init__(
            step_seq=step_seq,
            is_symmetric=False,
            is_implicit=False,
            fevals_per_step=step_seq,
            ode_fun=ode_fun,
            num_odes=num_odes,
            jac_fun=jac_fun,
            table_size=table_size,
            mass_matrix=mass_matrix,
            step_controller=step_controller,
            dtype=dtype,
        )

    @override
    def base_scheme(
        self,
        U0: NDArray[np.floating],
        t0: float,
        t_max: float,
        n_steps: int,
        jac0: NDArray[np.floating] | None = None,
    ) -> NDArray[np.floating]:
        delta_t = (t_max - t0) / n_steps

        U_n = U0
        t_n = t0
        for _ in range(n_steps):
            delta_U = delta_t * self.ode_fun(t_n, U_n)
            U_n = U_n + delta_U
            t_n += delta_t
        return U_n

    def base_scheme_mass(
        self,
        U0: NDArray[np.floating],
        t0: float,
        t_max: float,
        n_steps: int,
        jac0: NDArray[np.floating] | None = None,
    ) -> NDArray[np.floating]:
        delta_t = (t_max - t0) / n_steps

        U_n = U0
        t_n = t0
        for _ in range(n_steps):
            delta_U = lu_solve(
                self.lu_and_piv_mass,
                delta_t * self.ode_fun(t_n, U_n),
                overwrite_b=True,
                check_finite=False,
            )
            U_n = U_n + delta_U
            t_n += delta_t
        return U_n


class ODEX(Extrapolation_Solver):
    def __init__(
        self,
        ode_fun: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
        num_odes: int,
        jac_fun: (
            Callable[[float, NDArray[np.floating]], NDArray[np.floating]] | None
        ) = None,
        mass_matrix: NDArray[np.floating] | None = None,
        atol: float = 1e-11,
        rtol: float = 1e-5,
        table_size: int = 8,
        step_seq: (
            NDArray[np.integer]
            | Literal["harmonic", "Romberg", "Bulirsch", "harmonic2", "fours", "SODEX"]
        ) = "harmonic",
        step_controller: StepControllerExtrap | None = None,
        dtype: np.floating = np.double,
    ):

        if mass_matrix is not None:
            self.lu_and_piv_mass = lu_factor(self.mass_matrix)
            self.base_scheme = self.base_scheme_mass  # TODO: this is dangerous

        super().__init__(
            step_seq=step_seq,
            is_symmetric=True,
            is_implicit=False,
            fevals_per_step=step_seq,  # TODO: Gragg smoothing?
            ode_fun=ode_fun,
            num_odes=num_odes,
            jac_fun=jac_fun,
            table_size=table_size,
            mass_matrix=mass_matrix,
            step_controller=step_controller,
            dtype=dtype,
        )

    @override
    def base_scheme(
        self,
        U0: NDArray[np.floating],
        t0: float,
        t_max: float,
        n_steps: int,
        jac0: NDArray[np.floating] | None = None,
    ) -> NDArray[np.floating]:
        delta_t = (t_max - t0) / n_steps

        # TODO: Gragg smoothing?

        # t_n = t0
        U_2prev = U0
        U_n = U0 + delta_t * self.ode_fun(t0, U0)  # start with an Euler step
        t_n = t0 + delta_t
        for _ in range(1, n_steps):
            delta_U = 2 * delta_t * self.ode_fun(t_n, U_n)
            U_temp = U_n
            U_n = U_2prev + delta_U
            t_n += delta_t
            U_2prev = U_temp
        return U_n

    def base_scheme_mass(
        self,
        U0: NDArray[np.floating],
        t0: float,
        t_max: float,
        n_steps: int,
        jac0: NDArray[np.floating] | None = None,
    ) -> NDArray[np.floating]:
        delta_t = (t_max - t0) / n_steps

        # TODO: Gragg smoothing?

        # t_n = t0
        U_2prev = U0
        U_n = U0 + lu_solve(
            self.lu_and_piv_mass,
            delta_t * self.ode_fun(t0, U0),
            overwrite_b=True,
            check_finite=False,
        )  # start with an Euler step

        t_n = t0 + delta_t
        for _ in range(1, n_steps):
            delta_U = lu_solve(
                self.lu_and_piv_mass,
                2 * delta_t * self.ode_fun(t_n, U_n),
                overwrite_b=True,
                check_finite=False,
            )
            U_temp = U_n
            U_n = U_2prev + delta_U
            t_n += delta_t
            U_2prev = U_temp
        return U_n


class SEULEX(Extrapolation_Solver):
    def __init__(
        self,
        ode_fun: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
        num_odes: int,
        jac_fun: (
            Callable[[float, NDArray[np.floating]], NDArray[np.floating]] | None
        ) = None,
        mass_matrix: NDArray[np.floating] | None = None,
        atol: float = 1e-11,
        rtol: float = 1e-5,
        table_size: int = 8,
        step_seq: (
            NDArray[np.integer]
            | Literal["harmonic", "Romberg", "Bulirsch", "harmonic2", "fours", "SODEX"]
        ) = "harmonic2",
        step_controller: StepControllerExtrap | None = None,
        dtype: np.floating = np.double,
    ):

        super().__init__(
            step_seq=step_seq,
            is_symmetric=False,
            is_implicit=True,
            fevals_per_step=step_seq,
            ode_fun=ode_fun,
            num_odes=num_odes,
            jac_fun=jac_fun,
            table_size=table_size,
            mass_matrix=mass_matrix,
            step_controller=step_controller,
            dtype=dtype,
        )

    @override
    def base_scheme(
        self,
        U0: NDArray[np.floating],
        t0: float,
        t_max: float,
        n_steps: int,
        jac0: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        r"""calculates the specified number of steps with the linearly-implicit euler scheme (Rosenbrock-like) (I - \Delta t J) U^{n+1} = \Delta t f(U^n) with a constant jacobian evaluated at U0"""
        delta_t = (t_max - t0) / n_steps
        lu, piv = lu_factor(self.mass_matrix - delta_t * jac0)

        U_n = U0
        t_n = t0
        for _ in range(n_steps):
            rhs = delta_t * self.ode_fun(t_n, U_n)
            delta_U = lu_solve((lu, piv), rhs, overwrite_b=True, check_finite=False)
            U_n = U_n + delta_U
            t_n += delta_t
            # if n_steps == (2 or 3) and (delta_U >= delta_U_prev):
            #     # TODO: restart
            #     # TODO: thius should be chaked in the base class
            # delta_U_prev = delta_U
        return U_n


class SODEX(Extrapolation_Solver):
    def __init__(
        self,
        ode_fun: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
        num_odes: int,
        jac_fun: (
            Callable[[float, NDArray[np.floating]], NDArray[np.floating]] | None
        ) = None,
        mass_matrix: NDArray[np.floating] | None = None,
        atol: float = 1e-11,
        rtol: float = 1e-5,
        table_size: int = 8,
        step_seq: NDArray[np.integer] | Literal["fours", "SODEX"] = "SODEX",
        step_controller: StepControllerExtrap | None = None,
        dtype: np.floating = np.double,
    ):
        if isinstance(step_seq, np.ndarray):
            assert (step_seq % 2 == 0).all(), "Number of steps for SODEX must be even"

        super().__init__(
            step_seq=step_seq,
            is_symmetric=True,
            is_implicit=True,
            fevals_per_step=step_seq + 1,
            ode_fun=ode_fun,
            num_odes=num_odes,
            jac_fun=jac_fun,
            table_size=table_size,
            mass_matrix=mass_matrix,
            step_controller=step_controller,
            dtype=dtype,
        )

    @override
    def base_scheme(
        self,
        U0: NDArray[np.floating],
        t0: float,
        t_max: float,
        n_steps: int,
        jac0: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        delta_t = (t_max - t0) / n_steps
        lu, piv = lu_factor(self.mass_matrix - delta_t * jac0)

        # start with a SEULER step
        rhs = delta_t * self.ode_fun(t0, U0)
        delta_U = np.linalg.solve(self.mass_matrix - delta_t * jac0, rhs)
        U_n = U0 + delta_U
        t_n = t0 + delta_t

        # continue with linearly implicit midpoint
        for _ in range(1, n_steps):
            rhs = 2 * delta_t * (self.ode_fun(t_n, U_n) - self.mass_matrix * delta_U)
            delta_U = delta_U + lu_solve(
                (lu, piv), rhs, overwrite_b=True, check_finite=False
            )
            U_n = U_n + delta_U
            t_n += 2 * delta_t
        # Gragg's smoothing, requires one additional step before which we save the previous value of U
        U_fprev = U_n - delta_U
        rhs = 2 * delta_t * (self.ode_fun(t_n, U_n) - self.mass_matrix * delta_U)
        delta_U = delta_U + lu_solve(
            (lu, piv), rhs, overwrite_b=True, check_finite=False
        )
        U_n = U_n + delta_U
        Sh_n = 0.5 * (U_nprev + U_n)

        return Sh_n

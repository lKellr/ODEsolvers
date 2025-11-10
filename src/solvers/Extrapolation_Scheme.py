from typing import Any, Callable
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
import logging
from helpers import norm_hairer

from scipy.linalg import lu_factor, lu_solve

logger = logging.getLogger(__name__)


class Extrapolation_Solver(ABC):
    def __init__(
        self,
        step_seq: NDArray[np.integer],
        is_symmetric: bool,
        base_order: int,
        ode_fun: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
        num_odes: int,
        jac_fun: (
            Callable[[float, NDArray[np.floating]], NDArray[np.floating]] | None
        ) = None,
        mass_matrix: NDArray[np.floating] | None = None,
        atol: float = 1e-11,
        rtol: float = 1e-5,
        stepfac_min: float = 0.2,
        stepfac_max: float = 5.,
        restart_step_multiplier: float = 0.5,
        table_size: int = 8,
    ):

        self.atol = atol
        self.rtol = rtol
        self.stepfac_min = stepfac_min
        self.stepfac_max = stepfac_max
        self.restart_step_multiplier = restart_step_multiplier

        self.table_size: int = table_size
        self.num_odes = num_odes
        self.ode_fun = ode_fun
        if jac_fun is None:

            def numerical_jacobian(t, x, f, delta):
                jac = np.empty((self.num_odes, self.num_odes))
                for j in range(self.num_odes):
                    shift = np.zeros_like(x)
                    shift[j] = delta
                    jac[:, j] = (f(t, x + shift) - f(t, x)) / delta
                return jac

            self.jac_fun = lambda t, x: numerical_jacobian(t, x, ode_fun, delta=1e-8)
        else:
            self.jac_fun = jac_fun

        self.mass_matrix = (
            mass_matrix if mass_matrix is not None else np.identity(num_odes)
        )

        self.is_symmetric = is_symmetric
        # self.base_order = base_order
        self.step_seq = step_seq
        # not all entries are needed, only the lower? triangular part and only beginning from j=1, but i cant index a list, so this has to be a padded array
        self.coeffs_Aitken = np.array(
            [
                np.array(
                    [
                        (
                            (
                                1.0
                                / (self.step_seq[j - 1] / self.step_seq[j - k - 1])
                                ** (2.0 if is_symmetric else 1.0)
                                - 1.0
                            )
                            if k <= j + 1
                            else 0.0
                        )
                        for k in range(1, table_size)
                    ]
                )
                for j in range(1, table_size)
            ]
        )
        self.fevals_per_step = self.step_seq + (
                1 if self.is_symmetric else 0
            )  # might not be correct for SODEX
        total_fevals_per_step = np.cumsum([self.fevals_per_step]*(table_size-1))
        self.feval_ratios = np.array([total_fevals_per_step[i+1]/total_fevals_per_step[i] for i in range(len(total_fevals_per_step)-1)])

    @abstractmethod
    def base_scheme(
        self,
        U0: NDArray[np.floating],
        t0: float,
        t_max: float,
        n_steps: int,
        jac0: NDArray[np.floating],
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
                T_finelow + (T_finelow - T_coarselow) * self.coeffs_Aitken[n_columns, col]
            )
            T_table_k[col] = T_finelow
        T_table_k[n_columns] = T_extrap

    def extrapolation_step(
        self, t0: float, U0: NDArray[np.floating], step_size
    ) -> tuple[NDArray[np.floating], dict[str, Any]]:
        """Performs an extrapolation step of U0 until t + step_size.
        Repeated extrapolation until the target tolerance is met."""
        err = np.empty_like(U0)
        err_prev = np.empty_like(U0)  # required for overflow remedy

        # calculate initial jacobian, will be reused at the start of each extrapolation step
        jac0 = self.jac_fun(t0, U0)

        # this is allocated with max size, alternative would be to extend the size each loop iteration, not sure if this would be smart in terms of repeated allocation performance cost
        T_table_k = np.empty((self.table_size, self.num_odes))
        T_table_k[0] = self.base_scheme(
            U0, t0, t_max=t0 + step_size, n_steps=self.step_seq[0], jac0=jac0
        )

        step_info: dict[str, Any] = dict(
            n_feval=self.step_seq[0] + (1 if self.is_symmetric else 0),
            n_jaceval=1,
            n_lu=1,
            local_error=np.nan,
            max_substeps=np.nan,
        )
        iterator_table = 1
        # TODO: parallelization: compute all T_k1 in parallel, then loop to extrapolate to target order,
        # dispatch all, but discard those (unfinished) which are not needed because of already sufficiently low error
        while True:
            # Basic operations: compute with more steps, then fill row in tableau
            T_fine_first_order = self.base_scheme(
                U0,
                t0,
                t_max=step_size,
                n_steps=self.step_seq[iterator_table],
                jac0=jac0,
            )
            self.fill_extrapolation_table(T_fine_first_order, T_table_k, iterator_table)
            err_prev = err
            err = np.abs(T_table_k[iterator_table - 1] - T_table_k[iterator_table])

            step_info["n_feval"] += self.fevals_per_step[iterator_table]
            step_info["n_lu"] += 1
            logger.debug(
                f"Stage reached: {iterator_table}, error: {err}"
            )  # might be wrong for symmetric methods

            # exit conditions
            if(iterator_table == iterator_target-1) # TODO: greater equal?, why can i not check this earlier?
                tol = self.atol + self.rtol * np.maximum(np.abs(U0), np.abs(T_table_k[iterator_table]))
                err_ratio = np.linalg.norm(err/tol, ord=np.inf) # alternative: norm_hairer
                step_opt, step_opt_prev, work_per_step, work_per_step_prev
                if(err_ratio < 1): # a) Convergence in line k − 1
                    if work_per_step < 0.9*work_per_step_prev:
                        iterator_target = iterator_target
                        step = step_opt_prev*self.feval_ratio[k]
                    else:
                    iterator_target =iterator_table
                    step = 
                    step_info["exit_status"] = "Success"
                    logger.debug("Success: Tolerance reached in line k-1")
                    break
            else: # b) Convergence monitor: do we expect conergence in later steps?


            elif iterator_table >= 2 and np.any(
                err >= err_prev
            ):  # Hairer & Wanner overflow remedy a)
            # TODO: this does not work since the step sizes are then not working any more
                    step_size *= self.restart_step_multiplier
                step_info["exit_status"] = "divergence"
                logger.debug(f"Restarting with step = {step_size} due to divergence")
                break
            elif iterator_table >= self.table_size - 2:
                step_info["exit_status"] = "Failure: Maximum order reached"
                logger.debug(msg="Failure: Maximum order reached")
                break

            iterator_table += 1

        step_info["local_error"] = err
        step_info["max_substeps"] = self.step_seq[iterator_table]
        return T_table_k[iterator_table], step_info

    def solve(
        self, U0: NDArray[np.floating], t_max: float, t0: float = 0, step0: float|None = None
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], dict[str, Any]]:

        solve_info: dict[str, Any] = dict(
            n_feval=0,
            n_jaceval=0,
            n_lu=0,
            n_restarts=0,
        )

        if(step0 == None):
            step = self.initial_step(U0, t0)
        else:
            step = step0

        time = [t0]
        solution = [U0]


        current_time = t0
        while current_time < t_max:
            logger.debug(f"Starting step at time {current_time} of {t_max}")

            # do step
            new_solution, step_info = self.extrapolation_step(
                current_time, solution[-1], step
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

            # find optimal step size
            step = self.optimal_step()

            if accepted:
                solution.append(new_solution)
                time.append(current_time)
                current_time += step
            else:
                logger.debug(f"Retrying step with h={step}")
            
        # finished
        logger.debug(
            "Finished\n"
            + "\n".join([f"{k}: {v}" for k, v in solve_info.items()])
            + "\n"
        )

        return np.array(time), np.array(solution), solve_info

    def optimal_step(self):
        step: Any = step * np.clip(0.94 * (0.65/err_ratio)**(1/(2*k-1)), self.stepfac_min, self.stepfac_max) # TODO: is this correct also for SEULEX?
        # TODO: check for step rejection
        # TODO: set stepfac_max after step rejection
        # TODO: PI controller, Gustafsson acceleration
        # TODO: try to keep steady?
        if (current_time + step > t_max):
            current_time: float = t_max


    def initial_step(self, U0: NDArray[np.floating], t0: float=0., norm:Callable[[NDArray[np.floating]], float] = norm_hairer):
        """From Hairer & Wanner eq. 4.14"""
        tol = self.atol + self.rtol * np.abs(U0)

        f0 = self.ode_fun(t0, U0)
        d0 = norm(U0)
        d1 = norm(f0)

        h0 = 0.01*(d0/d1)
        if d0 < 1e-5 or d1 < 1e-5:
            h0 = 1e-6

        U1_Eul = U0 + h0 * self.ode_fun(t0, U0)
        d2 = norm(self.ode_fun(t0 + h0, U1_Eul)- f0)/h0

        h1 = (0.01/max(d1, d2))**(1/(self.base_order?+1))
        if max(d1, d2) <= 1e-15:
            h1 = max(1e-6, h0*1e-3)

        return min(100*h0, h1)

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
    ):
        step_seq = np.array(range(1, table_size + 1))  # Harmonic

        super().__init__(
            step_seq=step_seq,
            is_symmetric=False,
            base_order=1,
            ode_fun=ode_fun,
            num_odes=num_odes,
            jac_fun=jac_fun,
            atol=atol,
            rtol=rtol,
            table_size=table_size,
        )

    def base_scheme(
        self, U0: NDArray[np.floating], t0: float, t_max: float, n_steps: int, jac0
    ) -> NDArray[np.floating]:
        r"""calculates the specified number of steps with the linearly-implicit euler scheme (Rosenbrock-like) (I - \Delta t J) U^{n+1} = \Delta t f(U^n) with a constant jacobian evaluated at U0"""
        delta_t = (t_max - t0) / n_steps

        U_n = U0
        t_n = t0
        for _ in range(n_steps):
            delta_U = self.inv_mass_matrix*(delta_t * self.ode_fun(t_n, U_n))
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
    ):
        step_seq = np.array(range(1, table_size + 1))  # Harmonic

        super().__init__(
            step_seq=step_seq,
            is_symmetric=True,
            base_order=1,
            ode_fun=ode_fun,
            num_odes=num_odes,
            jac_fun=jac_fun,
            atol=atol,
            rtol=rtol,
            table_size=table_size,
        )

    def base_scheme(
        self, U0: NDArray[np.floating], t0: float, t_max: float, n_steps: int, jac0
    ) -> NDArray[np.floating]:
        r"""calculates the specified number of steps with the linearly-implicit euler scheme (Rosenbrock-like) (I - \Delta t J) U^{n+1} = \Delta t f(U^n) with a constant jacobian evaluated at U0"""
        delta_t = (t_max - t0) / n_steps

        # t_n = t0
        U_2prev = U0
        U_n = U0 + delta_t*self.inv_mass_matrix*(delta_t * self.ode_fun(t0, U0)) # start with an Euler step
        t_n = t0 + delta_t
        for _ in range(1, n_steps):
            delta_U = self.inv_mass_matrix*(2*delta_t * self.ode_fun(t_n, U_n))
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
    ):

        step_seq = np.array(range(2, table_size + 2))
        # step_seq = np.array(range(1, table_size + 1))  # Harmonic
        # step_seq = np.array([i**2 for i in range(table_size + 1)]) # Romberg
        # self.base_scheme = self.seuler0
        super().__init__(
            step_seq=step_seq,
            is_symmetric=False,
            base_order=1,
            ode_fun=ode_fun,
            num_odes=num_odes,
            jac_fun=jac_fun,
            atol=atol,
            rtol=rtol,
            table_size=table_size,
        )

    def base_scheme(
        self, U0: NDArray[np.floating], t0: float, t_max: float, n_steps: int, jac0
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
            # if i == 1 and (delta_U > delta_U_prev): # TODO: delta_U_prev from Newton iteration!
            #     # TODO: restart
            #     # TODO: thius should be chaked in the base class
        return U_n


class SODEX(Extrapolation_Solver):
    def __init__(
        self,
        ode_fun: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
        num_odes: int,
        jac_fun: Callable[[float, NDArray[np.floating]], NDArray[np.floating]] | None = None,
        mass_matrix: NDArray[np.floating] | None = None,
        atol: float = 1e-11,
        rtol: float = 1e-5,
        table_size: int = 8,
    ):

        step_seq = np.array([2, 6, 10, 14, 22, 34, 50])  
        assert (step_seq%2==0).all(), "Number of steps for SODEX must be even"
        super().__init__(
            step_seq=step_seq,
            is_symmetric=True,
            ode_fun=ode_fun,
            num_odes=num_odes,
            jac_fun=jac_fun,
            atol=atol,
            rtol=rtol,
            table_size=table_size,
        )

    def base_scheme(
        self, U0: NDArray, t0: float, t_max: float, n_steps: int, jac0
    ) -> NDArray:
        delta_t = (t_max - t0) / n_steps
        lu, piv = lu_factor(self.mass_matrix - delta_t * jac0)

        # start with a SEULER step
        rhs = delta_t * self.ode_fun(t0, U0)
        delta_U = np.linalg.solve(self.mass_matrix - delta_t * jac0, rhs)
        U_n = U0 + delta_U
        t_n = t0 + delta_t

        # continue with linearly implicit midpoint
        for _ in range(1,n_steps):
            rhs = 2*delta_t * (self.ode_fun(t_n, U_n) - self.mass_matrix*delta_U)
            delta_U = delta_U + lu_solve((lu, piv), rhs, overwrite_b=True, check_finite=False)
            U_n = U_n + delta_U
            t_n += 2*delta_t
        # Gragg's smoothing, requires one additional step before which we save the previous value of U
        U_fprev = U_n - delta_U
        rhs = 2*delta_t * (self.ode_fun(t_n, U_n) - self.mass_matrix*delta_U)
        delta_U = delta_U + lu_solve((lu, piv), rhs, overwrite_b=True, check_finite=False)
        U_n = U_n + delta_U
        Sh_n = 0.5 * (U_nprev + U_n)

        return Sh_n



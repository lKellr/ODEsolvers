import numpy as np
import pytest
from solvers.embedded import *
from solvers.explicit import *
from solvers.implicit import *
from solvers.Extrapolation_Scheme import *
from modules.step_control import StepControllerExtrapDummy


class TestConvergence:
    norm = staticmethod(norm_hairer)

    N_list = 6 * np.array(  # start with 6 steps
        [
            2 ** (k // 2) if k == 1 or k % 2 == 0 else 1.5 * 2 ** (k // 2)
            for k in range(1, 8)
        ]
    )

    @pytest.mark.parametrize(
        "scheme, expected_order, additional_kwargs",
        [
            (Euler, 1, {}),
            (Midpoint, 2, {}),
            (Heun, 2, {}),
            (AB2, 2, {}),
            (AB3, 3, {}),
            (PECE, 3, {}),
            (PECE_tol, 3, {}),
            (PEC, 3, {}),
            (RK4, 4, {}),
            (SSPRK3, 3, {}),
            (SSPRK34, 3, {}),
        ]
        + [(AB_k, k, {"k": k}) for k in range(1, 10)],
    )
    def test_scheme_sinexp(
        self,
        scheme: Callable,
        expected_order: int,
        additional_kwargs: dict,
    ):
        x_dot: Callable[[float, NDArray[np.floating]], NDArray[np.floating]] = (
            lambda t, x: x * (2.0 - np.sin(t))
        )

        t_max = 1.0
        x0 = np.array([2.0])

        x_analytic: Callable[[float], NDArray[np.floating]] = lambda t: 2 * np.exp(
            2 * t + np.cos(t) - 1.0
        )
        errors = np.empty((len(self.N_list),))
        for i, n_steps in enumerate(self.N_list):
            time, result, solve_info = scheme(
                x_dot, x0, t_max, h=t_max / n_steps, **additional_kwargs
            )
            errors[i] = self.norm(result[-1] - x_analytic(time[-1]))

        conv_orders = np.log(errors[1:] / errors[:-1]) / np.log(
            self.N_list[:-1] / self.N_list[1:]
        )
        order_tol = 0.25 * np.log2(
            expected_order + 1
        )  # allow a bit more tolerance for higher orders
        assert (
            np.abs(conv_orders - expected_order) < order_tol
        ).all(), f"Unexpected convergence rate with p = {conv_orders[np.argmax(np.abs(conv_orders - expected_order))]:.2f}, should be {expected_order}."

        # @pytest.mark.parametrize("order", k_list)
        # def test_AB(self, order: int):
        #     errors = np.empty((len(N_list),))
        #     for i, n_steps in enumerate(N_list):
        #         time, result, solve_info = AB_k(
        #             x_dot, x0, t_max, h=t_max / n_steps, k=order
        #         )
        #         errors[i] = norm(result[-1] - x_analytic(time[-1]))

        #     conv_orders = np.log(errors[1:] / errors[:-1]) / np.log(
        #         N_list[:-1] / N_list[1:]
        #     )
        # order_tol = 0.25 * np.log2(
        #     expected_order + 1
        # )  # allow a bit more tolerance for higher orders
        # assert (
        #     np.abs(conv_orders - expected_order) < order_tol
        # ).all(), f"Unexpected convergence rate with p = {conv_orders[np.argmax(np.abs(conv_orders - expected_order))]:.2f}, should be {expected_order}."

    @pytest.mark.parametrize(
        "scheme, expected_order, additional_kwargs",
        [
            (Backwards_Euler, 1, {}),
            (BDF2, 2, {}),
            (TRBDF2, 2, {}),
            (BDF3, 3, {}),
        ]
        + [(AM_k, k, {"k": k}) for k in range(1, 10)],
    )
    def test_scheme_Dalquist(
        self,
        scheme: Callable,
        expected_order: int,
        additional_kwargs: dict,
    ):
        x_dot: Callable[[float, NDArray[np.floating]], NDArray[np.floating]] = (
            lambda t, x: -1000.0 * x
        )
        jac: Callable[[float, NDArray[np.floating]], NDArray[np.floating]] = (
            lambda t, x: -1000.0
        )
        t_max = 1.0
        x0 = np.array([2.0])

        x_analytic: Callable[[float], NDArray[np.floating]] = lambda t: np.exp(
            -1000.0 * t
        )
        errors = np.empty((len(self.N_list),))
        for i, n_steps in enumerate(self.N_list):
            time, result, solve_info = scheme(
                x_dot, x0, t_max, h=t_max / n_steps, jac_fun=jac, **additional_kwargs
            )
            errors[i] = self.norm(result[-1] - x_analytic(time[-1]))

        conv_orders = np.log(errors[1:] / errors[:-1]) / np.log(
            self.N_list[:-1] / self.N_list[1:]
        )

        order_tol = 0.25 * np.log2(
            expected_order + 1
        )  # allow a bit more tolerance for higher orders
        assert (
            np.abs(conv_orders - expected_order) < order_tol
        ).all(), f"Unexpected convergence rate with p = {conv_orders[np.argmax(np.abs(conv_orders - expected_order))]:.2f}, should be {expected_order}."

    @pytest.mark.parametrize(
        "solver, k, expected_order, additional_kwargs",
        [(EulerExtrapolation, k, k + 1, {}) for k in range(1, 10)]
        + [
            (EulerExtrapolation, k, k + 1, {"substep_seq": "Romberg"})
            for k in [1, 3, 5]
        ]
        + [(EulerExtrapolationMass, k, k + 1, {}) for k in [1, 3, 5]]
        + [(ModMidpointExtrapolation, k, k + 1, {}) for k in [1, 3, 5]]
        + [
            (ModMidpointExtrapolation, k, k + 1, {"use_smoothing": True})
            for k in [1, 3, 5]
        ]
        + [(ModMidpointExtrapolationMass, k, k + 1, {}) for k in [1, 3, 5]]
        + [
            (ModMidpointExtrapolationMass, k, k + 1, {"use_smoothing": True})
            for k in [1, 3, 5]
        ]
        + [(ModMidpointExtrapolationRational, k, k + 1, {}) for k in [1, 3, 5]]
        + [(LimplicitEulerExtrapolation, k, k + 1, {}) for k in [1, 3, 5]]
        + [(LimplicitMidpointExtrapolation, k, k + 1, {}) for k in [1, 3, 5]]
        + [
            (LimplicitMidpointExtrapolation, k, k + 1, {"use_smoothing": True})
            for k in [1, 3, 5]
        ],
    )
    def test_extrap_scheme(
        self,
        solver: Callable,
        k: int,
        expected_order: int,
        additional_kwargs: dict,
    ):
        x_dot: Callable[[float, NDArray[floating]], NDArray[floating]] = (
            lambda t, x: np.array(
                [
                    2 * t * x[0] * np.log(np.maximum(x[1], 1e-3)),
                    -2 * t * x[1] * np.log(np.maximum(x[0], 1e-3)),
                ]
            )
        )

        t_max = 5.0
        x0 = np.array([1.0, np.e])

        x_analytic: Callable[[float], NDArray[floating]] = lambda t: np.array(
            [np.exp(np.sin(t * t)), np.exp(np.cos(t * t))]
        ).T

        errors = np.empty((len(self.N_list),))
        for i, n_steps in enumerate(self.N_list):
            solver_extrap = solver(
                ode_fun=x_dot,
                table_size=12,
                step_controller=StepControllerExtrapDummy(),
            )
            time, result, solve_info = solver_extrap.solve(
                x0, t_max, k_initial=k, h_initial=t_max / n_steps
            )

            errors[i] = self.norm(result[-1] - x_analytic(time[-1]))

        conv_orders = np.log(errors[1:] / errors[:-1]) / np.log(
            self.N_list[:-1] / self.N_list[1:]
        )

        order_tol = 0.25 * np.log2(
            expected_order + 1
        )  # allow a bit more tolerance for higher orders
        assert (
            np.abs(conv_orders - expected_order) < order_tol
        ).all(), f"Unexpected convergence rate with p = {conv_orders[np.argmax(np.abs(conv_orders - expected_order))]:.2f}, should be {expected_order}."

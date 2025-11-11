from typing import Callable, Any
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root


def Backwards_Euler(
    f: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
    x0: NDArray[np.floating],
    t_max: float,
    h: float,
    t0: float = 0.0,
    solvertol: float = 1e-5,
) -> tuple[NDArray[np.floating], NDArray[np.floating], dict[str, Any]]:
    """Backwards Euler Method. System of Equations solved by solver(f==0, a, b, tol)"""
    steps = np.ceil((t_max - t0) / h).astype(int)
    if t0 + steps * h != t_max:
        print("final step not hitting t_max exactly")

    info: dict[str, Any] = dict(
        n_feval=0,
        n_jaceval=0,
        n_lu=0,
        n_restarts=0,
    )

    t = np.linspace(t0, steps * h, steps + 1, dtype=x0.dtype)
    x = np.zeros((steps + 1, x0.shape[0]), dtype=x0.dtype)
    t[0] = t0
    x[0] = x0

    for i in range(steps):
        t[i + 1] = t[i] + h
        f_imp = lambda x_next: x_next - x[i] - h * f(t[i + 1], x_next)
        # x[ i + 1 : i + 2] = solver(
        #     f_imp, x[ i : i + 1], np.eye(x0.shape[1]), tol=solvertol
        # )
        sol = root(f_imp, x0=x[i], tol=solvertol, method="hybr")
        x[i + 1] = sol.x
        if not sol.success:
            print("solver did not converge")
            break
        info["n_feval"] += sol.nfev
        info["n_jaceval"] += sol.njev
        info["n_lu"] += sol.nfev  # TODO: is this correct?
        print(sol.keys())
        exit()  # TODO: wip

    return t, x, info


def BDF2(
    f: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
    x0: NDArray[np.floating],
    t_max: float,
    h: float,
    t0: float = 0.0,
    solvertol: float = 1e-5,
) -> tuple[NDArray[np.floating], NDArray[np.floating], dict[str, Any]]:
    """Backward differantiation Formula of order 2 for stiff systems.
    Starting values generated with backwards Euler method
    System of Equations solved by solver(f==0, a, b, tol)"""
    steps = np.ceil((t_max - t0) / h) + 1

    info: dict[str, Any] = dict(
        n_feval=0,
        n_jaceval=0,
        n_lu=0,
        n_restarts=0,
    )

    t = np.linspace(t0, steps * h, steps + 1, dtype=x0.dtype)
    x = np.zeros((steps + 1, x0.shape[0]), dtype=x0.dtype)
    t[:2], x[:2], inf_starter = Backwards_Euler(f, x0, t0 + h, h, t0, solver, solvertol)
    info = inf_starter

    for i in range(steps - 1):
        t[i + 1] = t[i] + h
        f_imp = (
            lambda x_next: x_next
            - 4 / 3 * x[i]
            + 1 / 3 * x[i - 1]
            - 2 / 3 * h * f(t[i + 1], x_next)
        )
        sol = root(f_imp, x0=x[i], tol=solvertol, method="hybr")
        x[i + 1] = sol.x
        if not sol.success:
            print("solver did not converge")
            break
        info["n_feval"] += sol.nfev
        info["n_jaceval"] += sol.njev
        info["n_lu"] += sol.nfev  # TODO: is this correct?
        print(sol.keys())
        exit()  # TODO: wip
    return t, x, info


def BDF3(
    f: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
    x0: NDArray[np.floating],
    t_max: float,
    h: float,
    t0: float = 0.0,
    solvertol: float = 1e-5,
) -> tuple[NDArray[np.floating], NDArray[np.floating], dict[str, Any]]:
    """Backward differantiation Formula of order 3 for stiff systems.
    Starting values generated with backwards Euler method and BDF2
    System of Equations solved by solver(f==0, a, b, tol)"""
    steps = np.ceil((t_max - t0) / h) + 1

    info: dict[str, Any] = dict(
        n_feval=0,
        n_jaceval=0,
        n_lu=0,
        n_restarts=0,
    )

    t = np.linspace(t0, steps * h, steps + 1, dtype=x0.dtype)
    x = np.zeros((steps + 1, x0.shape[0]), dtype=x0.dtype)
    t[:3], x[:3], inf_starter = BDF2(f, x0, t0 + 2 * h, h, t0, solvertol)
    info = inf_starter

    for i in range(steps - 1):
        t[i + 1] = t[i] + h
        f_imp = (
            lambda x_next: x_next
            - 18 / 11 * x[i]
            + 9 / 11 * x[i - 1]
            - 2 / 11 * x[i]
            - 6 / 11 * h * f(t[i + 1], x_next)
        )
        x[i + 1] = sol.x
        if not sol.success:
            print("solver did not converge")
            break
        info["n_feval"] += sol.nfev
        info["n_jaceval"] += sol.njev
        info["n_lu"] += sol.nfev  # TODO: is this correct?
        print(sol.keys())
        exit()  # TODO: wip
    return t, x, info

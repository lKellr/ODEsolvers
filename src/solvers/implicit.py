from numpy._typing._array_like import NDArray
from numpy import floating
from numpy._typing._array_like import NDArray
from typing import Callable, Any
import numpy as np
from numpy.typing import NDArray
from scipy.special import comb
from modules.helpers import numerical_jacobian, root_wrapped
from modules.root_finding import Newton


def Backwards_Euler(
    f: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
    x0: NDArray[np.floating],
    t_max: float,
    h: float,
    t0: float = 0.0,
    nl_solver: Callable[
        [
            Callable[[NDArray[np.floating]], NDArray[np.floating]],
            NDArray[np.floating],
            Callable[[NDArray[np.floating]], NDArray[np.floating]] | None,
            float | NDArray[np.floating],
        ],
        tuple[NDArray[np.floating], dict[str, Any]],
    ] = Newton,
    jac_fun: Callable[[NDArray[np.floating]], NDArray[np.floating]] | None = None,
    solvertol: float = 1e-5,
) -> tuple[NDArray[np.floating], NDArray[np.floating], dict[str, Any]]:
    """Backwards Euler Method. System of Equations solved by solver(f==0, a, b, tol)"""
    steps = np.ceil((t_max - t0) / h).astype(int)
    if t0 + steps * h != t_max:
        print(f"final step not hitting t_max exactly, instead t_max = {steps * h}")

    info: dict[str, Any] = dict(
        n_feval=0,
        n_jaceval=0,
        n_lu=0,
        n_restarts=0,
    )

    t = np.linspace(t0, steps * h, steps + 1, dtype=x0.dtype)
    x = np.zeros((steps + 1, x0.shape[0]), dtype=x0.dtype)

    x[0] = x0
    for i in range(steps):
        f_imp = lambda x_next: x_next - x[i] - h * f(t[i + 1], x_next)
        x[i + 1], sol_info = nl_solver(f_imp, x0=x[i], tol=solvertol, jac_fun=jac_fun)
        if not sol_info["success"]:
            print("solver did not converge")
            break
        info["n_feval"] += sol_info["n_feval"]
        info["n_jaceval"] += sol_info["n_jaceval"]
        info["n_lu"] += sol_info["n_lu"]

    return t, x, info


def AM_k(
    f: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
    x0: NDArray[np.floating],
    t_max: float,
    h: float,
    k: int,
    t0: float = 0.0,
    nl_solver: Callable[
        [
            Callable[[NDArray[np.floating]], NDArray[np.floating]],
            NDArray[np.floating],
            Callable[[NDArray[np.floating]], NDArray[np.floating]] | None,
            float | NDArray[np.floating],
        ],
        tuple[NDArray[np.floating], dict[str, Any]],
    ] = Newton,
    jac_fun: Callable[[NDArray[np.floating]], NDArray[np.floating]] | None = None,
    solvertol: float = 1e-5
) -> tuple[NDArray[np.floating], NDArray[np.floating], dict[str, Any]]:
    """Adams-Moulton formula of variable order k, maximum implemented is 9"""
    steps = np.ceil((t_max - t0) / h).astype(int)
    if steps * h / (t_max - t0) - 1.0 > 1e-4:
        print(f"final step not hitting t_max exactly, instead t_max = {steps * h}")

    t = np.linspace(t0, steps * h, steps + 1, dtype=x0.dtype)
    x, info, f_values = _AM_k(f=f, x0=x0, steps=steps, h=h, k=k, t0=t0)
    return t, x, info


def _AM_k(
    f: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
    x0: NDArray[np.floating],
    steps: int,
    h: float,
    k: int,
    t0: float = 0.0,
    nl_solver: Callable[
        [
            Callable[[NDArray[np.floating]], NDArray[np.floating]],
            NDArray[np.floating],
            Callable[[NDArray[np.floating]], NDArray[np.floating]] | None,
            float | NDArray[np.floating],
        ],
        tuple[NDArray[np.floating], dict[str, Any]],
    ] = Newton,
    jac_fun: Callable[[NDArray[np.floating]], NDArray[np.floating]] | None = None,
    solvertol: float = 1e-5
) -> tuple[NDArray[np.floating], dict[str, Any], NDArray[np.floating]]:
    """Adams-Moulton of variable order k, this function also returns the computed function values"""
    assert k <= 9, "highest implemented order is 9"

    info: dict[str, Any] = dict(
        n_feval=0,
        n_jaceval=0,
        n_lu=0,
        n_restarts=0,
    )

    # compute the coefficients
    gamma = [
        1.0,
        -1 / 2,
        -1 / 12,
        -1/24,
        -19 / 720,
        -3/160,
        -863 / 60480,
        -275 / 24192,
        -22953 / 3628800,
    ]
    beta: NDArray[Any] = np.array(
        [
            (-1) ** (j - 1)
            * sum([gamma[i] * comb(i, j - 1, exact=True) for i in range(j - 1, k)])
            for j in range(1, k + 1)
        ]
    )
    # TODO: i am not sure about the (-1)**j term, it is not given in my ource, but results are wrong without it

    x = np.zeros((steps + 1, x0.shape[0]), dtype=x0.dtype)
    f_i = np.empty((k, x0.shape[0]), dtype=x0.dtype)
    if k > 2:
        x[:k-1], inf_starter, f_i[: k - 1] = _AM_k(f, x0, k - 2, h, k - 1, t0)
        info = inf_starter
    else: # start with trapezoidal rule
        x[0] = x0
        f_i[0] = f(t0, x0)

    steps_starter = k - 2 if k>1 else 0
    for i in range(steps_starter, steps):
        f_i = np.roll(f_i, 1, axis=0)

        if k > 1: # precompute the constant part
            f_const = x[i] + h * beta[1:] @ f_i[1:]
        else:
            f_const = x[i]

        def f_imp(x_next): 
            f_i[0] = f(t0 + (i+1) * h, x_next)
            return x_next - (f_const + h * beta[0] * f_i[0])

        if jac_fun is None: # Jacobian without setting f_i[0] # TODO: this is probably not efficient
            jac_fun = lambda x_next: np.eye(x_next.shape[0]) - h*beta[0]*numerical_jacobian(x_next, lambda x: f(t0 + (i+1) * h, x), 1e-8)
        
        x[i + 1], sol_info = nl_solver(f_imp, x0=x[i], tol=solvertol, jac_fun=jac_fun)

        if not sol_info["success"]:
            print("solver did not converge")
            break
        info["n_feval"] += sol_info["n_feval"]
        info["n_jaceval"] += sol_info["n_jaceval"]
        info["n_lu"] += sol_info["n_lu"]
    return x, info, f_i


def BDF2(
    f: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
    x0: NDArray[np.floating],
    t_max: float,
    h: float,
    t0: float = 0.0,
    nl_solver: Callable[
        [
            Callable[[NDArray[np.floating]], NDArray[np.floating]],
            NDArray[np.floating],
            Callable[[NDArray[np.floating]], NDArray[np.floating]] | None,
            float | NDArray[np.floating],
        ],
        tuple[NDArray[np.floating], dict[str, Any]],
    ] = Newton,
    jac_fun: Callable[[NDArray[np.floating]], NDArray[np.floating]] | None = None,
    solvertol: float = 1e-5,
) -> tuple[NDArray[np.floating], NDArray[np.floating], dict[str, Any]]:
    """Backward differantiation Formula of order 2 for stiff systems.
    Starting values generated with backwards Euler method
    System of Equations solved by solver(f==0, a, b, tol)"""
    steps = np.ceil((t_max - t0) / h).astype(int)
    if t0 + steps * h != t_max:
        print(f"final step not hitting t_max exactly, instead t_max = {steps * h}")

    info: dict[str, Any] = dict(
        n_feval=0,
        n_jaceval=0,
        n_lu=0,
        n_restarts=0,
    )

    t = np.linspace(t0, steps * h, steps + 1, dtype=x0.dtype)
    x = np.zeros((steps + 1, x0.shape[0]), dtype=x0.dtype)
    t[:2], x[:2], inf_starter = Backwards_Euler(
        f=f, x0=x0, t_max=t0 + h, h=h, t0=t0, nl_solver=nl_solver, solvertol=solvertol
    )
    info = inf_starter

    for i in range(1, steps):
        f_imp = (
            lambda x_next: x_next
            - 4 / 3 * x[i]
            + 1 / 3 * x[i - 1]
            - 2 / 3 * h * f(t[i + 1], x_next)
        )
        x[i + 1], sol_info = nl_solver(f_imp, x0=x[i], tol=solvertol, jac_fun=jac_fun)
        if not sol_info["success"]:
            print("solver did not converge")
            break
        info["n_feval"] += sol_info["n_feval"]
        info["n_jaceval"] += sol_info["n_jaceval"]
        info["n_lu"] += sol_info["n_lu"]
    return t, x, info


def TRBDF2(
    f: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
    x0: NDArray[np.floating],
    t_max: float,
    h: float,
    t0: float = 0.0,
    solvertol: float = 1e-5,
    nl_solver: Callable[
        [
            Callable[[NDArray[np.floating]], NDArray[np.floating]],
            NDArray[np.floating],
            Callable[[NDArray[np.floating]], NDArray[np.floating]] | None,
            float | NDArray[np.floating],
        ],
        tuple[NDArray[np.floating], dict[str, Any]],
    ] = Newton,
    jac_fun: Callable[[NDArray[np.floating]], NDArray[np.floating]] | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating], dict[str, Any]]:
    """Combination of the trapezoidal method with BDF2 to get a DIRK scheme,
    see "Analysis and implementation of TR-BDF2", Hosea and Shampine 1996"""
    steps = np.ceil((t_max - t0) / h).astype(int)
    if t0 + steps * h != t_max:
        print(f"final step not hitting t_max exactly, instead t_max = {steps * h}")

    info: dict[str, Any] = dict(
        n_feval=0,
        n_jaceval=0,
        n_lu=0,
        n_restarts=0,
    )

    t = np.linspace(t0, steps * h, steps + 1, dtype=x0.dtype)
    x = np.zeros((steps + 1, x0.shape[0]), dtype=x0.dtype)

    x[0] = x0
    for i in range(steps):
        f_imp1 = lambda x_halftrapz: x_halftrapz - (
            x[i] + 0.25 * h * (f(t[i], x[i]) + f(t[i] + 0.5 * h, x_halftrapz))
        )
        x_halftrapz, sol1_info = nl_solver(
            f_imp1, x0=x[i], tol=solvertol, jac_fun=jac_fun
        )
        if not sol1_info["success"]:
            print("solver did not converge")
            break
        info["n_feval"] += sol1_info["n_feval"]
        info["n_jaceval"] += sol1_info["n_jaceval"]
        info["n_lu"] += sol1_info["n_lu"]

        f_imp2 = lambda x_next: x_next - 1.0 / 3.0 * (
            4 * x_halftrapz
            - x[i]
            + h
            * f(
                t[i + 1], x_next
            )  # Note that the step is here half of what it is in the normal BDF2 scheme!
        )
        x[i + 1], sol2_info = nl_solver(
            f_imp2, x0=x_halftrapz, tol=solvertol, jac_fun=jac_fun
        )
        if not sol2_info["success"]:
            print("solver did not converge")
            break
        info["n_feval"] += sol2_info["n_feval"]
        info["n_jaceval"] += sol2_info["n_jaceval"]
        info["n_lu"] += sol2_info["n_lu"]
    return t, x, info


def BDF3(
    f: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
    x0: NDArray[np.floating],
    t_max: float,
    h: float,
    t0: float = 0.0,
    nl_solver: Callable[
        [
            Callable[[NDArray[np.floating]], NDArray[np.floating]],
            NDArray[np.floating],
            Callable[[NDArray[np.floating]], NDArray[np.floating]] | None,
            float | NDArray[np.floating],
        ],
        tuple[NDArray[np.floating], dict[str, Any]],
    ] = Newton,
    jac_fun: Callable[[NDArray[np.floating]], NDArray[np.floating]] | None = None,
    solvertol: float = 1e-5,
) -> tuple[NDArray[np.floating], NDArray[np.floating], dict[str, Any]]:
    """Backward differantiation Formula of order 3 for stiff systems.
    Starting values generated with backwards Euler method and BDF2
    System of Equations solved by solver(f==0, a, b, tol)"""
    steps = np.ceil((t_max - t0) / h).astype(int)
    if t0 + steps * h != t_max:
        print(f"final step not hitting t_max exactly, instead t_max = {steps * h}")

    info: dict[str, Any] = dict(
        n_feval=0,
        n_jaceval=0,
        n_lu=0,
        n_restarts=0,
    )

    t = np.linspace(t0, steps * h, steps + 1, dtype=x0.dtype)
    x = np.zeros((steps + 1, x0.shape[0]), dtype=x0.dtype)
    t[:3], x[:3], inf_starter = BDF2(
        f=f,
        x0=x0,
        t_max=t0 + 2 * h,
        h=h,
        t0=t0,
        nl_solver=nl_solver,
        solvertol=solvertol,
        jac_fun=jac_fun,
    )
    info = inf_starter

    for i in range(2, steps):
        f_imp = (
            lambda x_next: 11 * x_next
            - 18 * x[i]
            + 9 * x[i - 1]
            - 2 * x[i - 2]
            - 6 * h * f(t[i + 1], x_next)
        )
        x[i + 1], sol_info = nl_solver(f_imp, x0=x[i], tol=solvertol, jac_fun=jac_fun)

        if not sol_info["success"]:
            print("solver did not converge")
            break
        info["n_feval"] += sol_info["n_feval"]
        info["n_jaceval"] += sol_info["n_jaceval"]
        info["n_lu"] += sol_info["n_lu"]
    return t, x, info


def RADAU5(
    f: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
    x0: NDArray[np.floating],
    t_max: float,
    h: float,
    jac_fun: (
        Callable[[float, NDArray[np.floating]], NDArray[np.floating]] | None
    ) = None,
    t0: float = 0.0,
    solvertol: float | NDArray[np.floating] = 1e-5,
) -> tuple[NDArray[np.floating], NDArray[np.floating], dict[str, Any]]:
    """Radau IIa method with three stages of order 5, following Hairer & Wanner ch. IV.8"""
    steps = np.ceil((t_max - t0) / h).astype(int)
    if t0 + steps * h != t_max:
        print(f"final step not hitting t_max exactly, instead t_max = {steps * h}")

    c = np.array([(4 + sqrt(6)) / 10, (4 + sqrt(6)) / 10, 1])

    info: dict[str, Any] = dict(
        n_feval=0,
        n_jaceval=0,
        n_lu=0,
        n_restarts=0,
    )

    t = np.linspace(t0, steps * h, steps + 1, dtype=x0.dtype)
    x = np.zeros((steps + 1, x0.shape[0]), dtype=x0.dtype)
    x[0] = x0
    for i in range(steps):
        jac = jac_fun(t[i], x[i])
        lu, piv = lu_factor(mass_matrix - h * A * jac)  # TODO: make A explicit
        # Newton iterations
        while True:
            rhs = -Z + h * (A * I) * f(t[i] + c * h, x[i] + Z)
            delta_Z = lu_solve((lu, piv), rhs, overwrite_b=True, check_finite=False)
            Z += delta_Ts

        # TODO:
        raise NotImplementedError()

        # update
        x[i + 1] = x[i] + z_3  # eq 8.2b

        info["n_feval"] += 3  # TODO: cehck
        # info["n_jaceval"] += sol.njev # TODO:
        # info["n_lu"] += sol.nfev
    return t, x, info

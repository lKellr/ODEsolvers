from numpy._typing._array_like import NDArray
from numpy import copy, floating
from typing import Any, Callable
import numpy as np
from numpy.typing import NDArray
from scipy.special import comb


def Euler(
    ode_fun: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
    x0: NDArray[np.floating],
    t_max: float,
    h: float,
    t0: float = 0.0,
) -> tuple[NDArray[np.floating], NDArray[np.floating], dict[str, Any]]:
    """Eulers method for numerical solving of ODEs"""
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
        x[i + 1] = x[i] + h * ode_fun(t[i], x[i])
    info["n_feval"] = steps
    return t, x, info


def Midpoint(
    ode_fun: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
    x0: NDArray[np.floating],
    t_max: float,
    h: float,
    t0: float = 0.0,
) -> tuple[NDArray[np.floating], NDArray[np.floating], dict[str, Any]]:
    """Explicit Midpoint method for numerical solving of ODEs of the form x_dot=ode_fun(t,x)"""
    steps = np.ceil((t_max - t0) / h).astype(int)
    if steps * h / (t_max - t0) - 1.0 > 1e-4:
        print(f"final step not hitting t_max exactly, instead t_max = {steps * h}")

    info: dict[str, Any] = dict(
        n_feval=0,
        n_jaceval=0,
        n_lu=0,
        n_restarts=0,
    )

    t = np.linspace(t0, t_max, steps + 1)
    x = np.zeros((steps + 1, x0.shape[0]), dtype=x0.dtype)

    x[0] = x0
    for i in range(steps):
        x[i + 1] = x[i] + h * ode_fun(
            t[i] + 0.5 * h,
            x[i] + 0.5 * h * ode_fun(t[i], x[i]),
        )

    info["n_feval"] = 2 * steps
    return t, x, info


def Heun(
    ode_fun: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
    x0: NDArray[np.floating],
    t_max: float,
    h: float,
    t0: float = 0.0,
) -> tuple[NDArray[np.floating], NDArray[np.floating], dict[str, Any]]:
    """Heuns method for numerical solving of ODEs of the form x_dot=ode_fun(t,x). THius is equal to SSPRK2"""
    steps = np.ceil((t_max - t0) / h).astype(int)
    if steps * h / (t_max - t0) - 1.0 > 1e-4:
        print(f"final step not hitting t_max exactly, instead t_max = {steps * h}")

    info: dict[str, Any] = dict(
        n_feval=0,
        n_jaceval=0,
        n_lu=0,
        n_restarts=0,
    )

    t = np.linspace(t0, t_max, steps + 1)
    x = np.zeros((steps + 1, x0.shape[0]), dtype=x0.dtype)

    x[0] = x0
    for i in range(steps):
        x1 = x[i] + h * ode_fun(t[i], x[i])
        x[i + 1] = 0.5 * x[i] + 0.5 * (x1 + h * ode_fun(t[i] + h, x1))

    info["n_feval"] = 2 * steps
    return t, x, info


def AB2(
    ode_fun: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
    x0: NDArray[np.floating],
    t_max: float,
    h: float,
    t0: float = 0.0,
) -> tuple[NDArray[np.floating], NDArray[np.floating], dict[str, Any]]:
    """Adams Bashforth of order 2, started by Midpoint method"""
    steps = np.ceil((t_max - t0) / h).astype(int)
    if steps * h / (t_max - t0) - 1.0 > 1e-4:
        print(f"final step not hitting t_max exactly, instead t_max = {steps * h}")

    info: dict[str, Any] = dict(
        n_feval=0,
        n_jaceval=0,
        n_lu=0,
        n_restarts=0,
    )

    t = np.linspace(t0, steps * h, steps + 1, dtype=x0.dtype)
    x = np.zeros((steps + 1, x0.shape[0]), dtype=x0.dtype)
    _, x[:2], inf_starter = Midpoint(ode_fun, x0, t0 + h, h, t0)
    info = inf_starter

    f_ii = ode_fun(t[0], x[0])  # TODO: this has already been evaluated in the starting method
    for i in range(1, steps):
        f_i = ode_fun(t[i], x[i])
        x[i + 1] = x[i] + h / 2 * (3 * f_i - f_ii)
        f_ii = f_i

    info["n_feval"] += steps - 1
    return t, x, info


def AB3(
    ode_fun: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
    x0: NDArray[np.floating],
    t_max: float,
    h: float,
    t0: float = 0.0,
) -> tuple[NDArray[np.floating], NDArray[np.floating], dict[str, Any]]:
    """Adams Bashforth of order 3, first values calculated with Midpoint and AB2"""
    steps = np.ceil((t_max - t0) / h).astype(int)
    if steps * h / (t_max - t0) - 1.0 > 1e-4:
        print(f"final step not hitting t_max exactly, instead t_max = {steps * h}")

    info: dict[str, Any] = dict(
        n_feval=0,
        n_jaceval=0,
        n_lu=0,
        n_restarts=0,
    )

    t = np.linspace(t0, steps * h, steps + 1, dtype=x0.dtype)
    x = np.zeros((steps + 1, x0.shape[0]), dtype=x0.dtype)
    _, x[:3], inf_starter = AB2(ode_fun, x0, t0 + 2 * h, h, t0)
    info = inf_starter

    f_ii = ode_fun(t[1], x[1])  # TODO: this has already been evaluated in the starting method
    f_iii = ode_fun(t[0], x[0])

    for i in range(2, steps):
        f_i = ode_fun(t[i], x[i])
        print(
            f"xi = {x[i]}; f_i= {f_i}, {f_ii}, {f_iii}; xi+1 = {x[i] + h / 12.0 * (23.0 * f_i - 16.0 * f_ii + 5.0 * f_iii)}"
        )
        x[i + 1] = x[i] + h / 12.0 * (23.0 * f_i - 16.0 * f_ii + 5.0 * f_iii)
        f_iii = f_ii
        f_ii = f_i

    info["n_feval"] += steps - 2
    return t, x, info


def PECE(
    ode_fun: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
    x0: NDArray[np.floating],
    t_max: float,
    h: float,
    n_rep: int = 1,
    t0: float = 0.0,
) -> tuple[NDArray[np.floating], NDArray[np.floating], dict[str, Any]]:
    """PECE Method using AB3, AM4, starting with Midpoint and AB2"""
    steps = np.ceil((t_max - t0) / h).astype(int)
    if steps * h / (t_max - t0) - 1.0 > 1e-4:
        print(f"final step not hitting t_max exactly, instead t_max = {steps * h}")

    info: dict[str, Any] = dict(
        n_feval=0,
        n_jaceval=0,
        n_lu=0,
        n_restarts=0,
    )

    t = np.linspace(t0, steps * h, steps + 1, dtype=x0.dtype)
    x = np.zeros((steps + 1, x0.shape[0]), dtype=x0.dtype)
    _, x[:3], inf_starter = AB2(ode_fun, x0, t0 + 2 * h, h, t0)
    info = inf_starter

    f_i = ode_fun(t[1], x[1])  # TODO: this has already been evaluated in the starting method
    f_ii = ode_fun(t[0], x[0])
    for i in range(2, steps):
        f_iii = f_ii
        f_ii = f_i
        f_i = ode_fun(t[i], x[i])

        x[i + 1] = x[i] + h / 12 * (
            23 * f_i - 16 * f_ii + 5 * f_iii
        )  # AB3 predict/evaluate
        k = x[i] + h / 24 * (19 * f_i - 5 * f_ii + f_iii)
        for _ in range(n_rep):
            x[i + 1] = k + 9 * h / 24 * ode_fun(
                t[i + 1], x[i + 1]
            )  # AM4 correct/evaluate loop
    info["n_feval"] += (1 + n_rep) * (steps - 2)
    return t, x, info


def PECE_tol(
    ode_fun: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
    x0: NDArray[np.floating],
    t_max: float,
    h: float,
    tol: float = 1e-4,
    t0: float = 0.0,
) -> tuple[NDArray[np.floating], NDArray[np.floating], dict[str, Any]]:
    """PECE Method using AB3, AM4, starting with Midpoint and AB2, iterates until convergence with tolerance tol is met"""
    steps = np.ceil((t_max - t0) / h).astype(int)
    if steps * h / (t_max - t0) - 1.0 > 1e-4:
        print(f"final step not hitting t_max exactly, instead t_max = {steps * h}")

    info: dict[str, Any] = dict(
        n_feval=0,
        n_jaceval=0,
        n_lu=0,
        n_restarts=0,
    )

    t = np.linspace(t0, steps * h, steps + 1, dtype=x0.dtype)
    x = np.zeros((steps + 1, x0.shape[0]), dtype=x0.dtype)
    _, x[:3], inf_starter = AB2(ode_fun, x0, t0 + 2 * h, h, t0)
    info = inf_starter

    f_i = ode_fun(t[1], x[1])  # TODO: this has already been evaluated in the starting method
    f_ii = ode_fun(t[0], x[0])
    for i in range(2, steps):
        f_iii = f_ii
        f_ii = f_i
        f_i = ode_fun(t[i], x[i])

        x[i + 1] = x[i] + h / 12 * (
            23 * f_i - 16 * f_ii + 5 * f_iii
        )  # AB3 predict/evaluate
        k = x[i] + h / 24 * (19 * f_i - 5 * f_ii + f_iii)

        last = np.inf
        while np.linalg.norm(x[i + 1] - last, ord=np.inf) > tol:
            last = x[i + 1]
            x[i + 1] = k + 9 * h / 24 * ode_fun(
                t[i + 1], x[i + 1]
            )  # AM4 correct/evaluate loop
            info["n_feval"] += 1

    info["n_feval"] += steps - 2
    return t, x, info


def PEC(
    ode_fun: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
    x0: NDArray[np.floating],
    t_max: float,
    h: float,
    n_rep: int = 1,
    t0: float = 0.0,
) -> tuple[NDArray[np.floating], NDArray[np.floating], dict[str, Any]]:
    """PEC Method using AB3, AM4, starting with Midpoint and AB2"""
    steps = np.ceil((t_max - t0) / h).astype(int)
    if steps * h / (t_max - t0) - 1.0 > 1e-4:
        print(f"final step not hitting t_max exactly, instead t_max = {steps * h}")

    info: dict[str, Any] = dict(
        n_feval=0,
        n_jaceval=0,
        n_lu=0,
        n_restarts=0,
    )

    t = np.linspace(t0, steps * h, steps + 1, dtype=x0.dtype)
    x = np.zeros((steps + 1, x0.shape[0]), dtype=x0.dtype)
    _, x[:3], inf_starter = AB2(ode_fun, x0, t0 + 2 * h, h, t0)
    info = inf_starter

    f_i = ode_fun(t[2], x[2])  # TODO: this has already been evaluated in the starting method
    f_ii = ode_fun(t[1], x[1])
    f_iii = ode_fun(t[0], x[0])
    for i in range(2, steps):

        x[i + 1] = x[i] + h / 12 * (
            23 * f_i - 16 * f_ii + 5 * f_iii
        )  # AB3 predict/evaluate
        k = x[i] + h / 24 * (19 * f_i - 5 * f_ii + f_iii)

        f_iii = f_ii
        f_ii = f_i

        for _ in range(n_rep):
            f_i = ode_fun(t[i + 1], x[i + 1])
            x[i + 1] = k + 9 * h / 24 * f_i  # AM4 correct/evaluate loop
    info["n_feval"] += n_rep * (steps - 2)
    return t, x, info


def AB_k(
    ode_fun: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
    x0: NDArray[np.floating],
    t_max: float,
    h: float,
    k: int,
    t0: float = 0.0,
) -> tuple[NDArray[np.floating], NDArray[np.floating], dict[str, Any]]:
    """Adams Bashforth of variable order k, maximum implemented is 9"""
    # This funtion is just a wrapper for tehe real one that also returns the function values

    steps = np.ceil((t_max - t0) / h).astype(int)
    if steps * h / (t_max - t0) - 1.0 > 1e-4:
        print(f"final step not hitting t_max exactly, instead t_max = {steps * h}")

    t = np.linspace(t0, steps * h, steps + 1, dtype=x0.dtype)
    x, info, f_values = _AB_k(ode_fun=ode_fun, x0=x0, steps=steps, h=h, k=k, t0=t0)
    return t, x, info


def _AB_k(
    ode_fun: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
    x0: NDArray[np.floating],
    steps: int,
    h: float,
    k: int,
    t0: float = 0.0,
) -> tuple[NDArray[np.floating], dict[str, Any], NDArray[np.floating]]:
    """Adams Bashforth of variable order k, this function also returns the computed function values"""
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
        1 / 2,
        5 / 12,
        3 / 8,
        251 / 720,
        95 / 288,
        19087 / 60480,
        5257 / 17280,
        1070017 / 3628800,
    ]
    beta = np.array(
        [
            (-1) ** (j - 1)
            * sum([gamma[i] * comb(i, j - 1, exact=True) for i in range(j - 1, k)])
            for j in range(1, k + 1)
        ]
    )
    # TODO: i am not sure about the (-1)**j term, it is not given in my ource, but results are wrong without it

    x = np.zeros((steps + 1, x0.shape[0]), dtype=x0.dtype)
    f_i = np.empty((k, x0.shape[0]), dtype=x0.dtype)
    if k > 1:
        x[:k], inf_starter, f_i[: k - 1] = _AB_k(ode_fun, x0, k - 1, h, k - 1, t0)
        info = inf_starter
    else:
        x[0] = x0

    for i in range(k - 1, steps):
        f_i = np.roll(f_i, 1, axis=0)
        f_i[0] = ode_fun(t0 + i * h, x[i])
        x[i + 1] = x[i] + h * beta @ f_i

    info["n_feval"] += steps - k + 1
    return x, info, f_i


def RK4(
    ode_fun: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
    x0: NDArray[np.floating],
    t_max: float,
    h: float,
    t0: float = 0.0,
) -> tuple[NDArray[np.floating], NDArray[np.floating], dict[str, Any]]:
    """Classical Runge-Kutta Method, order 4"""
    steps = np.ceil((t_max - t0) / h).astype(int)
    if steps * h / (t_max - t0) - 1.0 > 1e-4:
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
        k1 = ode_fun(t[i], x[i])
        k2 = ode_fun(t[i] + 0.5 * h, x[i] + 0.5 * h * k1)
        k3 = ode_fun(t[i] + 0.5 * h, x[i] + 0.5 * h * k2)
        k4 = ode_fun(t[i] + h, x[i] + h * k3)
        x[i + 1] = x[i] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    info["n_feval"] = 4 * steps
    return t, x, info


def SSPRK3(
    ode_fun: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
    x0: NDArray[np.floating],
    t_max: float,
    h: float,
    t0: float = 0.0,
) -> tuple[NDArray[np.floating], NDArray[np.floating], dict[str, Any]]:
    """Strong stability preserving RK method of order 3, cfl_max <=1"""
    steps = np.ceil((t_max - t0) / h).astype(int)
    if steps * h / (t_max - t0) - 1.0 > 1e-4:
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
        x1 = x[i] + h * ode_fun(t[i], x[i])
        x2 = 3.0 / 4.0 * x[i] + 1.0 / 4.0 * (x1 + h * ode_fun(t[i] + h, x1))
        x[i + 1] = 1.0 / 3.0 * x[i] + 2.0 / 3.0 * (x2 + h * ode_fun(t[i] + h / 2.0, x2))

    info["n_feval"] = 3 * steps
    return t, x, info


def SSPRK34(
    ode_fun: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
    x0: NDArray[np.floating],
    t_max: float,
    h: float,
    t0: float = 0.0,
) -> tuple[NDArray[np.floating], NDArray[np.floating], dict[str, Any]]:
    """Four stage strong stability preserving RK method of order 3, cfl_max <=2"""
    steps = np.ceil((t_max - t0) / h).astype(int)
    if steps * h / (t_max - t0) - 1.0 > 1e-4:
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
        x1 = 1.0 / 2.0 * x[i] + 1.0 / 2.0 * (x[i] + h * ode_fun(t[i], x[i]))
        x2 = 1.0 / 2.0 * x1 + 1.0 / 2.0 * (x1 + h * ode_fun(t[i] + h / 2.0, x1))
        x3 = 2.0 / 3.0 * x[i] + 1.0 / 6.0 * x2 + 1.0 / 6.0 * (x2 + h * ode_fun(t[i] + h, x2))
        x[i + 1] = 1.0 / 2.0 * x3 + 1.0 / 2.0 * (x3 + h * ode_fun(t[i] + h / 2.0, x3))

    info["n_feval"] = 4 * steps
    return t, x, info


# TODO: add Radau, TSRK, RK853

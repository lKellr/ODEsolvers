from typing import Callable
import numpy as np
from numpy.typing import NDArray, ANy


def Euler(
    f: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
    x0: NDArray[np.floating],
    t_max: float,
    h: float,
    t0: float = 0.0,
) -> tuple[NDArray[np.floating], NDArray[np.floating], dict[str, Any]]:
    """Eulers method for numerical solving of ODEs"""
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
    x = np.zeros((x0.shape[0], steps + 1), dtype=x0.dtype)
    t[0] = t0
    x[:, 0] = x0
    for i in range(steps):
        # t[:,i+1:i+2]=t[:, i:i+1]+h
        x[:, i + 1 : i + 2] = x[:, i : i + 1] + h * f(t[i], x[:, i : i + 1])
    info["n_feval"] = steps
    return t, x, info


def Midpoint(
    f: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
    x0: NDArray[np.floating],
    t_max: float,
    h: float,
    t0: float = 0.0,
) -> tuple[NDArray[np.floating], NDArray[np.floating], dict[str, Any]]:
    """Explicit Midpoint method for numerical solving of ODEs of the form x_dot=f(t,x)"""
    steps = np.ceil((t_max - t0) / h) + 1
            info: dict[str, Any] = dict(
            n_feval=0,
            n_jaceval=0,
            n_lu=0,
            n_restarts=0,
        )

    info: dict[str, Any] = dict(
            n_feval=0,
            n_jaceval=0,
            n_lu=0,
            n_restarts=0,
        )


    t = np.linspace(t0, t_max, steps)
    x = np.zeros((x0.shape[0], steps))
    t[:, 0] = t0
    x[:, :1] = x0
    for i in range(steps - 1):
        # t[:,i+1:i+2]=t[:, i:i+1]+h
        x[:, i + 1] = x[:, i] + h * f(
            t[i] + 0.5 * h,
            x[:, i] + 0.5 * h * f(t[i], x[:, i]),
        )

    info["n_feval"] = 2 * steps
    return t, x, info


def AB2(
    f: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
    x0: NDArray[np.floating],
    t_max: float,
    h: float,
    t0: float = 0.0,
) -> tuple[NDArray[np.floating], NDArray[np.floating], dict[str, Any]]:
    """Adams Bashforth of order 2, started by Midpoint method"""
    steps = np.ceil((t_max - t0) / h) + 1

    info: dict[str, Any] = dict(
            n_feval=0,
            n_jaceval=0,
            n_lu=0,
            n_restarts=0,
        )

    t = np.linspace(t0, steps * h, steps + 1, dtype=x0.dtype)
    x = np.zeros((x0.shape[0], steps + 1), dtype=x0.dtype)
    _, x[:, :2], inf_starter = Midpoint(f, x0, t0 + h, h, t0)
    info = inf_starter

    for i in range(1, steps - 1):
        # t[:,i+1:i+2]=t[:, i:i+1]+h
        x[:, i + 1] = x[:, i] + h / 2 * (
            3 * f(t[i], x[:, i])
            - f(t[i-1], x[:, i-1])
        )

    info['n_feval'] += 2 * (steps-1)
    return t, x, info


def AB3(
    f: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
    x0: NDArray[np.floating],
    t_max: float,
    h: float,
    t0: float = 0.0,
) -> tuple[NDArray[np.floating], NDArray[np.floating], dict[str, Any]]:
    """Adams Bashforth of order 3, first values calculated with Midpoint and AB2"""
    steps = np.ceil((t_max - t0) / h) + 1

    info: dict[str, Any] = dict(
        n_feval=0,
        n_jaceval=0,
        n_lu=0,
        n_restarts=0,
    )

    t = np.linspace(t0, steps * h, steps + 1, dtype=x0.dtype)
    x = np.zeros((x0.shape[0], steps + 1), dtype=x0.dtype)
    _, x[:, :3], inf_starter = AB2(f, x0, t0 + 2 * h, h, t0)
    info = inf_starter

    for i in range(2, steps - 1):
        # t[:,i+1:i+2]=t[:, i:i+1]+h
        x[:, i + 1] = x[:, i] + h / 12 * (
            23 * f(t[i], x[:, i ])
            - 16 * f(t[i - 1], x[:, i - 1])
            + 5 * f(t[i - 2 ], x[:, i - 2])
        )
    info['n_feval'] += 3 * (steps-2)
    return t, x, info


def PECE(
    f: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
    x0: NDArray[np.floating],
    t_max: float,
    h: float,
    n_rep: int = 1,
    t0: float = 0.0,
) -> tuple[NDArray[np.floating], NDArray[np.floating], dict[str, Any]]:

    """PECE Method using AB3, AM4, starting with Midpoint and AB2"""
    steps = np.ceil((t_max - t0) / h) + 1

    info: dict[str, Any] = dict(
        n_feval=0,
        n_jaceval=0,
        n_lu=0,
        n_restarts=0,
    )

    t = np.linspace(t0, steps * h, steps + 1, dtype=x0.dtype)
    x = np.zeros((x0.shape[0], steps + 1), dtype=x0.dtype)
    # t[:,:3], x[:,:3]=AB2(f, x0, t0+2*h, h, t0)
    _, x[:, :3], inf_starter = AB2(f, x0, t0 + 2 * h, h, t0)
    info = inf_starter

    for i in range(2, steps - 1):
        # t[:,i+1:i+2]=t[:, i:i+1]+h
        f_i  = f(t[i], x[:, i])
        f_ii = f(t[i - 1], x[:, i - 1])
        f_iii = f(t[i - 2], x[:, i - 2 ])

        x[:, i + 1] = x[:, i] + h / 12 * (23 * f_i - 16 * f_ii + 5 * f_iii)  # AB3 predict/evaluate
        k = x[:, i] + h / 24 * (
            19 * f_i
            - 5 * f_ii
            + f_iii
        )
        for _ in range(n_rep):
            x[:, i + 1] = k + 9 * h / 24 * f(t[i + 1], x[:, i + 1])  # AM4 correct/evaluate loop
    info['n_feval'] += (3+n_rep) * (steps-2)
    return t, x, info


#######################   UNTESTED   ######################
def PECE(    f: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
    x0: NDArray[np.floating],
    t_max: float,
    h: float,
    tol: float = 1e-4,
    t0: float = 0.0,
) -> tuple[NDArray[np.floating], NDArray[np.floating], dict[str, Any]]:
    """PECE Method using AB3, AM4, starting with Midpoint and AB2, iterates until convergence with tolerance tol is met"""
    steps = np.ceil((t_max - t0) / h) + 1

    info: dict[str, Any] = dict(
        n_feval=0,
        n_jaceval=0,
        n_lu=0,
        n_restarts=0,
    )

    t = np.linspace(t0, steps * h, steps + 1, dtype=x0.dtype)
    x = np.zeros((x0.shape[0], steps + 1), dtype=x0.dtype)
    _, x[:, :3], inf_starter = AB2(f, x0, t0 + 2 * h, h, t0)
    info = inf_starter

    for i in range(2, steps - 1):
        # t[:,i+1:i+2]=t[:, i:i+1]+h
        f_i  = f(t[i], x[:, i])
        f_ii = f(t[i - 1], x[:, i - 1])
        f_iii = f(t[i - 2], x[:, i - 2 ])

        x[:, i + 1] = x[:, i] + h / 12 * (23 * f_i - 16 * f_ii + 5 * f_iii)  # AB3 predict/evaluate
        k = x[:, i] + h / 24 * (
            19 * f_i
            - 5 * f_ii
            + f_iii
        )

        last = np.inf
        while np.linalg.norm(x[:, i + 1] - last, ord=np.inf) > tol:
            last = x[:, i + 1]
            x[:, i + 1] = k + 9 * h / 24 * f(
                t[i + 1], x[:, i + 1]
            )  # AM4 correct/evaluate loop
            info['n_feval'] += 1

    info['n_feval'] += 3 * (steps-2)
    return t, x, info


def RK4(
    f: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
    x0: NDArray[np.floating],
    t_max: float,
    h: float,
    t0: float = 0.0,
) -> tuple[NDArray[np.floating], NDArray[np.floating], dict[str, Any]]:
    """Classical Runge-Kutta Method, order 4"""
    steps = np.ceil((t_max - t0) / h).astype(int)
    if steps * h / (t_max - t0) - 1.0 > 1e-4:
        print("final step not hitting t_max exactly")

    info: dict[str, Any] = dict(
        n_feval=0,
        n_jaceval=0,
        n_lu=0,
        n_restarts=0,
    )

    t = np.linspace(t0, steps * h, steps + 1, dtype=x0.dtype)
    x = np.zeros((x0.shape[0], steps + 1), dtype=x0.dtype)
    t[0] = t0
    x[:, 0] = x0
    for i in range(steps):
        k1 = f(t[i], x[:, i])
        k2 = f(t[i] + 0.5 * h, x[:, i] + 0.5 * h * k1)
        k3 = f(t[i] + 0.5 * h, x[:, i] + 0.5 * h * k2)
        k4 = f(t[i] + h, x[:, i] + h * k3)
        x[:, i + 1] = x[:, i] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        info["n_feval"] += 4
    return t, x, info




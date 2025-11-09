from typing import Callable
import numpy as np
from numpy.typing import NDArray

def Secant_method(f: Callable[[float], float], a: float, b: float, tol: float) -> float:
    """searches for a root of the function f in the interval [a, b]"""
    x0 = a  # starting value 1
    x1 = b  # starting value 2

    f0 = f(x0)
    f1 = f(x1)
    dx = -f1 / (f1 - f0) * (x1 - x0)
    x1 = x1 + dx

    while abs(f1) > tol:
        f0 = f1
        f1 = f(x1)
        dx = -f1 / (f1 - f0) * dx
        x1 = x1 + dx

    return x1


def Bisection(f: Callable[[float], float], a: float, b: float, tol: float) -> float:
    """searches for a root of the function f in the interval [a, b]"""
    x0 = a
    x1 = b

    f0 = f(x0)
    f1 = f(x1)

    if np.sign(f0) == np.sign(f1):
        print("invalid starting values")
        return np.nan

    xm = 0.5 * (x0 + x1)

    while abs(f1) > tol:
        fm = f(xm)
        if np.sign(fm) == np.sign(f0):
            f0 = fm
        else:
            f1 = fm
        xm = 0.5 * (x0 + x1)

    return xm
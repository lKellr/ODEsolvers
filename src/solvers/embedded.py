from typing import Any, Callable
import numpy as np
from numpy.typing import NDArray

def DP45(
    f: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
    x0: NDArray[np.floating],
    t_max: float,
    h: float,
    t0: float = 0.0,
    eps: float = 10**-5,
    s_clip: tuple[float, float] = (0.2, 5.),
    tol_safety: float = 0.9,
    h_limits: tuple[float, float] = (0, np.inf)
) -> tuple[NDArray[np.floating], NDArray[np.floating], dict[str, Any]]:
    """Dormand Prince 5(4) Method, MATLAB ode45 solver"""
    h = tol_safety * eps ** (1 / 5) / 4

    info: dict[str, Any] = dict(
      n_feval=0,
      n_jaceval=0,
      n_lu=0,
      n_restarts=0,
    )

    t = [t0]
    x = [x0]
    t_crit = []
    x_crit = []

    k = 0
    k1 = f(t[0], x[0]) # FSAL property
    while t[k] < t_max:  # iterate until t_max is reached
        if t[k] + h > t_max:  # shorten h if we would go further than necessary
            h = t_max - t[k]

        t_pred = t[k] + h
        # calulate k's
        # k1=f(t[k], x[k])
        k2 = f(t[k] + 1 / 5 * h, x[k] + 1 / 5 * h * k1)
        k3 = f(t[k] + 3 / 10 * h, x[k] + h * (3 * k1 + 9 * k2) / 40)
        k4 = f(t[k] + 4 / 5 * h, x[k] + h * (44 / 45 * k1 - 56 / 15 * k2 + 32 / 9 * k3))
        k5 = f(
            t[k] + 8 / 9 * h,
            x[k]
            + h
            * (
                19372 / 6561 * k1
                - 25360 / 2187 * k2
                + 64448 / 6561 * k3
                - 212 / 729 * k4
            ),
        )
        k6 = f(
            t_pred,
            x[k]
            + h
            * (
                9017 / 3168 * k1
                - 355 / 33 * k2
                + 46732 / 5247 * k3
                + 49 / 176 * k4
                - 5103 / 18656 * k5
            ),
        )

        x_pred = x[k] + h * (
            35 / 384 * k1
            + 500 / 1113 * k3
            + 125 / 192 * k4
            - 2187 / 6784 * k5
            + 11 / 84 * k6
        )  # embedding, as f(x_pred)==k2
        k2 = f(
            t_pred, x_pred
        )  # reusing k2 instead of new variable k7, as it is not used again

        err = h * np.linalg.norm(
            71 / 57600 * k1
            - 71 / 16695 * k3
            + 71 / 1920 * k4
            - 17253 / 339200 * k5
            + 22 / 525 * k6
            - 1 / 40 * k2,
            ord=np.inf,
        )  # h*np.linalg.norm(71/57600*k1-71/16695*k3+71/1920*k4-17253/339200*k5+22/525*k6-1/40*k2, ord=np.inf) #local error:||x_pred-X_pred7||
        if err == 0:  # set s to max tp prevent divide by zero error
            s = s_clip[1]
        else:
            s = np.clip(
                tol_safety * (eps / err) ** (1 / 5), s_clip[0], s_clip[1]
            )  # s gets clipped to prevent to extreme changes of h, also err might become zero

        if (err < eps or h <= h_limits[0]): #accept result if tolerance is met, or we cant decrease h anymore
            k1 = k2
            k += 1
            x.append(x_pred)
            t.append(t_pred)
        else:
            if s > 1:
                print("h gets increased to decrease error? This shouldnt happen!")
            t_crit.append(t_pred)
            x_crit.append(x_pred)

        h = h * s
        h = np.clip(h, h_limits[0], h_limits[1])
    info['n_feval'] = (k + len(t_crit))*6
    info['n_restarts'] = len(t_crit)
    info['restarts'] = (t_crit, x_crit)
    return t, x, info


def BS23(
    f: Callable[[float, NDArray[np.floating]], NDArray[np.floating]],
    x0: NDArray[np.floating],
    t_max: float,
    h: float,
    t0: float = 0.0,
    eps: float = 10**-5,
    s_clip: tuple[float, float] = (0.2, 5.),
    tol_safety: float = 0.9
    h_limits: tuple[float, float] = (0, np.inf)
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], dict[str, Any]]:
    """Bogacki–Shampine Method, MATLAB ode23 solver"""
    h = tol_safety * eps ** (1 / 3) / 4  # t_max-t0

    info: dict[str, Any] = dict(
      n_feval=0,
      n_jaceval=0,
      n_lu=0,
      n_restarts=0,
    )

    t = [t0]
    x = [x0]
    t_crit = []
    x_crit = []

    k = 0
    k1 = f(t[0], x[0]) # FSAL
    while t[k] < t_max:  # iterate until t_max is reached
        if t[k] + h > t_max:  # shorten h if we would go further than necessary
            h = t_max - t[k]

        t_pred = t[k] + h
        # calulate k's
        k2 = f(t[k] + 1 / 2 * h, x[k] + 1 / 2 * h * k1)
        k3 = f(t[k] + 3 / 4 * h, x[k] + 3 / 4 * h * k2)
        x_pred = x[k] + h * (2 * k1 + 3 * k2 + 4 * k3) / 9
        k4 = f(t_pred, x_pred)

        err = h * np.linalg.norm(
            -5 / 72 * k1 + 1 / 12 * k2 + 1 / 9 * k3 - 1 / 8 * k4, ord=np.inf
        )  # h*np.linalg.norm(71/57600*k1-71/16695*k3+71/1920*k4-17253/339200*k5+22/525*k6-1/40*k2, ord=np.inf) #local error:||x_pred-X_pred7||
        if err == 0:  # set s to max tp prevent divide by zero error
            s = s_clip[1]
        else:
            s = np.clip(
                tol_safety * (eps / err) ** (1 / 3), s_clip[0], s_clip[1]
            )  # s gets clipped to prevent to extreme changes of h, also err might become zero

        if (
            err < eps
        or h <= h_limits[0]): #accept result if tolerance is met, or we cant decrease h anymore
            k1 = k4
            k += 1
            x.append(x_pred)
            t.append(t_pred)
        else:
            if s > 1:
                print("h gets increased to decrease error? This shouldnt happen!")
            t_crit.append(t_pred)
            x_crit.append(x_pred)

        h = h * s
        h=np.clip(h, h_limits[0], h_limits[1])
    info['n_feval'] = (len(t)-1 + len(t_crit))*3
    info['n_restarts'] = len(t_crit)
    info['restarts'] = (t_crit, x_crit)
    return t, x, info
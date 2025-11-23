from numpy._typing._array_like import NDArray


from numpy._typing._array_like import NDArray


from typing import Any, Callable


import os
import numpy as np
from matplotlib import pyplot as plt
# from solvers.embedded import BS23
from solvers.explicit import PECE, AB_k
from solvers.implicit import *
import logging
from numpy.typing import NDArray


logging.basicConfig(level=logging.DEBUG)
# logger_ode = logging.getLogger("solvers.root_finding")
# logger_ode.setLevel(logging.INFO)
logger_mpb = logging.getLogger("matplotlib")
logger_mpb.setLevel(logging.INFO)

cmap = plt.get_cmap("tab10")


## Duffing oscillator
alpha = -1.0
beta = 1.0
gamma = 3.0
delta = 0.02
omega = 1.0


x_dot: Callable[[float, NDArray[np.floating]], NDArray[np.floating]] = (
    lambda t, x: np.array(
        [
            x[1],
            gamma * np.cos(omega * t)
            - (delta * x[1] + alpha * x[0] + beta * x[0] ** 3),
        ]
    )
)

t_max = 8 * np.pi
x0 = np.array([1.0, 0])

# # rescaled Van der Pol oscillator
# epsilon = 1e-6
# x_dot: Callable[[float, NDArray[np.floating]], NDArray[np.floating]] = (
#     lambda t, x: np.array([x[1], ((1 - x[0] ** 2) * x[1] - x[0]) / epsilon])
# )
# t_max = 1
# x0: NDArray[np.floating] = np.array([2.0, 0.0])


ref_path = (
    f"reference_duffing"  # TODO: change if t_max, x0, oscillator parameters are changed
)
if os.path.exists(ref_path + ".npz"):
    dat = np.load(ref_path + ".npz")
    t_high, x_high = dat["t"], dat["x"]
else:
    t_high, x_high, _ = BDF3(x_dot, x0, t_max, 1e-4)
    np.savez_compressed(ref_path, t=t_high, x=x_high)
x_analytic = lambda t: np.array(
    [np.interp(t, t_high, x_high[:, i]) for i in range(2)]
).T

results = dict()
h = 1e-2
# results["BS23"] = BS23(x_dot, x0, t_max, atol=1e-5, rtol=1e-3)
results["BackwardsEuler"] = Backwards_Euler(x_dot, x0, t_max, h)
results["AM3"] = AM_k(x_dot, x0, t_max, h, k=3)
results["PECE"] = PECE(x_dot, x0, t_max, h)
results["AM4"] = AM_k(x_dot, x0, t_max, h, k=4, solvertol=1e-8)
results["AM5"] = AM_k(x_dot, x0, t_max, h, k=5, solvertol=1e-8)
results["BDF2"] = BDF2(x_dot, x0, t_max, h)
results["TR-BDF2"] = TRBDF2(x_dot, x0, t_max, h)
results["BDF3"] = BDF3(x_dot, x0, t_max, h)

fig, axes = plt.subplots(2, 1)
axes[0].set_ylim(-5, 5)
axes[1].set_xlim(0.1)
axes[1].set_yscale("log")
axes[1].set_ylabel("error")

t_ref = np.linspace(0, t_max, 101)
axes[0].plot(t_ref, x_analytic(t_ref)[:, 0], label="analyic", linestyle="--")

for i, (scheme_name, (time, result, solve_info)) in enumerate(results.items()):
    axes[0].plot(time, result[:, 0], label=scheme_name,color=cmap(i))
    axes[1].plot(
        time, np.linalg.norm(result - x_analytic(time), axis=1), label=scheme_name,color=cmap(i)
    )
axes[0].legend(frameon=False)
axes[1].legend(frameon=False)
plt.show()


##  efficiency
fig, axes = plt.subplots(3, 1)

axes[0].set_title("Work-Precision I")
axes[0].set_xlabel("function evaluations")
axes[0].set_ylabel("error")
axes[0].set_yscale("log")
axes[1].set_title("Work-Precision II")
axes[1].set_xlabel("jacobian evaluations")
axes[1].set_ylabel("error")
axes[1].set_yscale("log")
axes[2].set_title("Work-Precision III")
axes[2].set_xlabel("LU decompositions")
axes[2].set_ylabel("error")
axes[2].set_yscale("log")

for i, (scheme_name, (time, result, solve_info)) in enumerate(results.items()):
    axes[0].plot(
        solve_info["n_feval"],
        np.linalg.norm(result - x_analytic(time)),
        label=scheme_name,
        marker="o",
    )
    axes[1].plot(
        solve_info["n_jaceval"],
        np.linalg.norm(result - x_analytic(time)),
        label=scheme_name,
        marker="o",
    )
    axes[2].plot(
        solve_info["n_lu"],
        np.linalg.norm(result - x_analytic(time)),
        label=scheme_name,
        marker="o",
    )
plt.legend(frameon=False)
plt.show()

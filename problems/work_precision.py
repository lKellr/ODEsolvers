from typing import Any, Callable

from time import perf_counter
from scipy.optimize import root
import numpy as np
from matplotlib import pyplot as plt
import os

from solvers.embedded import *
from solvers.explicit import *
from solvers.implicit import *

# from solvers.Extrapolation_Scheme import SEULEX

# ODE problem
x_dot = lambda t, x: np.array(
    [
        2 * t * x[0] * np.log(np.maximum(x[1], 1e-3)),
        -2 * t * x[1] * np.log(np.maximum(x[0], 1e-3)),
    ]
)

t_max = 5.0
x0 = np.array([1.0, np.e])

x_analytic = lambda t: np.array([np.exp(np.sin(t * t)), np.exp(np.cos(t * t))]).T


# create reference solution
def create_solution(x_dot, x0, t_max):
    t, x, info = DP45(x_dot, x0.astype(np.longdouble), t_max, atol=1e-8, rtol=1e-6)

    fig, ax = plt.subplots()
    ax.plot(t, x[:, 0])
    ax.plot(t, x[:, 1])
    plt.show()

    return t, x.astype(x0.dtype)


# ref_path = "reference_solution"
# if os.path.exists(ref_path + ".npz"):
#     dat = np.load(ref_path + ".npz")
#     t_ref, x_ref = dat["t"], dat["x"]
# else:
#     t_ref, x_ref = create_solution(x_dot, x0, t_max)
#     np.savez_compressed(ref_path, t=t_ref, x=x_ref)
# x_analytic = lambda t: np.array(
#     [np.interp(t, t_ref, x_ref[:, i]) for i in range(x0.size)]
# ).T


def benchmark_run(scheme, h):
    timer_start = perf_counter()

    t, x, info = scheme(x_dot, x0, t_max, h=h)

    time = perf_counter() - timer_start
    error = np.mean(np.linalg.norm(x - x_analytic(t), axis=1))

    return time, t.size, info["n_feval"], error

def benchmark_run_controlled(scheme, error):
    timer_start = perf_counter()

    t, x, info = scheme(x_dot, x0, t_max, atol=?, rtol=?)

    time = perf_counter() - timer_start
    error = np.mean(np.linalg.norm(x - x_analytic(t), axis=1))

    return time, t.size, info["n_feval"], error


# benchmark function
def benchmark_scheme(
    scheme,
    expected_ord: float,
    h0: float = 0.1,
    target_errs: list[float] = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
):

    timings = []
    f_evals = []
    errors = []
    steps = []

    data = benchmark_run(scheme, h0)
    timings.append(data[0])
    steps.append(data[1])
    f_evals.append(data[2])
    errors.append(data[3])

    h = h0
    for err in target_errs:
        n_steps = int(
            t_max / h * (errors[-1] / err) ** (1 / expected_ord)
        )  # TODO: would be even better to build a logarithmic polynomial
        h = t_max / n_steps
        data = benchmark_run(scheme, h)
        timings.append(data[0])
        steps.append(data[1])
        f_evals.append(data[2])
        errors.append(data[3])
        print(f"h: {h}, target error: {err}, actual error: {errors[-1]}")

    return timings, steps, f_evals, errors


# solve
results = dict()
results["Euler"] = benchmark_scheme(
    Euler, 1, h0=0.01, target_errs=[1.0, 1e-1, 1e-2, 1e-3, 1e-4]
)
results["BWEuler"] = benchmark_scheme(
    Backwards_Euler, 1, h0=0.01, target_errs=[1.0, 1e-1, 1e-2, 1e-3]
)
results["AB3"] = benchmark_scheme(
    AB3, 3, h0=0.01, target_errs=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
)
results["RK4"] = benchmark_scheme(
    RK4, 4, h0=0.01, target_errs=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
)
results["DP45"] = benchmark_scheme(
    DP45, None, h0=0.1, target_errs=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
)
# results["SEULEX"] = benchmark_scheme(
#     , , h0=0.1, target_errs=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
# )

# plot
## convergence
fig, ax = plt.subplots()

ax.set_title("Convergence")
ax.set_xlabel("steps")
ax.set_ylabel("error")
ax.set_xscale("log")
ax.set_yscale("log")

for solver_name, (timings, steps, f_evals, errors) in results.items():
    ax.plot(steps, errors, label=solver_name)
ax.legend(frameon=False)
plt.show()

##  efficiency
fig, ax = plt.subplots()

ax.set_title("Efficiency")
ax.set_xlabel("function evaluations")
ax.set_ylabel("error")
ax.set_yscale("log")

for solver_name, (timings, steps, f_evals, errors) in results.items():
    ax.plot(f_evals, errors, label=solver_name)
ax.legend(frameon=False)
plt.show()

fig, ax = plt.subplots()

ax.set_title("Efficiency2")
ax.set_xlabel("time")
ax.set_ylabel("error")
ax.set_yscale("log")

for solver_name, (timings, steps, f_evals, errors) in results.items():
    ax.plot(timings, errors, label=solver_name)
ax.legend(frameon=False)
plt.show()

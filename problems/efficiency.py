from numpy._typing._array_like import NDArray
from numpy._typing._array_like import NDArray
from typing import Any, Callable

from time import perf_counter
from scipy.optimize import root
import numpy as np
from matplotlib import pyplot as plt
import os

from solvers.solvers import *
from solvers.Extrapolation_Scheme import SEULEX

# ODE problem
## Pendulum
omega = 2 * np.pi

x_dot = lambda t, x: np.array(
    [
        x[1],
        -(omega**2) * np.sin(x[0]),
    ],
    dtype=x.dtype,
)
t_max = 1.7
x0 = np.array([np.pi / 2, 0.0])


# create reference solution
def create_solution(x_dot, x0, t_max):
    h = t_max / 1e5
    t, x, info = RK4(x_dot, x0.astype(np.longdouble), t_max, h, t0=0)

    fig, ax = plt.subplots()
    ax.plot(t, x[0])
    ax.plot(t, x[1])
    plt.show()
    return x[:, -1]


ref_path = "reference_solution.npy"
if os.path.exists(ref_path):
    x_ref = np.load(ref_path)
else:
    x_ref = create_solution(x_dot, x0, t_max)
    np.save(ref_path, x_ref)


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

    def benchmark_run(scheme, h):
        timer_start = perf_counter()

        t, x, info = scheme(x_dot, x0, t_max, h)
        fig, ax = plt.subplots()
        ax.plot(t, x[0])
        plt.show()

        time = timer_start - perf_counter()
        error = np.linalg.norm(x[:, -1] - x_ref)  # TODO: this does not really work!
        n_steps = t.size

        timings.append(time)
        f_evals.append(info["n_feval"])
        errors.append(error)

    benchmark_run(scheme, h0)

    for err in target_errs:
        n_steps: int = int(t_max / h0 * (errors[0] / err) ** (1 / expected_ord))
        h = t_max / n_steps
        benchmark_run(scheme, h)
        print(f"h: {h}, target: {err}, err: {errors[-1]}")

    # for i in range(len(target_errs) + 1):
    #     if i == 0:
    #         h = h0
    #     else:
    #         h = h0 * (errors[0] / target_errs[i - 1]) ** (1 / expected_ord)
    #     time, n_steps, error = benchmark_run(scheme, h)
    #     times.append(time)
    #     steps.append(n_steps)
    #     errors.append(error)

    return timings, f_evals, errors


# solve
results = dict()
# results["Euler"] = benchmark_scheme(
#     Euler, 1, h0=0.1, target_errs=[1e-1, 1e-2, 1e-3, 1e-4]
# )
results["BWEuler"] = benchmark_scheme(
    Backwards_Euler, 1, h0=0.1, target_errs=[1e-1, 1e-2, 1e-3]
)
results["RK4"] = benchmark_scheme(
    RK4, 4, h0=0.1, target_errs=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
)
# results["DP45"] = benchmark_scheme(
#     , 4, h0=0.1, target_errs=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
# )
# results["SEULEX"] = benchmark_scheme(
#     , , h0=0.1, target_errs=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
# )

# plot
## convergence
fig, ax = plt.subplots()

ax.set_title("Convergence")
ax.set_xlabel("steps")
ax.set_ylabel("error")
ax.set_yscale("log")

for solver_name, (timings, f_evals, errors) in results:
    ax.plot(solver_results[1], errors, legend=solver_name)
ax.legend(frameon=False)

##  efficiency
fig, ax = plt.subplots()

ax.set_title("Efficiency")
ax.set_xlabel("function evaluations")
ax.set_ylabel("error")
ax.set_yscale("log")

for solver_name, (timings, f_evals, errors) in results:
    ax.plot(f_evals, errors, legend=solver_name)
ax.legend(frameon=False)

fig, ax = plt.subplots()

ax.set_title("Efficiency2")
ax.set_xlabel("time")
ax.set_ylabel("error")
ax.set_yscale("log")

for solver_name, (timings, f_evals, errors) in results:
    ax.plot(timings, errors, legend=solver_name)
ax.legend(frameon=False)

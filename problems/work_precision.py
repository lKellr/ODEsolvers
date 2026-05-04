from typing import Any, Callable

from time import perf_counter
from scipy.optimize import root
import numpy as np
from matplotlib import pyplot as plt
import os

from solvers.embedded import *
from solvers.explicit import *
from solvers.implicit import *
from solvers.Extrapolation_Scheme import *

logging.basicConfig(level=logging.DEBUG)
logger_mpb = logging.getLogger("matplotlib")
logger_mpb.setLevel(logging.INFO)
logger_pil = logging.getLogger("PIL")
logger_pil.setLevel(logging.INFO)

cmap = plt.get_cmap("tab20")

norm = norm_hairer

# ODE problem
x_dot: Callable[[float, NDArray[floating]], NDArray[floating]] = lambda t, x: np.array(
    [
        2 * t * x[0] * np.log(np.maximum(x[1], 1e-3)),
        -2 * t * x[1] * np.log(np.maximum(x[0], 1e-3)),
    ]
)

t_max = 5.0
x0:NDArray[floating] = np.array([1.0, np.e])

x_analytic: Callable[[float], NDArray[floating]] = lambda t: np.array([np.exp(np.sin(t * t)), np.exp(np.cos(t * t))]).T


# create reference solution
def create_solution(x_dot: Callable[[float, NDArray[floating]], NDArray[floating]], x0: NDArray[loating], t_max: float)-> tuple(float, NDArray[floating]):# -> tuple[NDArray[floating[Any]], ndarray[_AnyShape, dtype[Any]]]:# -> tuple[NDArray[floating[Any]], ndarray[_AnyShape, dtype[Any]]]:# -> tuple[NDArray[floating[Any]], ndarray[_AnyShape, dtype[Any]]]:# -> tuple[NDArray[floating[Any]], ndarray[_AnyShape, dtype[Any]]]:# -> tuple[NDArray[floating[Any]], ndarray[_AnyShape, dtype[Any]]]:
    t, x, info = DP54(x_dot, x0.astype(np.longdouble), t_max, atol=1e-8, rtol=1e-6)

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

def get_run_kwargs(type, t_max, target_err):

    if type == 'controlledRK':
        run_kwargs = dict(atol=, rtol=1.0)
    if type == 'ExtrapKH':
        step_controller=StepControllerExtrapKH(atol=, rtol=)
        run_kwargs = dict(h_initial=, k_initial=)
    elif type == 'ExtrapK':
        step_controller=StepControllerExtrapK(atol=, rtol=)

        run_kwargs = dict(k_initial=, h_initial=)
    elif type == 'ExtrapH':
        step_controller=StepControllerExtrapH(atol=, rtol=)

        run_kwargs = dict(k_initial=, h_initial=)
    else:
        n_steps = int(
            t_max / h_last * (err_last / target_err) ** (1 / expected_ord)
        ) 

        run_kwargs = dict(h=t_max / n_steps)
        
    return run_kwargs

def benchmark_run(scheme, h):
    t, x, info = scheme(x_dot, x0, t_max, **run_kwargs)
    return time, t.size, info, error

def benchmark_extrap(scheme, h, solver_kwargs):
    solver = EulerExtrapolation(
        ode_fun=x_dot,
        table_size=8,
        step_controller=StepControllerExtrapDummy(),
        **solver_kwargs
    )

    t, x, info = solver.solve(x0, t_max, k_initial=, h_initial=)
    return time, t.size, info, error

# benchmark function
def benchmark_scheme(
    scheme,
    expected_ord: float,
    h0: float = 0.1,
    target_errs: list[float] = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
):

    timings = []
    infos = []
    errors = []
    steps = []

    data = benchmark_run(scheme, h0)
    timings.append(data[0])
    steps.append(data[1])
    infos.append(data[2])
    errors.append(data[3])

    h = h0
    for err in target_errs:
        n_steps = int(
            t_max / h * (errors[-1] / err) ** (1 / expected_ord)
        ) 
        h = t_max / n_steps

        timer_start = perf_counter()
        data = benchmark_run(scheme, h)
        time = perf_counter() - timer_start
        error = np.mean(norm(x - x_analytic(t), axis=1))

        timings.append(data[0])
        steps.append(data[1])
        infos.append(data[2])
        errors.append(data[3])
        print(f"h: {h}, target error: {err}, actual error: {errors[-1]}")

    return timings, steps, infos, errors


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
results["DP54"] = benchmark_scheme(
    DP54, None, h0=0.1, target_errs=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
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

for i, (solver_name, (timings, steps, info, errors)) in enumerate(results.items()):
    ax.plot(steps, errors, label=solver_name, color=cmap(i))
ax.legend(frameon=False)
plt.show()

##  efficiency
fig, ax = plt.subplots()

ax.set_title("Efficiency")
ax.set_xlabel("relative function evaluation cost")
ax.set_ylabel("error")
ax.set_yscale("log")

implicit_rel_costs = ImplicitRelCosts(rel_lu_cost=2.0, rel_jac_cost=3.0)
def relative_cost(n_fevals: int, n_lu: int, n_jac: int):
    return n_fevals + n_lu*implicit_rel_costs.rel_lu_cost + n_jac*implicit_rel_costs.rel_jac_cost

for i, (solver_name, (timings, steps, info, errors)) in enumerate(results.items()):
    ax.plot(relative_cost(info['n_feval'], info['n_lu'], info['n_jac']), errors, label=solver_name, color=cmap(i))
ax.legend(frameon=False)
plt.show()

fig, ax = plt.subplots()

ax.set_title("Efficiency2")
ax.set_xlabel("time")
ax.set_ylabel("error")
ax.set_yscale("log")

for i, (solver_name, (timings, steps, info, errors)) in enumerate(results.items()):
    ax.plot(timings, errors, label=solver_name, color=cmap(i))
ax.legend(frameon=False)
plt.show()

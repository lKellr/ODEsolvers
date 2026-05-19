import numpy as np
from matplotlib import pyplot as plt
from scipy.differentiate import jacobian
from scipy.integrate import solve_ivp
from modules.helpers import norm_hairer_jit
from modules.step_control import ControllerPIParams, StepControllerExtrapKH_HW
from solvers.embedded import DP54
from solvers.implicit import *
from solvers.Extrapolation_Scheme import *
import logging
from numba import jit, float64
from time import perf_counter
import cProfile

logging.basicConfig(level=logging.WARN)
logger_mpb = logging.getLogger("matplotlib")
logger_mpb.setLevel(logging.INFO)
logger_nb = logging.getLogger("numba")
logger_nb.setLevel(logging.INFO)

cmap: plt.Colormap = plt.get_cmap("tab20")


@jit(float64[:](float64, float64[:]))
def x_dot(t, x):
    return np.array(
        [1.0 + x[0] * x[0] * x[1] - 4 * x[0], 3 * x[0] - x[0] * x[0] * x[1]],
        dtype=x.dtype,
    )


@jit(float64[:, :](float64, float64[:]))
def jac(t: float, x: NDArray[np.floating]):
    return np.array(
        [
            [2 * x[0] * x[1] - 4.0, x[0] * x[0]],
            [3.0 - 2 * x[0] * x[1], -x[0] * x[0]],
        ],
        dtype=x.dtype,
    )


t_max = 1e1  # interesting solutions for t_max ~ 1e11
x0 = np.array([1.5, 3.0])


# jac = lambda t, x: numerical_jacobian_t(
#                 t, x, x_dot, delta=1e-12
#             )
assert (
    np.linalg.norm(jac(0.0, x=x0) - numerical_jacobian_t(0.0, x0, x_dot, 1e-9)) < 1e-6
)

# compile numba
print(f"x_dot0 = {x_dot(0.0, x0)}")
print(f"jac0 = {jac(0.0, x0)}")

results = dict()

# EULEX
prof_tim_start = perf_counter()
solver_eulex = EulerExtrapolation(
    x_dot,
    table_size=12,
    step_controller=StepControllerExtrapKH_HW(
        atol=1e-8, rtol=1e-5, norm=norm_hairer_jit
    ),
)
with cProfile.Profile() as pr:
    time, result, solve_info = solver_eulex.solve(x0, t_max)
    pr.create_stats()
    pr.dump_stats("EULEX.prof")

prof_elapsed = perf_counter() - prof_tim_start
print(
    f"solution took {prof_elapsed:.3f} s for EULEX, {time.size} steps, {solve_info['n_feval']} function evals, dt_ave {t_max/time.size}"
)

# 0.6 s with numba njit, 1.6 s without
results["EULEX"] = time, result, solve_info

# ODEX
prof_tim_start = perf_counter()
solver_odex = ModMidpointExtrapolation(
    x_dot,
    table_size=12,
    step_controller=StepControllerExtrapKH_HW(
        atol=1e-8, rtol=1e-5, norm=norm_hairer_jit
    ),
)
with cProfile.Profile() as pr:
    time, result, solve_info = solver_odex.solve(x0, t_max)
    pr.create_stats()
    pr.dump_stats("ODEX.prof")

prof_elapsed = perf_counter() - prof_tim_start
print(
    f"solution took {prof_elapsed:.3f} s for ODEX, {time.size} steps, {solve_info['n_feval']} function evals, dt_ave {t_max/time.size}"
)

results["ODEX"] = time, result, solve_info

# SEULEX
prof_tim_start = perf_counter()
solver_seulex = LimplicitEulerExtrapolation(
    x_dot,
    table_size=12,
    jac_fun=jac,
    num_odes=x0.size,
    step_controller=StepControllerExtrapKH_HW(
        atol=1e-8, rtol=1e-5, norm=norm_hairer_jit
    ),
)
with cProfile.Profile() as pr:
    time, result, solve_info = solver_seulex.solve(x0, t_max)
    pr.create_stats()
    pr.dump_stats("SEULEX.prof")

prof_elapsed = perf_counter() - prof_tim_start
print(
    f"solution took {prof_elapsed:.3f} s for SEULEX, {time.size} steps, {solve_info['n_feval']} function evals, dt_ave {t_max/time.size}, jac evals: {solve_info['n_jaceval']}, LU decompositions: {solve_info['n_lu']}"
)
results["SEULEX"] = time, result, solve_info

# SODEX
prof_tim_start = perf_counter()
solver_sodex = LimplicitMidpointExtrapolation(
    x_dot,
    table_size=12,
    jac_fun=jac,
    num_odes=x0.size,
    step_controller=StepControllerExtrapKH_HW(
        atol=1e-8, rtol=1e-5, norm=norm_hairer_jit
    ),
)
with cProfile.Profile() as pr:
    time, result, solve_info = solver_sodex.solve(x0, t_max)
    pr.create_stats()
    pr.dump_stats("SODEX.prof")

prof_elapsed = perf_counter() - prof_tim_start
print(
    f"solution took {prof_elapsed:.3f} s for SODEX, {time.size} steps, {solve_info['n_feval']} function evals, dt_ave {t_max/time.size}, jac evals: {solve_info['n_jaceval']}, LU decompositions: {solve_info['n_lu']}"
)

results["SODEX"] = time, result, solve_info

# DP54
prof_tim_start = perf_counter()
with cProfile.Profile() as pr:
    time, result, solve_info = DP54(
        x_dot,
        x0,
        t_max,
        h_limits=(1e-16, np.inf),
        atol=1e-8,
        rtol=1e-5,
    )
    pr.create_stats()
    pr.dump_stats("DP54.prof")

prof_elapsed = perf_counter() - prof_tim_start
print(
    f"solution took {prof_elapsed:.3f} s for DP54, {time.size} steps, {solve_info['n_feval']} function evals, dt_ave {t_max/time.size}"
)

results["DP54"] = time, result, solve_info  # 0.9 s with numba njit, 9.4 s without

# scipy DP54
prof_tim_start = perf_counter()
sol = solve_ivp(x_dot, (0.0, t_max), x0, "RK45", atol=1e-8, rtol=1e-5)
prof_elapsed = perf_counter() - prof_tim_start
print(
    f"solution took {prof_elapsed:.3f} s for scipy DP54, {sol.t.size} steps, {sol.nfev} function evals, dt_ave {t_max/sol.t.size}"
)
results["SP_RK45"] = (
    sol.t,
    sol.y.T,
    dict(
        n_feval=sol.nfev,
        n_jaceval=sol.njev,
        n_lu=sol.nlu,
        n_restarts=0,
        local_errors=[],
    ),
)

# scipy DOP853
prof_tim_start: float = perf_counter()
sol = solve_ivp(x_dot, (0.0, t_max), x0, "DOP853", atol=1e-8, rtol=1e-5)
prof_elapsed = perf_counter() - prof_tim_start
print(
    f"solution took {prof_elapsed:.3f} s for scipy DOP853, {sol.t.size} steps, {sol.nfev} function evals, dt_ave {t_max/sol.t.size}"
)
results["SP_DOP853"] = (
    sol.t,
    sol.y.T,
    dict(
        n_feval=sol.nfev,
        n_jaceval=sol.njev,
        n_lu=sol.nlu,
        n_restarts=0,
        local_errors=[],
    ),
)

# scipy BDF
prof_tim_start = perf_counter()
sol = solve_ivp(x_dot, (0.0, t_max), x0, "BDF", atol=1e-8, rtol=1e-5, jac=jac)
prof_elapsed = perf_counter() - prof_tim_start
print(
    f"solution took {prof_elapsed:.3f} s for scipy BDF, {sol.t.size} steps, {sol.nfev} function evals, dt_ave {t_max/sol.t.size}, jac evals: {sol.njev}, LU decompositions: {sol.nlu}"
)
results["SP_BDF"] = (
    sol.t,
    sol.y.T,
    dict(
        n_feval=sol.nfev,
        n_jaceval=sol.njev,
        n_lu=sol.nlu,
        n_restarts=0,
        local_errors=[],
    ),
)

# scipy Radau
prof_tim_start: float = perf_counter()
sol = solve_ivp(x_dot, (0.0, t_max), x0, "Radau", atol=1e-8, rtol=1e-5, jac=jac)
prof_elapsed = perf_counter() - prof_tim_start
print(
    f"solution took {prof_elapsed:.3f} s for scipy Radau, {sol.t.size} steps, {sol.nfev} function evals, dt_ave {t_max/sol.t.size}, jac evals: {sol.njev}, LU decompositions: {sol.nlu}"
)
results["SP_Radau"] = (
    sol.t,
    sol.y.T,
    dict(
        n_feval=sol.nfev,
        n_jaceval=sol.njev,
        n_lu=sol.nlu,
        n_restarts=0,
        local_errors=[],
    ),
)

# results
fig, ax = plt.subplots(nrows=2, sharex=True)


for i, (scheme_name, (time, result, solve_info)) in enumerate(results.items()):
    ax[0].plot(
        time,
        result[:, 0],
        label=scheme_name,
        color=cmap(i),
    )
    ax[1].plot(
        time,
        result[:, 1],
        label=scheme_name,
        color=cmap(i),
    )
plt.legend()
plt.show()

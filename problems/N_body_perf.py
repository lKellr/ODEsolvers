import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from modules.helpers import norm_hairer_jit
from modules.step_control import ControllerPIParams, StepControllerExtrapKH
from solvers.embedded import DP54
from solvers.explicit import *
from solvers.Extrapolation_Scheme import *
import logging
from numba import njit
from time import perf_counter
import cProfile

logging.basicConfig(level=logging.WARN)
logger_mpb = logging.getLogger("matplotlib")
logger_mpb.setLevel(logging.INFO)
logger_nb = logging.getLogger("numba")
logger_nb.setLevel(logging.INFO)

cmap: plt.Colormap = plt.get_cmap("tab20")

# We solve the equations nondimensionalized, with reference length 1 AU, reference time 1 year and reference mass equal to the total sun system's mass
masses = np.array([0.987, 1.67e-7, 2.47e-6, 3.03e-6, 3.24e-7, 9.62e-4, 2.88e-4])
masses2 = np.repeat(masses, 2)
names = ["Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn"]
G = 4 * np.pi**2
n = masses.size
dim = 2


@njit
def x_dot(t, x):
    velocities = x[n * dim :] / masses2

    forces = np.zeros_like(velocities)
    for i in range(n):
        for j in range(i):
            slice_i = slice(i * dim, (i + 1) * dim)
            slice_j = slice(j * dim, (j + 1) * dim)
            dist = x[slice_j] - x[slice_i]
            q = masses[i] * masses[j] * dist * max(dist @ dist.T, 1e-6) ** -1.5
            forces[slice_i] += q
            forces[slice_j] -= q
    forces *= G
    return np.hstack((velocities, forces))


def compute_energies(x):
    velocities = x[..., n * dim :] / masses2

    e_kin = np.sum(
        0.5 * masses * (velocities[..., ::2] ** 2 + velocities[..., 1::2] ** 2), axis=-1
    )

    e_pot = 0.0
    for i in range(n):
        for j in range(i):
            slice_i = slice(i * dim, (i + 1) * dim)
            slice_j = slice(j * dim, (j + 1) * dim)
            dist = x[..., slice_j] - x[..., slice_i]
            e_pot += (
                masses[i] * masses[j] / np.sqrt(dist[..., 0] ** 2 + dist[..., 1] ** 2)
            )
    e_pot *= -G
    return e_kin, e_pot


t_max = 100.0
initial_positions = np.array(
    [
        [0.0, 0.0],
        [0.387, 0.0],
        [0.723, 0.0],
        [1.0, 0.0],
        [1.52, 0.0],
        [5.2, 0.0],
        [9.57, 0.0],
    ]
)
initial_velocities = np.array(  # calculation from circular orbit
    [[0.0, 0.0]]
    + [
        [
            0.0,
            np.sqrt(
                G
                * masses[0]
                / np.sqrt(initial_positions[i, 0] ** 2 + initial_positions[i, 1] ** 2)
            ),
        ]
        for i in range(1, n)
    ]
)

x0 = np.hstack([initial_positions.flatten(), masses2 * initial_velocities.flatten()])

assert x0.size == (2 * n * dim), f"Wrong shape {x0.shape} of initial condition"

# compile numba
print(f"x_dot0 = {x_dot(0.0, x0)}")
e_kin, e_pot = compute_energies(x0)
print(f"initial energies: e_kin = {e_kin}, e_pot = {e_pot}, e_tot = {e_kin+e_pot}")

results = dict()

# EULEX
prof_tim_start = perf_counter()
solver_eulex = EulerExtrapolation(
    x_dot,
    table_size=12,
    step_controller=StepControllerExtrapKH(atol=1e-8, rtol=1e-5, norm=norm_hairer_jit),
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
    step_controller=StepControllerExtrapKH(atol=1e-8, rtol=1e-5, norm=norm_hairer_jit),
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

# ODEX high
prof_tim_start = perf_counter()
solver_odex2 = ModMidpointExtrapolation(
    x_dot,
    table_size=16,
    step_controller=StepControllerExtrapKH(atol=1e-12, rtol=1e-8, norm=norm_hairer_jit),
)
time, result, solve_info = solver_odex2.solve(x0, t_max)

prof_elapsed = perf_counter() - prof_tim_start
print(
    f"solution took {prof_elapsed:.3f} s for ODEX-high, {time.size} steps, {solve_info['n_feval']} function evals, dt_ave {t_max/time.size}"
)
results["ODEX_high"] = time, result, solve_info

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

# results
# energy
fig, ax = plt.subplots(figsize=(9.6, 7.2))
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$E_\mathrm{tot}/E_{\mathrm{tot}, 0}$")
for i, (scheme_name, (time, result, solve_info)) in enumerate(results.items()):
    e_kin, e_pot = compute_energies(result)
    energy_tot = e_kin + e_pot
    ax.plot(
        time,
        energy_tot / energy_tot[0],
        label=scheme_name,
        color=cmap(i),
    )
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("energy_conservation.png")
plt.show()

# orbits
fig, axes = plt.subplots(4, 1)

for i, (scheme_name, (time, result, solve_info)) in enumerate(results.items()):
    axes[0].plot(
        time,
        result[:, 2],
        label=scheme_name,
        color=cmap(i),
    )
    axes[1].plot(
        time,
        result[:, 3],
        label=scheme_name,
        color=cmap(i),
    )
    axes[2].plot(
        time,
        result[:, 4],
        label=scheme_name,
        color=cmap(i),
    )
    # if "restarts" in solve_info.keys():
    #     ax.plot(
    #         solve_info["restarts"][0],
    #         solve_info["restarts"][1],
    #         color=cmap(i),
    #         marker="o",
    #         linestyle="None",
    #     )
    axes[3].plot(
        0.5 * (time[1:] + time[:-1]),
        np.diff(time),
        label=scheme_name,
        color=cmap(i),
    )

plt.legend(frameon=False)
plt.tight_layout()
plt.show()

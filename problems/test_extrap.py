import numpy as np
from matplotlib import pyplot as plt

from modules.post_processing import find_local_errors
from modules.step_control import (
    ControllerPIParams,
    StepControllerExtrapK,
    StepControllerExtrapH,
    StepControllerExtrapKH,
)
from solvers.embedded import *
from solvers.explicit import *
from solvers.Extrapolation_Scheme import *
import logging

logging.basicConfig(level=logging.DEBUG)
logger_mpb = logging.getLogger("matplotlib")
logger_mpb.setLevel(logging.INFO)
logger_pil = logging.getLogger("PIL")
logger_pil.setLevel(logging.INFO)

cmap = plt.get_cmap("tab20")


x_dot: Callable[[float, NDArray[floating]], NDArray[floating]] = lambda t, x: np.array(
    [
        2 * t * x[0] * np.log(np.maximum(x[1], 1e-3)),
        -2 * t * x[1] * np.log(np.maximum(x[0], 1e-3)),
    ]
)

t_max = 5.0
x0 = np.array([1.0, np.e])

x_analytic: Callable[[float], NDArray[floating]] = lambda t: np.array(
    [np.exp(np.sin(t * t)), np.exp(np.cos(t * t))]
).T


results = dict()
results["DP45"] = DP45(
    x_dot,
    x0,
    t_max,
    h_limits=(1e-16, np.inf),
    atol=1e-7,
    rtol=1e-5,
)
h_average = t_max / len(
    results["DP45"][0]
)  # use the same number of steps as the adaptive scheme

# solver_eulex = EulerExtrapolation(
#     x_dot, table_size=8, step_controller=StepControllerExtrapKH(atol=1e-7, rtol=1e-5)
# )
# results["EULEX"] = solver_eulex.solve(x0, t_max)

# solver_eulex_quad = EulerExtrapolation(
#     x_dot,
#     table_size=8,
#     step_controller=StepControllerExtrapKH(atol=1e-7, rtol=1e-5),
#     dtype=np.longdouble,
# )
# results["EULEX_quad"] = solver_eulex_quad.solve(x0, t_max)

# solver_eulex_step = EulerExtrapolation(
#     x_dot, table_size=8, step_controller=StepControllerExtrapK(atol=1e-7, rtol=1e-5)
# )
# results["EULEX_const_step"] = solver_eulex_step.solve(x0, t_max, h_initial=h_average)

solver_eulex_ord = EulerExtrapolation(
    x_dot,
    table_size=8,
    step_controller=StepControllerExtrapH(atol=1e-7, rtol=1e-5),
)
results["EULEX_const_ord"] = solver_eulex_ord.solve(
    x0,
    t_max,
    k_initial=solver_eulex_ord.table_size - 1,
    # h_initial=0.5,
)

# solver_eulex_mass = EulerExtrapolationMass(
#     x_dot, np.identity(2), table_size=4, step_controller=StepControllerExtrapH()
# )
# results["EULEX_mass"] = solver_eulex_mass.solve(x0, t_max)

# solver_odex = ModMidpointExtrapolation(x_dot, table_size=8)
# results["ODEX"] = solver_odex.solve(x0, t_max)

# solver_odex_smoothed = ModMidpointExtrapolation(x_dot, table_size=8, use_smoothing=True)
# results["ODEX_smoothed"] = solver_odex_smoothed.solve(x0, t_max)

# solver_odex_mass = ModMidpointExtrapolationMass(x_dot, np.identity(2), table_size=8)
# results["ODEX_mass"] = solver_odex_mass.solve(x0, t_max)

# solver_seulex = LimplicitEulerExtrapolation(x_dot, table_size=8, num_odes=x0.size)
# results["SEULEX"] = solver_seulex.solve(x0, t_max)

# solver_seulex_quad = LimplicitEulerExtrapolation(
#     x_dot, table_size=20, num_odes=x0.size, dtype=np.longdouble
# )
# results["SEULEX_quad"] = solver_seulex_quad.solve(x0, t_max)

# solver_sodex = LimplicitMidpointExtrapolation(x_dot, num_odes=x0.size)
# results["SODEX"] = solver_sodex.solve(x0, t_max)

# solver_sodex_smoothed = LimplicitMidpointExtrapolation(
#     x_dot, table_size=8, num_odes=x0.size, use_smoothing=True
# )
# results["SODEX_smoothed"] = solver_sodex_smoothed.solve(x0, t_max)

# results
fig, axes = plt.subplots(3, 1, sharex=True, tight_layout=True)
axes[0].set_ylim(-5, 5)
axes[1].set_yscale("log")
axes[1].set_ylabel("error")
axes[2].set_xlabel("time")
axes[2].set_ylabel(r"$\Delta t$")

t_ref = np.linspace(0, t_max, 101)
axes[0].plot(
    t_ref,
    x_analytic(t_ref)[:, 0],
    label="analyic",
    color="dimgray",
    linestyle="--",
    zorder=10,
)

for i, (scheme_name, (time, result, solve_info)) in enumerate(results.items()):
    axes[0].plot(time, result[:, 0], label=scheme_name, color=cmap(i), marker="x")
    if "restarts" in solve_info.keys():
        axes[0].plot(
            solve_info["restarts"][0],
            solve_info["restarts"][1],
            color=cmap(i),
            marker="o",
            linestyle="None",
        )

    axes[1].plot(
        time,
        np.linalg.norm(result - x_analytic(time), axis=1),
        label=scheme_name,
        color=cmap(i),
    )
    axes[2].plot(
        0.5 * (time[1:] + time[:-1]),
        np.diff(time),
        label=scheme_name,
        color=cmap(i),
    )

axes[0].legend(frameon=False)
axes[1].legend(frameon=False)
axes[2].legend(frameon=False)
plt.tight_layout()
plt.show()
# plt.savefig("./tmp_results.pdf", backend="pgf")

#  efficiency
fig_eff, ax_eff = plt.subplots()

ax_eff.set_title("Work-Precision")
ax_eff.set_xlabel("function evaluations")
ax_eff.set_ylabel("error")
ax_eff.set_yscale("log")

for i, (scheme_name, (time, result, solve_info)) in enumerate(results.items()):
    ax_eff.plot(
        solve_info["n_feval"],
        np.mean(np.linalg.norm(result - x_analytic(time), axis=1)),
        label=scheme_name,
        marker="o",
        color=cmap(i),
    )
ax_eff.set_xlim(0.0)
ax_eff.legend(frameon=False)
plt.tight_layout()
plt.show()

# local errors
fig_le, ax_le = plt.subplots()
ax_le.set_yscale("log")
ax_le.set_ylabel("local error")
ax_le.set_xlabel("time")
for i, (scheme_name, (time, result, solve_info)) in enumerate(results.items()):
    local_error = find_local_errors(x_dot, time, result)

    ax_le.plot(
        time,
        local_error,
        label=scheme_name,
        marker="o",
        color=cmap(i),
    )
    if "local_errors" in solve_info.keys():
        ax_le.plot(
            time[1:],
            solve_info["local_errors"],
            label=scheme_name + "_estimated",
            linestyle="--",
            marker="x",
            color=cmap(i),
        )
ax_le.legend(frameon=False)
plt.tight_layout()
plt.show()

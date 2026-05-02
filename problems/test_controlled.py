import numpy as np
from matplotlib import pyplot as plt

from modules.post_processing import find_local_errors
from modules.step_control import ControllerPIParams, StepControllerExtrapKH
from solvers.embedded import *
from solvers.explicit import *
from solvers.Extrapolation_Scheme import EulerExtrapolation, ModMidpointExtrapolation
import logging

logging.basicConfig(level=logging.DEBUG)
logger_mpb = logging.getLogger("matplotlib")
logger_mpb.setLevel(logging.INFO)
logger_pil = logging.getLogger("PIL")
logger_pil.setLevel(logging.INFO)


cmap: plt.Colormap = plt.get_cmap("tab20")


x_dot = lambda t, x: np.array(
    [
        2 * t * x[0] * np.log(np.maximum(x[1], 1e-3)),
        -2 * t * x[1] * np.log(np.maximum(x[0], 1e-3)),
    ]
)

t_max = 5.0
x0 = np.array([1.0, np.e])

x_analytic = lambda t: np.array([np.exp(np.sin(t * t)), np.exp(np.cos(t * t))]).T

results = dict()

results["BS23"] = BS23(x_dot, x0, t_max, atol=1e-5, rtol=1e-3)
results["DP45"] = DP45(
    x_dot,
    x0,
    t_max,
    h_limits=(1e-16, np.inf),
    atol=1e-5,
    rtol=1e-3,
)
results["DP45_I"] = DP45(
    x_dot,
    x0,
    t_max,
    h_limits=(1e-16, np.inf),
    atol=1e-5,
    rtol=1e-3,
    control_params=ControllerPIParams(
        coeff_i=1.0 / 4.0, coeff_p=0.0, s_limits=(0.2, 5.0)
    ),
)
results["RKX4"] = RKX4(
    x_dot,
    x0,
    t_max,
    h_limits=(1e-16, np.inf),
    atol=1e-5,
    rtol=1e-3,
)
results["RKX4_3"] = RKX4(
    x_dot,
    x0,
    t_max,
    h_limits=(1e-16, np.inf),
    atol=1e-5,
    rtol=1e-3,
    extrap_step_ratio=3,
)
results["RK4"] = RK4(x_dot, x0, t_max, 0.67 * t_max / len(results["DP45"][0]))

solver_eulex = EulerExtrapolation(
    x_dot, table_size=8, step_controller=StepControllerExtrapKH(atol=1e-5, rtol=1e-3)
)
results["EULEX"] = solver_eulex.solve(x0, t_max)

solver_odex = ModMidpointExtrapolation(
    x_dot, table_size=8, step_controller=StepControllerExtrapKH(atol=1e-5, rtol=1e-3)
)
results["ODEX"] = solver_odex.solve(x0, t_max)


# results
fig, axes = plt.subplots(2, 1)
axes[0].set_ylim(-5, 5)
axes[1].set_yscale("log")
axes[1].set_ylabel("error")

t_ref = np.linspace(0, t_max, 101)
axes[0].plot(t_ref, x_analytic(t_ref)[:, 0], label="analyic", linestyle="--")

for i, (scheme_name, (time, result, solve_info)) in enumerate(results.items()):
    axes[0].plot(time, result[:, 0], label=scheme_name, color=cmap(i))
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

plt.legend(frameon=False)
plt.tight_layout()
plt.show()


#  efficiency
fig, ax = plt.subplots()

ax.set_title("Work-Precision")
ax.set_xlabel("function evaluations")
ax.set_ylabel("error")
ax.set_yscale("log")

for i, (scheme_name, (time, result, solve_info)) in enumerate(results.items()):
    ax.plot(
        solve_info["n_feval"],
        np.mean(np.linalg.norm(result - x_analytic(time), axis=1)),
        label=scheme_name,
        marker="o",
        color=cmap(i),
    )
ax.legend(frameon=False)
plt.tight_layout()
plt.show()

# step control
fig, ax = plt.subplots()
ax.set_ylabel("step")
ax.set_xlabel("time")

ax.plot(
    0.5 * (results["BS23"][0][:-1] + results["BS23"][0][1::]),
    np.diff(results["BS23"][0]),
    label="BS23",
    color=cmap(0),
)
ax.plot(
    0.5 * (results["DP45"][0][:-1] + results["DP45"][0][1::]),
    np.diff(results["DP45"][0]),
    label="DP45",
    color=cmap(1),
)
ax.plot(
    0.5 * (results["DP45_I"][0][:-1] + results["DP45_I"][0][1::]),
    np.diff(results["DP45_I"][0]),
    label="DP45_I",
    color=cmap(2),
)
ax.plot(
    0.5 * (results["RKX4"][0][:-1] + results["RKX4"][0][1::]),
    np.diff(results["RKX4"][0]),
    label="RKX4",
    color=cmap(3),
)
ax.plot(
    0.5 * (results["RKX4_3"][0][:-1] + results["RKX4_3"][0][1::]),
    np.diff(results["RKX4_3"][0]),
    label="RKX4_3",
    color=cmap(4),
)

ax.legend(frameon=False)
plt.tight_layout()
plt.show()

# local errors
fig_le, ax_le = plt.subplots()
ax_le.set_yscale("log")
ax_le.set_ylabel("local error per unit step")
ax_le.set_xlabel("time")
for i, (scheme_name, (time, result, solve_info)) in enumerate(results.items()):
    local_error = find_local_errors(x_dot, time[1:], result[1:])

    ax_le.plot(
        time[1:],
        local_error / np.diff(time),
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

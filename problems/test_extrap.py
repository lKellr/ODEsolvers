import numpy as np
from matplotlib import pyplot as plt

from modules.step_control import (
    ControllerPIParams,
    StepControllerExtrapP,
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

cmap = plt.get_cmap("tab20")


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
results["DP45"] = DP45(
    x_dot,
    x0,
    t_max,
    h_limits=(1e-16, np.inf),
    atol=1e-5,
    rtol=1e-3,
)
h = t_max / len(
    results["DP45"][0]
)  # use the same number of steps as the adaptive scheme
results["AB_5"] = AB_k(x_dot, x0, t_max, h, k=5)

# solver_eulex = EULEX(x_dot, x0.size, table_size=8)
# results["EULEX"] = solver_eulex.solve(x0, t_max)
# solver_eulex_step = EULEX(
#     x_dot, num_odes=x0.size, table_size=8, step_controller=StepControllerExtrapK()
# )
# results["EULEX_const_step"] = solver_eulex_step.solve(x0, t_max)
solver_eulex_ord = EULEX(step_controller=StepControllerExtrapH())
results["EULEX_const_ord"] = solver_eulex_ord.solve(x0, t_max)
# solver_odex = ODEX(x_dot, x0.size, table_size=8)
# results["ODEX"] = solver_odex.solve(x0, t_max)
# solver_seulex = SEULEX(x_dot, x0.size, table_size=8)
# results["SEULEX"] = solver_seulex.solve(x0, t_max)
# solver_seulex_quad = SEULEX(x_dot, x0.size, table_size=20, dtype=np.longdouble)
# results["SEULEX_quad"] = solver_seulex_quad.solve(x0, t_max)

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

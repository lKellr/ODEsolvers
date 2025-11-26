import numpy as np
from matplotlib import pyplot as plt

from modules.step_control import ControllerParams
from solvers.embedded import *
from solvers.explicit import *
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
results["BS23"] = BS23(x_dot, x0, t_max, atol=1e-5, rtol=1e-3)
h = t_max / len(
    results["BS23"][0]
)  # use the same number of steps as the adaptive scheme to show its advantages
results["Euler"] = Euler(x_dot, x0, t_max, h)
results["Midpoint"] = Midpoint(x_dot, x0, t_max, h)
results["Heun"] = Heun(x_dot, x0, t_max, h)
results["AB2"] = AB2(x_dot, x0, t_max, h)
results["AB3"] = AB3(x_dot, x0, t_max, h)
results["PEC"] = PEC(x_dot, x0, t_max, h, n_rep=1)
results["PECE"] = PECE(x_dot, x0, t_max, h, n_rep=1)
results["AB_6"] = AB_k(x_dot, x0, t_max, h, k=6)
results["SSPRK3"] = SSPRK3(x_dot, x0, t_max, h)
results["SSPRK34"] = SSPRK34(x_dot, x0, t_max, h)
results["DP45"] = DP45(
    x_dot,
    x0,
    t_max,
    h_limits=(1e-16, np.inf),
    atol=1e-5,
    rtol=1e-3,
)
results["DP45_strict"] = DP45(
    x_dot,
    x0,
    t_max,
    h_limits=(1e-16, np.inf),
    atol=1e-8,
    rtol=1e-6,
    control_params=ControllerParams(
        coeff_i=1.0 / 4.0, coeff_p=0.0, s_limits=(0.2, 5.0)
    ),
)
results["DP45_I"] = DP45(
    x_dot,
    x0,
    t_max,
    h_limits=(1e-16, np.inf),
    atol=1e-5,
    rtol=1e-3,
    control_params=ControllerParams(
        coeff_i=1.0 / 4.0, coeff_p=0.0, s_limits=(0.2, 5.0)
    ),
)
results["RK4"] = RK4(x_dot, x0, t_max, t_max / len(
    results["DP45"][0]
))


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
ax.legend(frameon=False)
plt.tight_layout()
plt.show()

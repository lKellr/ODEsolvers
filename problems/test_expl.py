import numpy as np
from matplotlib import pyplot as plt
from solvers.embedded import BS23, DP45
from solvers.explicit import *
import logging

logging.basicConfig(level=logging.DEBUG)


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
results["PECE"] = PECE(x_dot, x0, t_max, h, n_rep=2)
results["SSPRK3"] = SSPRK3(x_dot, x0, t_max, h)
results["SSPRK34"] = SSPRK34(x_dot, x0, t_max, h)
results["RK4"] = RK4(x_dot, x0, t_max, h)
results["DP45"] = DP45(x_dot, x0, t_max, h)

fig, axes = plt.subplots(2, 1)
axes[0].set_ylim(-5, 5)
axes[1].set_xlim(0.1)
axes[1].set_yscale("log")
axes[1].set_ylabel("error")

t_ref = np.linspace(0, t_max, 101)
axes[0].plot(t_ref, x_analytic(t_ref)[:, 0], label="analyic", linestyle="--")

for scheme_name, (time, result, solve_info) in results.items():
    axes[0].plot(time, result[:, 0], label=scheme_name)

    axes[0].set_ylim(-5, 5)

for scheme_name, (time, result, solve_info) in results.items():
    axes[1].plot(
        time, np.linalg.norm(result - x_analytic(time), axis=1), label=scheme_name
    )
plt.legend(frameon=False)
plt.show()


##  efficiency
fig, ax = plt.subplots()

ax.set_title("Work-Precision")
ax.set_xlabel("function evaluations")
ax.set_ylabel("error")
ax.set_yscale("log")

for scheme_name, (time, result, solve_info) in results.items():
    ax.plot(
        solve_info["n_feval"],
        np.linalg.norm(result - x_analytic(time)),
        label=scheme_name,
        marker="o",
    )
ax.legend(frameon=False)
plt.show()

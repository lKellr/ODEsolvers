import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from solvers.embedded import DP45
from solvers.explicit import *
import logging
from numba import njit
from time import perf_counter

logging.basicConfig(level=logging.DEBUG)
logger_mpb = logging.getLogger("matplotlib")
logger_mpb.setLevel(logging.INFO)
logger_nb = logging.getLogger("numba")
logger_nb.setLevel(logging.INFO)

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


t_max = 1.0
# TODO: use astropy ephemerides, currently i am starting with all planets in phase (on the x-axis)
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

prof_tim_start = perf_counter()
time, result, solve_info = DP45(
    x_dot, x0, t_max, h_limits=(1e-16, np.inf), atol=1e-5, rtol=1e-3
)
# time, result, solve_info = Euler(x_dot, x0, t_max, h=1e-3)
prof_elapsed = perf_counter() - prof_tim_start
print(f"solution took {prof_elapsed:.3f} s")  # 5.4s with numba njit, 20.2 s without


fig, ax = plt.subplots(figsize=(16, 6), layout="tight")
fig.set_tight_layout(True)
time_text = ax.text(
    0.95, 0.05, f"t = 0 yr", transform=ax.transAxes, horizontalalignment="right"
)

traces = []
for i in range(0, n * dim, dim):
    ax.plot(result[:, i], result[:, i + 1], "--", alpha=0.2)
    traces.append(ax.plot(result[:1, i], result[:1, i + 1], label=names[i // dim])[0])

pcol = ax.scatter(
    result[0, : n * dim : 2],
    result[0, 1 : n * dim : 2],
    s=5e3 * np.log10(1 + masses ** (2 / 3)),
)


def update(frame):
    time_text.set_text(f"t = {time[frame]:.2f} yr")
    for i in range(n):
        traces[i].set_data(
            result[: frame + 1, i * dim], result[: frame + 1, i * dim + 1]
        )
    pcol.set_offsets(
        result[frame, : n * dim].reshape(-1, dim)
    )  # TODO: willl not work for 3D
    return (
        time_text,
        traces,
        pcol,
    )


ani = animation.FuncAnimation(
    fig=fig, func=update, frames=time.size, interval=200, repeat=True
)
ax.set_aspect("equal", "box")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=n)
ani.save("N-body.gif", fps=5)
plt.show()

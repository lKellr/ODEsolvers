from numba import njit
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from solvers.embedded import DP45
from solvers.explicit import *
import logging

logging.basicConfig(level=logging.DEBUG)
logger_mpb = logging.getLogger("matplotlib")
logger_mpb.setLevel(logging.INFO)

# We solve the equations nondimensionalized, with reference length 1 AU, reference time 1 year and reference mass equal to earth's mass; the second object is the moon
masses = np.array([1.0, 0.012])
masses2 = np.repeat(masses, 2)
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


t_max = 1e-5
initial_positions = np.array(
    [
        [0.0, 0.0],
        [0.002569, 0.0],
    ]
)
initial_velocities = np.array(  # calculation from circular orbit
    [[0.0, 0.0]]
    + [[0.0, np.sqrt(G * masses[0] / initial_positions[i, 0])] for i in range(1, n)]
)

x0 = np.hstack((initial_positions.flatten(), masses2 * initial_velocities.flatten()))

assert x0.size == (2 * n * dim), f"Wrong shape {x0.shape} of initial condition"

time, result, solve_info = DP45(
    x_dot, x0, t_max, h_limits=(1e-16, np.inf), atol=1e-5, rtol=1e-3
)


fig, ax = plt.subplots()

for i in range(0, n * dim, dim):
    ax.plot(result[:, i], result[:, i + 1], "--", label=f"object {i//dim}")

time_text = ax.text(0.5, 0.9, f"t = 0.", transform=ax.transAxes)

pcol = ax.scatter(
    result[0, : n * dim : 2], result[0, 1 : n * dim : 2], s=500 * np.log10(1 + masses)
)


def update(frame):
    time_text.set_text(f"t = {time[frame]}")
    # pcol.set_sizes(500 * np.log10(1 + masses))
    pcol.set_offsets(
        result[frame, : n * dim].reshape(-1, dim)
    )  # TODO: willl not work for 3D
    return (
        time_text,
        pcol,
    )


ani = animation.FuncAnimation(
    fig=fig, func=update, frames=time.size, interval=500, repeat=True
)
ax.set_aspect("equal", "box")
ax.legend()
plt.show()

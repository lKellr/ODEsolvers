import numpy as np
from matplotlib import pyplot as plt
from solvers.implicit import *
from solvers.embedded import DP45
from solvers.Extrapolation_Scheme import LimplicitEulerExtrapolation
import logging

logging.basicConfig(level=logging.INFO)
logger_mpb = logging.getLogger("matplotlib")
logger_mpb.setLevel(logging.INFO)

# Zhabotinsky-Belousov reaction / Oregonator
# Due to Field and Noyes 1974,
s = 77.27
q = 8.375e-6
w = 0.161
f = 1.0

x_dot: Callable[[float, NDArray[floating]], NDArray[floating]] = lambda t, x: np.array(
    [
        s * (x[1] - x[1] * x[0] + x[0] - q * x[0] ** 2),
        1 / s * (-x[1] - x[1] * x[0] + f * x[2]),
        w * (x[0] - x[2]),
    ],
    dtype=x.dtype,
)
t_max = 350.0
x0 = np.array([488.68, 0.99796, 488.68])

time, result, solve_info = BDF3(x_dot, x0, t_max, h=1e-1)
# time, result, solve_info = DP45(x_dot, x0, t_max)
# solver_seulex = LimplicitEulerExtrapolation(x_dot, num_odes=x0.size, table_size=16)
# time, result, solve_info = solver_seulex.solve(x0, t_max)

fig, ax = plt.subplots(nrows=3, sharex=True)

ax[2].set_xlabel(r"$\tau$")
ax[0].set_ylabel(r"$\log(\alpha)$")
ax[1].set_ylabel(r"$\log(\eta)$")
ax[2].set_ylabel(r"$\log(\rho)$")
ax[0].set_yscale("log")
ax[1].set_yscale("log")
ax[2].set_yscale("log")

ax[0].plot(time, result[:, 0])
ax[1].plot(time, result[:, 1])
ax[2].plot(time, result[:, 2])
plt.show()


fig, ax = plt.subplots()

ax.set_xlabel(r"$\log(\alpha)$")
ax.set_ylabel(r"$\log(\eta)$")
ax.set_xscale("log")
ax.set_yscale("log")

ax.plot(result[:, 1], result[:, 0])
plt.show()

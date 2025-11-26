import numpy as np
from matplotlib import pyplot as plt
import logging

from solvers.embedded import DP45

# logging.basicConfig(level=logging.DEBUG)
# logger_mpb = logging.getLogger("matplotlib")
# logger_mpb.setLevel(logging.INFO)

## Duffing oscillator
alpha = -1.0
beta = 1.0
gamma = 3.0
delta = 0.02
omega = 1.0


x_dot = lambda t, x: np.array(
    [x[1], gamma * np.cos(omega * t) - (delta * x[1] + alpha * x[0] + beta * x[0] ** 3)]
)

t_max = 800 * np.pi
x0 = np.array([1.0, 0])


time, result, solve_info = DP45(
    x_dot,
    x0,
    t_max,
    h_limits=(1e-16, np.inf),
    atol=1e-8,
    rtol=1e-6,
)

# fig, ax = plt.subplots()
# # ax.plot(result[0], result[1])
# ax.plot(time, result[:, 0], marker="o")
# plt.show()

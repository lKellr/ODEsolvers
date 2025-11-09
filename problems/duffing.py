import numpy as np
from matplotlib import pyplot as plt
from solvers.simple_explicit import *
from solvers.Extrapolation_Scheme import SEULEX
import logging

logging.basicConfig(level=logging.DEBUG)

## Duffing oscillator
alpha = -1.0
beta = 1.0
gamma = 3.0
delta = 0.02
omega = 1.0


x_dot = lambda t, x: np.array(
    [x[1], gamma * np.cos(omega * t) - (delta * x[1] + alpha * x[0] + beta * x[0] ** 3)]
)

t_max = 8 * np.pi
x0 = np.array([1.0, 0])


s = SEULEX(ode_fun=x_dot, num_odes=2, jac_fun=None, atol=1e-3, rtol=1e-2, table_size=12)
time, result, solve_info = s.solve(x0, t_max, 1e-3)

fig, ax = plt.subplots()
# ax.plot(result[0], result[1])
ax.plot(time, result[:, 0], marker="o")
plt.show()

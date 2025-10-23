import numpy as np
from matplotlib import pyplot as plt
from solvers.simple import *
from solvers.Extrapolation_Scheme import SEULEX
import logging

logging.basicConfig(level=logging.DEBUG)

# Lorenz System
rho = 28.0
sigma = 10.0
beta = 8.0 / 3.0
x_dot = lambda t, x: np.array(
    [sigma * (x[1] - x[0]), x[0] * (rho - x[2]) - x[1], x[0] * x[1] - beta * x[2]]
)
t_max = 100.0
x0 = np.array([1.0, 1.0, 1.0])

s = SEULEX(ode_fun=x_dot, num_odes=2, jac_fun=None, atol=1e-3, rtol=1e-2, table_size=12)
time, result, solve_info = s.solve(x0, t_max, 1e-3)

fig, ax = plt.subplots()
# ax.plot(result[0], result[1])
ax.plot(time, result[:, 0], marker="o")
plt.show()

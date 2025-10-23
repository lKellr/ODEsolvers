import numpy as np
from matplotlib import pyplot as plt
from solvers.solvers import *
from solvers.Extrapolation_Scheme import SEULEX
import logging

logging.basicConfig(level=logging.DEBUG)

# Robertson reaction
a = 0.04
b = 1e4
c = 3e7
x_dot = lambda t, x: np.array(
    [
        -a * x[0] + b * x[1] * x[2],
        a * x[0] - b * x[1] * x[2] - c * x[1] * x[1],
        c * x[1] * x[1],
    ],
    dtype=x.dtype,
)
t_max = 40.0  # interesting solutions for t_max ~ 1e11
x0 = np.array([1.0, 0.0, 0.0])

s = SEULEX(ode_fun=x_dot, num_odes=2, jac_fun=None, atol=1e-3, rtol=1e-2, table_size=12)
time, result, solve_info = s.solve(x0, t_max, 1e-3)

fig, ax = plt.subplots()
# ax.plot(result[0], result[1])
ax.plot(time, result[:, 0], marker="o")
plt.show()

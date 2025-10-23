from numpy._typing._array_like import NDArray


from typing import Any


import numpy as np
from matplotlib import pyplot as plt
from solvers.simple import *
from solvers.Extrapolation_Scheme import SEULEX
import logging

logging.basicConfig(level=logging.DEBUG)

# van der Pol
# mu = 5.
# x_dot = lambda t, x: np.array([x[1], mu*(1 - x[0]**2)*x[1] - x[0]])
# t_max = 1
# x0 = np.array([2., 0])
# jac = lambda t, x: np.array([[0, 1], [-2*mu*x[0]*x[1] - 1., mu*(1-x[0]**2)]])

# rescaled Van der Pol oscillator
epsilon = 1e-6
x_dot = lambda t, x: np.array([x[1], ((1 - x[0] ** 2) * x[1] - x[0]) / epsilon])
t_max = 1
x0: NDArray[Any] = np.array([2.0, 0.0])


s = SEULEX(ode_fun=x_dot, num_odes=2, jac_fun=None, atol=1e-3, rtol=1e-2, table_size=12)
time, result, solve_info = s.solve(x0, t_max, 1e-3)

fig, ax = plt.subplots()
# ax.plot(result[0], result[1])
ax.plot(time, result[:, 0], marker="o")
plt.show()

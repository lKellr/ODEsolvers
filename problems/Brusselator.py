import numpy as np
from matplotlib import pyplot as plt
from solvers.Extrapolation_Scheme import EulerExtrapolation
import logging

logging.basicConfig(level=logging.DEBUG)

# Brusselator reaction
x_dot = lambda t, x: np.array(
    [1.0 + x[0] * x[0] * x[1] - 4 * x[0], 3 * x[0] - x[0] * x[0] * x[1]],
    dtype=x.dtype,
)
t_max = 20.0  # interesting solutions for t_max ~ 1e11
x0 = np.array([1.5, 3.0])

s = EulerExtrapolation(ode_fun=x_dot, table_size=12)
time, result, solve_info = s.solve(x0, t_max)
fig, ax = plt.subplots()
# ax.plot(result[0], result[1])
ax.plot(time, result[:, 0], marker="o")
plt.show()

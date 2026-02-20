from readline import backend
import numpy as np
from matplotlib import pyplot as plt

from modules.step_control import (
    ControllerPIParams,
    StepControllerExtrapDummy,
    StepControllerExtrapK,
    StepControllerExtrapH,
    StepControllerExtrapKH,
)
from solvers.embedded import *
from solvers.explicit import *
from solvers.Extrapolation_Scheme import *
import logging

# logging.basicConfig(level=logging.DEBUG)
# logger_mpb = logging.getLogger("matplotlib")
# logger_mpb.setLevel(logging.INFO)
# logger_pil = logging.getLogger("PIL")
# logger_pil.setLevel(logging.INFO)

cmap = plt.get_cmap("tab20")


x_dot: Callable[[float, NDArray[np.floating]], NDArray[np.floating]] = lambda t, x: x*(2. - np.sin(t))

t_max = 1.0
x0 = np.array([2.])

x_analytic: Callable[[float], NDArray[np.floating]] = lambda t: 2*np.exp(2*t + np.cos(t) - 1.)

norm = norm_hairer

N_list = np.array([3*2 ** (k // 2) if k == 1 or k % 2 == 0 else 3*1.5 * 2 ** (k // 2)for k in range(2, 8)])
k_list = np.array(range(5))
conv_data = dict()


errors: list[float] = list()
for n_steps in N_list:
  time, result, solve_info = AB_k(x_dot, x0, t_max, h=t_max/n_steps, k=5)
  errors.append(norm(result[-1] - x_analytic(time[-1])))
conv_data['AB_5'] = np.array(errors)

errors = list()
for n_steps in N_list:
  time, result, solve_info = AB3(x_dot, x0, t_max, h=t_max/n_steps)
  errors.append(norm(result[-1] - x_analytic(time[-1])))
conv_data['AB_3'] = np.array(errors)

errors = list()
for n_steps in N_list:
  time, result, solve_info = RK4(x_dot, x0, t_max, h=t_max/n_steps)
  errors.append(norm(result[-1] - x_analytic(time[-1])))
conv_data['RK4'] = np.array(errors)

errors = list()
for n_steps in N_list:
  solver_eulex2 = EulerExtrapolation(
    ode_fun=x_dot,
    table_size=8,
    step_controller=StepControllerExtrapDummy(),
  )
  time, result, solve_info = solver_eulex2.solve(x0, t_max, k_initial=2, h_initial=t_max/n_steps)
  errors.append(norm(result[-1]-x_analytic(time[-1])))
conv_data['EULEX2'] = np.array(errors)

# solver_eulex_quad = EulerExtrapolation(
#     x_dot,
#     table_size=8,
#     step_controller=StepControllerExtrapKH(atol=1e-7, rtol=1e-5),
#     dtype=np.longdouble,
# )
# results["EULEX7_quad"] = solver_eulex_quad.solve(x0, t_max)

# solver_eulex_step = EulerExtrapolation(
#     x_dot, table_size=16, step_controller=StepControllerExtrapK(atol=1e-5, rtol=1e-3)
# )
# results["EULEX_const_step"] = solver_eulex_step.solve(x0, t_max, h_initial=h_average)

# solver_odex = ModMidpointExtrapolation(x_dot, table_size=8)
# results["ODEX"] = solver_odex.solve(x0, t_max)

# solver_odex_smoothed = ModMidpointExtrapolation(x_dot, table_size=8, use_smoothing=True)
# results["ODEX_smoothed"] = solver_odex_smoothed.solve(x0, t_max)

# results
fig, ax = plt.subplots()
# ax.set_ylim(-5, 5)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("h")
ax.set_ylabel("error")

for i, (scheme_name, errors) in enumerate(conv_data.items()):
    ax.plot(t_max/N_list, errors, label=scheme_name, color=cmap(i))
    rate = np.log(errors[1:]/errors[:-1])/np.log(N_list[:-1]/N_list[1:])
    ax.text(t_max/N_list[-1], errors[-1], f"$p = {rate[-1]:.2f}$")


plt.legend(frameon=False)
plt.tight_layout()
plt.show()

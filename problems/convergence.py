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

logging.basicConfig(level=logging.DEBUG)
logger_mpb = logging.getLogger("matplotlib")
logger_mpb.setLevel(logging.INFO)
logger_pil = logging.getLogger("PIL")
logger_pil.setLevel(logging.INFO)

cmap = plt.get_cmap("tab20")


x_dot: Callable[[float, NDArray[np.floating]], NDArray[np.floating]] = lambda t, x: x*(2. - np.sin(t))
jac: Callable[[float, NDArray[np.floating]], NDArray[np.floating]] = (
    lambda t, x: 2.0 - np.sin(t)
)

t_max = 1.0
x0 = np.array([2.])

x_analytic: Callable[[float], NDArray[np.floating]] = lambda t: 2*np.exp(2*t + np.cos(t) - 1.)

norm = norm_hairer

N_list = 4*np.array([2 ** (k // 2) if k == 1 or k % 2 == 0 else 1.5 * 2 ** (k // 2) for k in range(1,8)])
k_list = np.array(range(5))
conv_data = dict()

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
  time, result, solve_info = solver_eulex2.solve(x0, t_max, k_initial=1, h_initial=t_max/n_steps)
  errors.append(norm(result[-1]-x_analytic(time[-1])))
conv_data['EULEX2'] = np.array(errors)

errors = list()
for n_steps in N_list:
  solver_eulex3 = EulerExtrapolation(
    ode_fun=x_dot,
    table_size=8,
    step_controller=StepControllerExtrapDummy(),
  )
  time, result, solve_info = solver_eulex3.solve(x0, t_max, k_initial=2, h_initial=t_max/n_steps)
  errors.append(norm(result[-1]-x_analytic(time[-1])))
conv_data['EULEX3'] = np.array(errors)

errors = list()
for n_steps in N_list:
  solver_eulex5 = EulerExtrapolation(
    ode_fun=x_dot,
    table_size=8,
    step_controller=StepControllerExtrapDummy(),
  )
  time, result, solve_info = solver_eulex5.solve(x0, t_max, k_initial=4, h_initial=t_max/n_steps)
  errors.append(norm(result[-1]-x_analytic(time[-1])))
conv_data['EULEX5'] = np.array(errors)

errors = list()
for n_steps in N_list:
    solver_eulex5_rat = EulerExtrapolationRational(
        ode_fun=x_dot,
        table_size=8,
        step_controller=StepControllerExtrapDummy(),
    )
    time, result, solve_info = solver_eulex5_rat.solve(
        x0, t_max, k_initial=4, h_initial=t_max / n_steps
    )
    errors.append(norm(result[-1] - x_analytic(time[-1])))
conv_data["EULEX5_rational"] = np.array(errors)

errors = list()
for n_steps in N_list:
  solver_eulex7 = EulerExtrapolation(
    ode_fun=x_dot,
    table_size=8,
    step_controller=StepControllerExtrapDummy(),
  )
  time, result, solve_info = solver_eulex7.solve(x0, t_max, k_initial=6, h_initial=t_max/n_steps)
  errors.append(norm(result[-1]-x_analytic(time[-1])))
conv_data['EULEX7'] = np.array(errors)

errors = list()
for n_steps in N_list:
    solver_eulex7ld = EulerExtrapolation(
        ode_fun=x_dot,
        table_size=8,
        step_controller=StepControllerExtrapDummy(),
        dtype=np.longdouble,
    )
    time, result, solve_info = solver_eulex7ld.solve(
        x0.astype(np.longdouble), t_max, k_initial=6, h_initial=t_max / n_steps
    )
    errors.append(norm(result[-1] - x_analytic(time[-1])))
conv_data["EULEX7_longdouble"] = np.array(errors)

# TODO: check ODEX order k
errors = list()
for n_steps in N_list:
    solver_odex3 = ModMidpointExtrapolation(
        ode_fun=x_dot, table_size=8, step_controller=StepControllerExtrapDummy()
    )
    time, result, solve_info = solver_odex3.solve(
        x0, t_max, k_initial=1, h_initial=t_max / n_steps
    )
    errors.append(norm(result[-1] - x_analytic(time[-1])))
conv_data["ODEX3"] = np.array(errors)

errors = list()
for n_steps in N_list:
    solver_odex3s = ModMidpointExtrapolation(
        ode_fun=x_dot,
        table_size=8,
        step_controller=StepControllerExtrapDummy(),
        use_smoothing=True,
    )
    time, result, solve_info = solver_odex3s.solve(
        x0, t_max, k_initial=1, h_initial=t_max / n_steps
    )
    errors.append(norm(result[-1] - x_analytic(time[-1])))
conv_data["ODEX3_smoothed"] = np.array(errors)

errors = list()
for n_steps in N_list:
    solver_odex5 = ModMidpointExtrapolation(
        ode_fun=x_dot, table_size=8, step_controller=StepControllerExtrapDummy()
    )
    time, result, solve_info = solver_odex5.solve(
        x0, t_max, k_initial=2, h_initial=t_max / n_steps
    )
    errors.append(norm(result[-1] - x_analytic(time[-1])))
conv_data["ODEX5"] = np.array(errors)

errors = list()
for n_steps in N_list:
    solver_seulex2 = LimplicitEulerExtrapolation(
        ode_fun=x_dot,
        table_size=8,
        num_odes=1,
        jac_fun=jac,
        step_controller=StepControllerExtrapDummy(),
    )
    time, result, solve_info = solver_seulex2.solve(
        x0, t_max, k_initial=1, h_initial=t_max / n_steps
    )
    errors.append(norm(result[-1] - x_analytic(time[-1])))
conv_data["SEULEX2"] = np.array(errors)

errors = list()
for n_steps in N_list:
    solver_seulex2 = LimplicitEulerExtrapolation(
        ode_fun=x_dot,
        table_size=8,
        num_odes=1,
        jac_fun=jac,
        step_controller=StepControllerExtrapDummy(),
    )
    time, result, solve_info = solver_seulex2.solve(
        x0, t_max, k_initial=4, h_initial=t_max / n_steps
    )
    errors.append(norm(result[-1] - x_analytic(time[-1])))
conv_data["SEULEX5"] = np.array(errors)

errors = list()
for n_steps in N_list:
    solver_seulex2_nj = LimplicitEulerExtrapolation(
        ode_fun=x_dot,
        table_size=8,
        num_odes=1,
        jac_fun=lambda t, x: numerical_jacobian_t(t, x, x_dot, delta=1e-12),
        step_controller=StepControllerExtrapDummy(),
    )
    time, result, solve_info = solver_seulex2_nj.solve(
        x0, t_max, k_initial=4, h_initial=t_max / n_steps
    )
    errors.append(norm(result[-1] - x_analytic(time[-1])))
conv_data["SEULEX5_numjac"] = np.array(errors)

errors = list()
for n_steps in N_list:
    solver_sodex3 = LimplicitMidpointExtrapolation(
        ode_fun=x_dot,
        table_size=8,
        num_odes=1,
        jac_fun=jac,
        step_controller=StepControllerExtrapDummy(),
        use_smoothing=True,
    )
    time, result, solve_info = solver_sodex3.solve(
        x0, t_max, k_initial=1, h_initial=t_max / n_steps
    )
    errors.append(norm(result[-1] - x_analytic(time[-1])))
conv_data["SODEX3"] = np.array(errors)

# errors = list()
# for n_steps in N_list:
#     solver_sodex3s = LimplicitMidpointExtrapolation(
#         ode_fun=x_dot,
#         table_size=8,
#         num_odes=1,
#         jac_fun=jac,
#         step_controller=StepControllerExtrapDummy(),
#         use_smoothing=True,
#     )
#     time, result, solve_info = solver_sodex3s.solve(
#         x0, t_max, k_initial=1, h_initial=t_max / n_steps
#     )
#     errors.append(norm(result[-1] - x_analytic(time[-1])))
# conv_data["SODEX3smoothed"] = np.array(errors)

# results
fig, ax = plt.subplots()
# ax.set_ylim(-5, 5)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("h")
ax.set_ylabel("error")

for i, (scheme_name, errors) in enumerate(conv_data.items()):
    ax.plot(t_max/N_list, errors, label=scheme_name, color=cmap(i), marker='o')
    rate = np.log(errors[1:]/errors[:-1])/np.log(N_list[:-1]/N_list[1:])
    ax.text(t_max / N_list[-1], errors[-1], rf"$\bar{{p}} = {np.mean(rate):.2f}$")
    ax.text(t_max / N_list[0], errors[0], f"$p_0 = {rate[0]:.2f}$")


plt.legend(frameon=False)
plt.tight_layout()
plt.show()

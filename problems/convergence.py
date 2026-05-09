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

N_list = 4 * np.array(
    [
        2 ** (k // 2) if k == 1 or k % 2 == 0 else 1.5 * 2 ** (k // 2)
        for k in range(1, 8)
    ]
)
k_list = np.array(range(5))
conv_data = dict()

errors = list()
h_mins = list()
for n_steps in N_list:
    time, result, solve_info = AB3(x_dot, x0, t_max, h=t_max / n_steps)
    errors.append(norm(result[-1] - x_analytic(time[-1])))
    h_mins.append(t_max / n_steps)
conv_data["AB3"] = np.array(errors), np.array(h_mins)

errors = list()
h_mins = list()
for n_steps in N_list:
    time, result, solve_info = RK4(x_dot, x0, t_max, h=t_max / n_steps)
    errors.append(norm(result[-1] - x_analytic(time[-1])))
    h_mins.append(t_max / n_steps)
conv_data["RK4"] = np.array(errors), np.array(h_mins)

errors = list()
h_mins = list()
for n_steps in N_list:
    solver_eulex2 = EulerExtrapolation(
        ode_fun=x_dot,
        table_size=8,
        step_controller=StepControllerExtrapDummy(),
    )
    time, result, solve_info = solver_eulex2.solve(
        x0, t_max, k_initial=1, h_initial=t_max / n_steps
    )
    errors.append(norm(result[-1] - x_analytic(time[-1])))
    h_mins.append(np.mean(solve_info["h_min"]))
conv_data["EULEX2"] = np.array(errors), np.array(h_mins)

errors = list()
h_mins = list()
for n_steps in N_list:
    solver_eulex3 = EulerExtrapolation(
        ode_fun=x_dot,
        table_size=8,
        step_controller=StepControllerExtrapDummy(),
    )
    time, result, solve_info = solver_eulex3.solve(
        x0, t_max, k_initial=2, h_initial=t_max / n_steps
    )
    errors.append(norm(result[-1] - x_analytic(time[-1])))
    h_mins.append(np.mean(solve_info["h_min"]))
conv_data["EULEX3"] = np.array(errors), np.array(h_mins)

errors = list()
h_mins = list()
for n_steps in N_list:
    solver_eulex5 = EulerExtrapolation(
        ode_fun=x_dot,
        table_size=8,
        step_controller=StepControllerExtrapDummy(),
    )
    time, result, solve_info = solver_eulex5.solve(
        x0, t_max, k_initial=4, h_initial=t_max / n_steps
    )
    errors.append(norm(result[-1] - x_analytic(time[-1])))
    h_mins.append(np.mean(solve_info["h_min"]))
conv_data["EULEX5"] = np.array(errors), np.array(h_mins)

errors = list()
h_mins = list()
for n_steps in N_list:
    solver_odex_rat = ModMidpointExtrapolationRational(
        ode_fun=x_dot,
        table_size=8,
        step_controller=StepControllerExtrapDummy(),
    )
    time, result, solve_info = solver_odex_rat.solve(
        x0, t_max, k_initial=4, h_initial=t_max / n_steps
    )
    errors.append(norm(result[-1] - x_analytic(time[-1])))
    h_mins.append(np.mean(solve_info["h_min"]))
conv_data["ODEX5_rational_smoothed"] = np.array(errors), np.array(h_mins)

errors = list()
h_mins = list()
for n_steps in N_list:
    solver_eulex7 = EulerExtrapolation(
        ode_fun=x_dot,
        table_size=8,
        step_controller=StepControllerExtrapDummy(),
    )
    time, result, solve_info = solver_eulex7.solve(
        x0, t_max, k_initial=6, h_initial=t_max / n_steps
    )
    errors.append(norm(result[-1] - x_analytic(time[-1])))
    h_mins.append(np.mean(solve_info["h_min"]))
conv_data["EULEX7"] = np.array(errors), np.array(h_mins)


errors = list()
h_mins = list()
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
    h_mins.append(np.mean(solve_info["h_min"]))
conv_data["EULEX7_longdouble"] = np.array(errors), np.array(h_mins)

# NOTE: ODEX schemes converge at higher order than predicted for linear problems
errors = list()
h_mins = list()
for n_steps in N_list:
    solver_odex3 = ModMidpointExtrapolation(
        ode_fun=x_dot, table_size=8, step_controller=StepControllerExtrapDummy()
    )
    time, result, solve_info = solver_odex3.solve(
        x0, t_max, k_initial=1, h_initial=t_max / n_steps
    )
    errors.append(norm(result[-1] - x_analytic(time[-1])))
    h_mins.append(np.mean(solve_info["h_min"]))
conv_data["ODEX3"] = np.array(errors), np.array(h_mins)

errors = list()
h_mins = list()
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
    h_mins.append(np.mean(solve_info["h_min"]))
conv_data["ODEX3_smoothed"] = np.array(errors), np.array(h_mins)

errors = list()
h_mins = list()
for n_steps in N_list:
    solver_odex5 = ModMidpointExtrapolation(
        ode_fun=x_dot, table_size=8, step_controller=StepControllerExtrapDummy()
    )
    time, result, solve_info = solver_odex5.solve(
        x0, t_max, k_initial=2, h_initial=t_max / n_steps
    )
    errors.append(norm(result[-1] - x_analytic(time[-1])))
    h_mins.append(np.mean(solve_info["h_min"]))
conv_data["ODEX5"] = np.array(errors), np.array(h_mins)

errors = list()
h_mins = list()
for n_steps in N_list:
    solver_odex7 = ModMidpointExtrapolation(
        ode_fun=x_dot, table_size=8, step_controller=StepControllerExtrapDummy()
    )
    time, result, solve_info = solver_odex7.solve(
        x0, t_max, k_initial=3, h_initial=t_max / n_steps
    )
    errors.append(norm(result[-1] - x_analytic(time[-1])))
    h_mins.append(np.mean(solve_info["h_min"]))
conv_data["ODEX7"] = np.array(errors), np.array(h_mins)

errors = list()
h_mins = list()
for n_steps in N_list:
    solver_odex7ld = ModMidpointExtrapolation(
        ode_fun=x_dot,
        table_size=8,
        step_controller=StepControllerExtrapDummy(),
        dtype=np.longdouble,
    )
    time, result, solve_info = solver_odex7ld.solve(
        x0.astype(np.longdouble), t_max, k_initial=3, h_initial=t_max / n_steps
    )
    errors.append(norm(result[-1] - x_analytic(time[-1])))
    h_mins.append(np.mean(solve_info["h_min"]))
conv_data["ODEX7_longdouble"] = np.array(errors), np.array(h_mins)

errors = list()
h_mins = list()
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
    h_mins.append(np.mean(solve_info["h_min"]))
conv_data["SEULEX2"] = np.array(errors), np.array(h_mins)

errors = list()
h_mins = list()
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
    h_mins.append(np.mean(solve_info["h_min"]))
conv_data["SEULEX5"] = np.array(errors), np.array(h_mins)

errors = list()
h_mins = list()
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
    h_mins.append(np.mean(solve_info["h_min"]))
conv_data["SEULEX5_numjac"] = np.array(errors), np.array(h_mins)

errors = list()
h_mins = list()
for n_steps in N_list:
    solver_sodex3 = LimplicitMidpointExtrapolation(
        ode_fun=x_dot,
        table_size=8,
        num_odes=1,
        jac_fun=jac,
        step_controller=StepControllerExtrapDummy(),
        use_smoothing=False,
    )
    time, result, solve_info = solver_sodex3.solve(
        x0, t_max, k_initial=1, h_initial=t_max / n_steps
    )
    errors.append(norm(result[-1] - x_analytic(time[-1])))
    h_mins.append(np.mean(solve_info["h_min"]))
conv_data["SODEX3"] = np.array(errors), np.array(h_mins)

errors = list()
h_mins = list()
for n_steps in N_list:
    solver_sodex3s = LimplicitMidpointExtrapolation(
        ode_fun=x_dot,
        table_size=8,
        num_odes=1,
        jac_fun=jac,
        step_controller=StepControllerExtrapDummy(),
        use_smoothing=True,
    )
    time, result, solve_info = solver_sodex3s.solve(
        x0, t_max, k_initial=1, h_initial=t_max / n_steps
    )
    errors.append(norm(result[-1] - x_analytic(time[-1])))
    h_mins.append(np.mean(solve_info["h_min"]))
conv_data["SODEX3_smoothed"] = np.array(errors), np.array(h_mins)

# results
fig, ax = plt.subplots(figsize=(9.6, 7.2))
# ax.set_ylim(-5, 5)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("H")
ax.set_ylabel("error")

for i, (scheme_name, (errors, _)) in enumerate(conv_data.items()):
    ax.plot(
        t_max / N_list,
        errors,
        label=scheme_name,
        color=cmap(i),
        marker="o",
        linestyle="--" if scheme_name in ["AB3", "RK4"] else "-",
    )
    rate = np.log(errors[1:]/errors[:-1])/np.log(N_list[:-1]/N_list[1:])
    ax.text(t_max / N_list[-1], errors[-1], rf"$\bar{{p}} = {np.mean(rate):.2f}$")
    ax.text(t_max / N_list[0], errors[0], f"$p_0 = {rate[0]:.2f}$")


plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("convergence.png")
plt.show()

# plot over minimum h
fig, ax = plt.subplots(figsize=(9.6, 7.2))
# ax.set_ylim(-5, 5)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$h_\mathrm{min}$")
ax.set_ylabel("error")

for i, (scheme_name, (errors, h_min)) in enumerate(conv_data.items()):
    ax.plot(
        h_min,
        errors,
        label=scheme_name,
        color=cmap(i),
        marker="o",
    )
    rate = np.log(errors[1:] / errors[:-1]) / np.log(h_min[1:] / h_min[:-1])
    ax.text(h_min[-1], errors[-1], rf"$\bar{{p}} = {np.mean(rate):.2f}$")
    ax.text(h_min[0], errors[0], f"$p_0 = {rate[0]:.2f}$")


plt.legend(frameon=False)
plt.tight_layout()
plt.show()

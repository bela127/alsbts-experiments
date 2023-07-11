from dataclasses import dataclass
import os
import numpy as np
from utils import MeasComputer, save, plot_meas_over_time, plot_sum_meas_over_time, plot_meas_per_step_vs_exp_quant, base_path, save_path, create_fig, appr_style
from matplotlib import pyplot as plt


def only_nth_to_mth(meas_res, n=0, m=10):
    nth_meas_result = list(zip(*meas_res))[n:m]
    return list(zip(*nth_meas_result))

dc =MeasComputer(
    base_path=os.path.join(base_path,"./brown_exp_var_stdsel"),
    sub_exp_folder="brown_exp_var_stdsel",
)
meas_res = dc.load_exp()

meas_res1_10 = only_nth_to_mth(meas_res)
meas_res1_10_20 = only_nth_to_mth(meas_res,n=10,m=-1)

dc =MeasComputer(
    base_path=os.path.join(base_path,"./brown_est_int_brown_var_sel_std"),
    sub_exp_folder="brown_est_int_brown_var_sel_std",
)
meas_res2 = dc.load_exp()

fig, ax = create_fig(subplots=(1,1), width="paper", fraction=1/2)
colors = plot_meas_over_time(ax, meas_res1_10, exp_quant_name=r"$Brown\:v_{target}$")

ax.set_title("Distribution of Measurements over Time")
plt.xlim(0,1000)
plt.ylim(0.1,1)
plt.legend()

save(fig, name="Distribution of Measurements over Time",path=save_path)

fig, ax = create_fig(subplots=(1,1), width="paper", fraction=1/2)
colors = plot_meas_over_time(ax, meas_res1_10_20, exp_quant_name=r"$Brown\:v_{target}$")

ax.set_title("Distribution of Measurements over Time BIG ERROR")
plt.legend()

save(fig, name="Distribution of Measurements over Time BIG ERROR",path=save_path)


exp_quant = np.linspace(0,1.6,100)
def f(x):
    a = 1.41e+6 # 1.409027e+6
    b = 1.89e+22 #1.892108563891492e+22
    c = 0.321 # 0.3209065
    d = 0.07 # 0.06983672
    return (a + d)/(1 + (b*x)**c) - d

est_meas_res = f(exp_quant)

meas_res2_n1 = only_nth_to_mth(meas_res,n=0,m=-2)

fig, ax = create_fig(subplots=(1,1), width="paper", fraction=1/2)

ax.plot(exp_quant, est_meas_res, label=r"Fitted Relation $\hat{m}_{su} = \frac{a + d}{1 + (b*v_{target})^c} - d$", linestyle='dotted', linewidth=0.5, color='black')
plot_meas_per_step_vs_exp_quant(ax, meas_res, exp_quant_name=r"$\mathbf{Brown}\:v_{target}\in[10^{-1},10^{-2,5}]$", color=appr_style["BR"][0])
plot_meas_per_step_vs_exp_quant(ax, meas_res2_n1, exp_quant_name=r"$\mathbf{Int\_Brown}\:v_{target}\in[10^{-1},10^{-2,5}]$", color=appr_style["IB"][0])


ax.set_yscale("log")
ax.set_title("Impact of Threshold on Number of Measurements")
plt.xlim(0,1.6)
plt.ylim(0.01,0.5)
plt.legend()

ax.text(.8, .95, r"$a = 1.41e^{+6}$\\[+0.4em]$b = 1.89e^{+22}$\\[+0.4em]$c = 0.321$\\[+0.4em]$d = 0.07$",
            transform=ax.transAxes, ha="left", va="top", fontsize=6)

save(fig, name="Impact of Threshold on Number of Measurements",path=save_path)
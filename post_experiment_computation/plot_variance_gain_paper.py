import os
import numpy as np
from utils import plot_rmse_over_mean_meas, save, DataComputer, base_path, save_path, create_fig, appr_style, EvalRes
from matplotlib import pyplot as plt

dc =DataComputer(
    base_path=os.path.join(base_path,"./fixed_intervals_var_int"),
    sub_exp_folder="fixed_intervals_var_int",
)
base_eval_res = dc.load_exp()

dc =DataComputer(
    base_path=os.path.join(base_path,"./brown_exp_var_stdsel"),
    sub_exp_folder="brown_exp_var_stdsel",
)
our_eval_res = dc.load_exp()

dc =DataComputer(
    base_path=os.path.join(base_path,"./brown_est_int_brown_var_sel_std"),
    sub_exp_folder="brown_est_int_brown_var_sel_std",
)
eval_res_ours_intb = dc.load_exp()

def calc_var_gain(appr: EvalRes, base: EvalRes):

        all_meas = np.concatenate((appr.mean_meas,base.mean_meas))
        sorted_meas = np.sort(all_meas, )

        max_index = np.where(sorted_meas >= np.max(appr.mean_meas))
        min_index = np.where(sorted_meas <= np.min(base.mean_meas))
        sorted_meas = sorted_meas[min_index[0][-1]:max_index[0][0]]

        appr_var = []
        base_var = []
        needed_meas = []
        for i in range(sorted_meas.shape[0]):
            appr_index = np.where(appr.mean_meas <= sorted_meas[i])
            base_index = np.where(base.mean_meas<= sorted_meas[i])

            try:
                base_index = base_index[0][0]
            except IndexError:
                base_index = base.mean_rmse.shape[0]-1

            try:
                appr_index = appr_index[0][0]

                appr_var.append(appr.std_rmse[appr_index])
                base_var.append(base.std_rmse[base_index])
                needed_meas.append(sorted_meas[i])

            except IndexError:
                ... # The approach never reached this accuracy, so we can not display it.

        appr_vars = np.asarray(appr_var)
        base_vars = np.asarray(base_var)
        needed_meass = np.asarray(needed_meas)

        gain = base_vars / appr_vars
        return (needed_meass, gain)


def plot_var_gain(ax, var_gain, exp_quant_name="ours", color=None):
    x, y = var_gain
    ax.set_xlim(np.max(x), np.min(x))
    #ax.set_ylim(0,1)
    ax.plot(x, y, label=exp_quant_name, color=color)

    ax.set_xlabel("Nr. of measurements")
    ax.set_ylabel(r"var gain vs. $CM$")

gain1 = calc_var_gain(our_eval_res, base_eval_res)
gain2 = calc_var_gain(eval_res_ours_intb, base_eval_res)

fig, ax = create_fig(subplots=(1,1), width="paper_2c", fraction=1, hfrac=0.3)

plot_var_gain(ax, gain1, exp_quant_name=r"$\mathbf{Brown}\:v_{target}\in[10^{-1},10^{-2,5}]$" , color=appr_style["BR"][0])
plot_var_gain(ax, gain2, exp_quant_name=r"$\mathbf{Int\_Brown}\:v_{target}\in[10^{-1},10^{-2,5}]$", color=appr_style["IB"][0])


ax.set_title("Gained Variance over Measurements")
plt.legend()

save(fig= fig, name="Gained Variance over Measurements",path=save_path)



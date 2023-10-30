import os
import numpy as np
from utils import plot_rmse_over_mean_meas, save, DataComputer, EvalRes,  base_path, save_path, create_fig, appr_style
from matplotlib import pyplot as plot
import matplotlib.ticker as mtick

dc =DataComputer(
    base_path=os.path.join(base_path,"./fixed_intervals_var_int"),
    sub_exp_folder="fixed_intervals_var_int",
)
eval_res_base_b = dc.load_exp()

dc =DataComputer(
    base_path=os.path.join(base_path,"./data_brownmix_fixed_intervals_var_int"),
    sub_exp_folder="data_brownmix_fixed_intervals_var_int",
)
eval_res_base_bmix = dc.load_exp()

dc =DataComputer(
    base_path=os.path.join(base_path,"./data_mix_fixed_intervals_var_int"),
    sub_exp_folder="data_mix_fixed_intervals_var_int",
)
eval_res_base_mix = dc.load_exp()

dc =DataComputer(
    base_path=os.path.join(base_path,"./data_rbf_fixed_intervals_var_int"),
    sub_exp_folder="data_rbf_fixed_intervals_var_int",
)
eval_res_base_rbf = dc.load_exp()

dc =DataComputer(
    base_path=os.path.join(base_path,"./data_sin_fixed_intervals_var_int"),
    sub_exp_folder="data_sin_fixed_intervals_var_int",
)
eval_res_base_sin = dc.load_exp()


dc =DataComputer(
    base_path=os.path.join(base_path,"./brown_exp_var_stdsel"),
    sub_exp_folder="brown_exp_var_stdsel",
)
eval_res_ours = dc.load_exp()

dc =DataComputer(
    base_path=os.path.join(base_path,"./data_sin_brown_exp_var_stdsel"),
    sub_exp_folder="data_sin_brown_exp_var_stdsel",
)
eval_res_ours_sin = dc.load_exp()


dc =DataComputer(
    base_path=os.path.join(base_path,"./data_rbf_brown_exp_var_stdsel"),
    sub_exp_folder="data_rbf_brown_exp_var_stdsel",
)
eval_res_ours_rbf = dc.load_exp()

dc =DataComputer(
    base_path=os.path.join(base_path,"./data_mix_brown_exp_var_stdsel"),
    sub_exp_folder="data_mix_brown_exp_var_stdsel",
)
eval_res_ours_mix = dc.load_exp()

dc =DataComputer(
    base_path=os.path.join(base_path,"./data_brownmix_brown_exp_var_stdsel"),
    sub_exp_folder="data_brownmix_brown_exp_var_stdsel",
)
eval_res_ours_bmix = dc.load_exp()

#dc =DataComputer(
#    base_path=os.path.join(base_path,"./brown_est_int_brown_var_sel_std"),
#    sub_exp_folder="brown_est_int_brown_var_sel_std",
#)
#eval_res_ours_intb = dc.load_exp()

def calc_gain(appr: EvalRes, base: EvalRes):

        all_rmse = np.concatenate((appr.mean_rmse,base.mean_rmse))
        sorted_rmse = np.sort(all_rmse, )
        
        appr_max = np.max(appr.mean_rmse)
        appr_min = np.min(appr.mean_rmse)
        base_max = np.max(base.mean_rmse)
        base_min = np.min(base.mean_rmse)


        appr_measures = []
        base_measures = []
        gains_rmse = []
        for i in range(sorted_rmse.shape[0]):
            appr_index = np.where(appr.mean_rmse <= sorted_rmse[i])
            base_index = np.where(base.mean_rmse<= sorted_rmse[i])

            try:
                base_index = base_index[0][0]
            except IndexError:
                base_index = base.mean_rmse.shape[0]-1

            try:
                appr_index = appr_index[0][0]

                if base.mean_rmse[base_index] < appr_min: #Only calculate gain for RMSE values both approaches have reached. 
                     continue
                if appr.mean_rmse[appr_index] < base_min:
                     continue
                if base.mean_rmse[base_index] > appr_max:
                     continue
                if appr.mean_rmse[appr_index] > base_max:
                     continue

                appr_measures.append(appr.mean_meas[appr_index])
                base_measures.append(base.mean_meas[base_index])
                gains_rmse.append(sorted_rmse[i])

            except IndexError:
                ... # The approach never reached this accuracy, so we can not display it.
                continue

        appr_measures = np.asarray(appr_measures)
        base_measures = np.asarray(base_measures)
        gains_rmse = np.asarray(gains_rmse)

        gain = (base_measures - appr_measures) / base_measures
        return (gains_rmse, gain)


def calc_avr_gain(data_gain):
    x, y = data_gain
    mean = np.mean(y)
    return mean

def plot_data_gain(ax, data_gain, exp_quant_name="ours", color=None):
    x, y = data_gain
    ax.set_ylim(0,1)
    ax.plot(x, y, label=exp_quant_name, color=color)

    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xlabel("$RMSE$")
    ax.set_ylabel(r"saved data $sd$")
    

gain1 = calc_gain(eval_res_ours, eval_res_base_b)
avr_gain1 = calc_avr_gain(gain1)
gain2 = calc_gain(eval_res_ours_sin, eval_res_base_sin)
avr_gain2 = calc_avr_gain(gain2)
gain3 = calc_gain(eval_res_ours_rbf, eval_res_base_rbf)
avr_gain3 = calc_avr_gain(gain3)
gain4 = calc_gain(eval_res_ours_mix, eval_res_base_mix)
avr_gain4 = calc_avr_gain(gain4)
gain5 = calc_gain(eval_res_ours_bmix, eval_res_base_bmix)
avr_gain5 = calc_avr_gain(gain5)
#gain2 = calc_gain(eval_res_ours_intb, eval_res_base)

fig, ax = create_fig(subplots=(1,1), width="paper_2c", fraction=1, hfrac=1, vfrac=0.85)

plot_data_gain(ax, gain1, exp_quant_name=r"$C_{b}(x(t),t))$ avr.: " + f"{100*avr_gain1:.1f}\%" , color=appr_style["CEw"][0])
plot_data_gain(ax, gain2, exp_quant_name=r"$C_{sin}(x(t),t))$ avr.: " + f"{100*avr_gain2:.1f}\%" , color=appr_style["CEm"][0])
plot_data_gain(ax, gain3, exp_quant_name=r"$C_{rbf}(x(t),t))$ avr.: " + f"{100*avr_gain3:.1f}\%" , color=appr_style["CEo"][0])
plot_data_gain(ax, gain4, exp_quant_name=r"$C_{mix}(x(t),t))$ avr.: " + f"{100*avr_gain4:.1f}\%" , color=appr_style["CAL"][0])
plot_data_gain(ax, gain5, exp_quant_name=r"$C_{bmix}(x(t),t))$ avr.: " + f"{100*avr_gain5:.1f}\%" , color=appr_style["BR"][0])
#plot_data_gain(ax, gain2, exp_quant_name=r"$\mathbf{Int\_Brown}\:v_{target}\in[10^{-1},10^{-2,5}]$", color=appr_style["IB"][0])

#ax.set_xlim(0, np.max(gain1[0]))
ax.set_title("Data Saved by $\mathbf{DEAL}$ vs. $CM$")
plot.legend()

save(fig, name="Data Saved by Our Approach", path=save_path)
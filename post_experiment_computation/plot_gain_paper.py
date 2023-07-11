import os
import numpy as np
from utils import plot_rmse_over_mean_meas, save, DataComputer, EvalRes,  base_path, save_path, create_fig, appr_style
from matplotlib import pyplot as plot

dc =DataComputer(
    base_path=os.path.join(base_path,"./fixed_intervals_var_int"),
    sub_exp_folder="fixed_intervals_var_int",
)
eval_res_base = dc.load_exp()


dc =DataComputer(
    base_path=os.path.join(base_path,"./brown_exp_var_stdsel"),
    sub_exp_folder="brown_exp_var_stdsel",
)
eval_res_ours = dc.load_exp()


dc =DataComputer(
    base_path=os.path.join(base_path,"./brown_est_int_brown_var_sel_std"),
    sub_exp_folder="brown_est_int_brown_var_sel_std",
)
eval_res_ours_intb = dc.load_exp()

def calc_gain(appr: EvalRes, base: EvalRes):

        all_rmse = np.concatenate((appr.mean_rmse,base.mean_rmse))
        sorted_rmse = np.sort(all_rmse, )
        
        max_index = np.where(sorted_rmse >= np.max(appr.mean_rmse))
        min_index = np.where(sorted_rmse <= np.min(base.mean_rmse))
        sorted_rmse = sorted_rmse[min_index[0][-1]:max_index[0][0]]

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

                appr_measures.append(appr.mean_meas[appr_index])
                base_measures.append(base.mean_meas[base_index])
                gains_rmse.append(sorted_rmse[i])

            except IndexError:
                ... # The approach never reached this accuracy, so we can not display it.

        appr_measures = np.asarray(appr_measures)
        base_measures = np.asarray(base_measures)
        gains_rmse = np.asarray(gains_rmse)

        gain = (base_measures - appr_measures) / base_measures
        return (gains_rmse, gain)


def plot_data_gain(ax, data_gain, exp_quant_name="ours", color=None):
    x, y = data_gain
    ax.set_ylim(0,1)
    ax.plot(x, y, label=exp_quant_name, color=color)

    ax.set_xlabel("$RMSE$")
    ax.set_ylabel(r"data gain vs. $CM$")
    

gain1 = calc_gain(eval_res_ours, eval_res_base)
gain2 = calc_gain(eval_res_ours_intb, eval_res_base)

fig, ax = create_fig(subplots=(1,1), width="paper_2c", fraction=1, hfrac=0.3)

plot_data_gain(ax, gain1, exp_quant_name=r"$\mathbf{Brown}\:v_{target}\in[10^{-1},10^{-2,5}]$" , color=appr_style["BR"][0])
plot_data_gain(ax, gain2, exp_quant_name=r"$\mathbf{Int\_Brown}\:v_{target}\in[10^{-1},10^{-2,5}]$", color=appr_style["IB"][0])

#ax.set_xlim(0, np.max(gain1[0]))
ax.set_title("Data Gained by Our Approach")
plot.legend()

save(fig, name="Data Gained by Our Approach", path=save_path)
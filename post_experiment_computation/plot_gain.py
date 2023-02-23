import os
import numpy as np
from utils import plot_rmse_over_mean_meas, save, DataComputer, EvalRes
from matplotlib import pyplot as plot

dc =DataComputer(
    base_path="/home/bela/Cloud/code/Git/alsbts-experiments/eval/fixed_intervals_var_int",
    sub_exp_folder="fixed_intervals_var_int",
)
eval_res_base = dc.load_exp()


dc =DataComputer(
    base_path="/home/bela/Cloud/code/Git/alsbts-experiments/eval/brown_exp_var_stdsel",
    sub_exp_folder="brown_exp_var_stdsel",
)
eval_res_ours = dc.load_exp()

def calc_gain(appr: EvalRes, base: EvalRes):

        all_rmse = np.concatenate((appr.mean_rmse,base.mean_rmse))
        sorted_rmse = np.sort(all_rmse, )

        appr_measures = []
        base_measures = []
        gains_rmse = []
        for i in range(sorted_rmse.shape[0]):
            appr_index = np.where(appr.mean_rmse <= sorted_rmse[i])
            base_index = np.where(base.mean_rmse<= sorted_rmse[i])

            try:
                base_index = base_index[0][0]
            except IndexError:
                base_index = base.mean_rmse.shape[0]

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


def plot_data_gain(data_gain):
    x, y = data_gain
    plot.xlim(np.max(x), np.min(x))
    plot.ylim(0,1)
    plot.plot(x, y, label=f'ours')

    plot.xlabel("rmse-value")
    plot.ylabel("data gain against baseline")
    plot.figlegend()
    

gain = calc_gain(eval_res_ours, eval_res_base)
plot_data_gain(gain)

plot.legend()

save(name="data_gain",path=dc.base_path)
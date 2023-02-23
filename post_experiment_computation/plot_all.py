import os
import numpy as np
from utils import plot_rmse_over_mean_meas, save, DataComputer
from matplotlib import pyplot as plot

dc =DataComputer(
    base_path="/home/bela/Cloud/code/Git/alsbts-experiments/eval/fixed_intervals_var_int",
    sub_exp_folder="fixed_intervals_var_int",
)
eval_res = dc.load_exp()

plot_rmse_over_mean_meas(eval_res, exp_quant_name="fixed_interval")

dc =DataComputer(
    base_path="/home/bela/Cloud/code/Git/alsbts-experiments/eval/behavior_change",
    sub_exp_folder="behavior_change",
)
eval_res = dc.load_exp()
print(eval_res)

plot.scatter(eval_res.mean_meas, eval_res.mean_rmse, label = "every_change")
plot.errorbar(eval_res.mean_meas, eval_res.mean_rmse, yerr=eval_res.std_rmse*1.96, color="red")


dc =DataComputer(
    base_path="/home/bela/Cloud/code/Git/alsbts-experiments/eval/change_exp_var_miss",
    sub_exp_folder="change_exp_var_miss",
)
eval_res = dc.load_exp()

plot_rmse_over_mean_meas(eval_res, exp_quant_name="missed_detection")

dc =DataComputer(
    base_path="/home/bela/Cloud/code/Git/alsbts-experiments/eval/change_offset",
    sub_exp_folder="change_offset",
)
eval_res = dc.load_exp()

plot_rmse_over_mean_meas(eval_res, exp_quant_name="change_offset")

dc =DataComputer(
    base_path="/home/bela/Cloud/code/Git/alsbts-experiments/eval/gauss_rbf",
    sub_exp_folder="gauss_rbf",
)
eval_res = dc.load_exp()

plot.scatter(eval_res.mean_meas, eval_res.mean_rmse, label = "gauss_rbf")
plot.errorbar(eval_res.mean_meas, eval_res.mean_rmse, yerr=eval_res.std_rmse*1.96, color="red")


dc =DataComputer(
    base_path="/home/bela/Cloud/code/Git/alsbts-experiments/eval/rbf_exp_var_stdsel",
    sub_exp_folder="rbf_exp_var_stdsel",
)
eval_res = dc.load_exp()

plot_rmse_over_mean_meas(eval_res, exp_quant_name="rbf_var_std")

dc =DataComputer(
    base_path="/home/bela/Cloud/code/Git/alsbts-experiments/eval/brown_exp_var_stdsel",
    sub_exp_folder="brown_exp_var_stdsel",
)
eval_res = dc.load_exp()

plot_rmse_over_mean_meas(eval_res, exp_quant_name="brown_var_std")


dc = DataComputer(
    base_path="/home/bela/Cloud/code/Git/alsbts-experiments/eval/brown_est_int_brown_var_sel_std",
    sub_exp_folder="brown_est_int_brown_var_sel_std",
)
eval_res = dc.load_exp()

plot_rmse_over_mean_meas(eval_res, exp_quant_name="int_brown_var_sel_std")


dc =DataComputer(
    base_path="/home/bela/Cloud/code/Git/alsbts-experiments/eval/learn_brown_exp_var_stdsel",
    sub_exp_folder="learn_brown_exp_var_stdsel",
)
eval_res = dc.load_exp()


plot_rmse_over_mean_meas(eval_res, exp_quant_name="learn_brown")

plot.legend()

save(name="combined rmse over nr of measurements",path=dc.base_path)
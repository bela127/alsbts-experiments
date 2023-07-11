import os
import numpy as np
from utils import plot_rmse_over_mean_meas, save, DataComputer, base_path, save_path
from matplotlib import pyplot as plt

fig = plt.figure()

dc =DataComputer(
    base_path=os.path.join(base_path,"./fixed_intervals_var_int"),
    sub_exp_folder="fixed_intervals_var_int",
)
eval_res = dc.load_exp()

plot_rmse_over_mean_meas(eval_res, exp_quant_name="fixed_interval")

dc =DataComputer(
    base_path=os.path.join(base_path,"./behavior_change"),
    sub_exp_folder="behavior_change",
)
eval_res = dc.load_exp()
print(eval_res)

plt.scatter(eval_res.mean_meas, eval_res.mean_rmse, label = "every_change")
plt.errorbar(eval_res.mean_meas, eval_res.mean_rmse, yerr=eval_res.std_rmse*1.96, color="red")


dc =DataComputer(
    base_path=os.path.join(base_path,"./change_exp_var_miss"),
    sub_exp_folder="change_exp_var_miss",
)
eval_res = dc.load_exp()

plot_rmse_over_mean_meas(eval_res, exp_quant_name="missed_detection")

dc =DataComputer(
    base_path=os.path.join(base_path,"./change_offset"),
    sub_exp_folder="change_offset",
)
eval_res = dc.load_exp()

plot_rmse_over_mean_meas(eval_res, exp_quant_name="change_offset")

dc =DataComputer(
    base_path=os.path.join(base_path,"./gauss_rbf"),
    sub_exp_folder="gauss_rbf",
)
eval_res = dc.load_exp()

plt.scatter(eval_res.mean_meas, eval_res.mean_rmse, label = "gauss_rbf")
plt.errorbar(eval_res.mean_meas, eval_res.mean_rmse, yerr=eval_res.std_rmse*1.96, color="red")


dc =DataComputer(
    base_path=os.path.join(base_path,"./rbf_exp_var_stdsel"),
    sub_exp_folder="rbf_exp_var_stdsel",
)
eval_res = dc.load_exp()

plot_rmse_over_mean_meas(eval_res, exp_quant_name="rbf_var_std")

dc =DataComputer(
    base_path=os.path.join(base_path,"./brown_exp_var_stdsel"),
    sub_exp_folder="brown_exp_var_stdsel",
)
eval_res = dc.load_exp()

plot_rmse_over_mean_meas(eval_res, exp_quant_name="brown_var_std")


dc = DataComputer(
    base_path=os.path.join(base_path,"./brown_est_int_brown_var_sel_std"),
    sub_exp_folder="brown_est_int_brown_var_sel_std",
)
eval_res = dc.load_exp()

plot_rmse_over_mean_meas(eval_res, exp_quant_name="int_brown_var_sel_std")


dc =DataComputer(
    base_path=os.path.join(base_path,"./learn_brown_exp_var_stdsel"),
    sub_exp_folder="learn_brown_exp_var_stdsel",
)
eval_res = dc.load_exp()


plot_rmse_over_mean_meas(eval_res, exp_quant_name="learn_brown")

plt.legend()

save(fig= fig, name="combined rmse over nr of measurements",path=save_path)
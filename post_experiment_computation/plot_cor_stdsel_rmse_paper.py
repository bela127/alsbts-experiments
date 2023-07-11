import os
import numpy as np
from utils import plot_rmse_over_stdsel, plot_rmse_over_mean_meas, DataComputer, save,  base_path, save_path, create_fig, appr_style, linestyle
from matplotlib import pyplot as plt

dc =DataComputer(
    base_path=os.path.join(base_path,"./brown_exp_var_stdsel"),
    sub_exp_folder="brown_exp_var_stdsel",
)
eval_res = dc.load_exp()

dc =DataComputer(
    base_path=os.path.join(base_path,"./brown_est_int_brown_var_sel_std"),
    sub_exp_folder="brown_est_int_brown_var_sel_std",
)
eval_res2 = dc.load_exp()

eval_res.exp_quantity = np.sqrt(eval_res.exp_quantity) / 1.96
eval_res2.exp_quantity = np.sqrt(eval_res2.exp_quantity) / 1.96

fig, ax = create_fig(subplots=(1,1), width="paper", fraction=1/2)

plot_rmse_over_stdsel(ax, eval_res, exp_quant_name=r"$\mathbf{Brown}\:v_{target}\in[10^{-1},10^{-2,5}]$", color=appr_style["BR"][0])
plot_rmse_over_stdsel(ax, eval_res2, exp_quant_name=r"$\mathbf{Int\_Brown}\:v_{target}\in[10^{-1},10^{-2,5}]$", color=appr_style["IB"][0])
equal_line = np.arange(0, 0.8, 0.1)
plt.plot(equal_line, equal_line,label="Ideal Relation", color="black", linestyle=linestyle["dotted"])

ax.set_title("Correlation Between Threshold and RMSE")
plt.xlim(0.0,0.6)
plt.ylim(0.0,0.9)
plt.legend()
save(fig, name="Correlation Between Threshold and RMSE",path=save_path)

fig, ax = create_fig(subplots=(1,1), width="paper", fraction=1/2)

plot_rmse_over_mean_meas(ax, eval_res)
save(fig, name="rmse over nr of measurements",path=save_path)
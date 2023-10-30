import os
import numpy as np
from utils import scatter_rmse_over_meas, save, ScatterComputer, base_path, save_path, create_fig, appr_style
from matplotlib import pyplot as plt

fig, ax = create_fig(subplots=(1,1), width="paper_2c", fraction=1)

dc =ScatterComputer(
    base_path=os.path.join(base_path,"./fixed_intervals_var_int"),
    sub_exp_folder="fixed_intervals_var_int",
)
base_eval_res = dc.load_exp()

scatter_rmse_over_meas(ax, base_eval_res, exp_quant_name=r"$\textit{CM}\:\mathbf{\delta t_{meas}\in[1,20]}$", color=appr_style["CM"][0], marker=appr_style["CM"][2])


dc =ScatterComputer(
    base_path=os.path.join(base_path,"./change_exp_var_wrong"),
    sub_exp_folder="change_exp_var_wrong",
)
eval_res = dc.load_exp()

scatter_rmse_over_meas(ax, eval_res, exp_quant_name=r"$\textit{CE}\:\mathbf{p_{wrong}\in[0, 0.10]}$", color=appr_style["CEw"][0], marker=appr_style["CEw"][2])


dc =ScatterComputer(
    base_path=os.path.join(base_path,"./change_exp_var_miss"),
    sub_exp_folder="change_exp_var_miss",
)
eval_res = dc.load_exp()

scatter_rmse_over_meas(ax, eval_res, exp_quant_name=r"$\textit{CE}\:\mathbf{p_{miss}\in[0, 0.80]}$", color=appr_style["CEm"][0], marker=appr_style["CEm"][2])

dc =ScatterComputer(
    base_path=os.path.join(base_path,"./change_exp_var_offset"),
    sub_exp_folder="change_exp_var_offset",
)
eval_res = dc.load_exp()

scatter_rmse_over_meas(ax, eval_res, exp_quant_name=r"$\textit{CE}\:\mathbf{std_{offset}\in[0, 15]}$", color=appr_style["CEo"][0], marker=appr_style["CEo"][2])


dc =ScatterComputer(
    base_path=os.path.join(base_path,"./gauss_rbf"),
    sub_exp_folder="gauss_rbf",
)
eval_res = dc.load_exp()

#y = eval_res.mean_rmse = BIG
scatter_rmse_over_meas(ax, eval_res, exp_quant_name= r"$\textit{CAL}\:\mathbf{Nr_{meas.}=100}$", color=appr_style["CALm"][0])

dc =ScatterComputer(
    base_path=os.path.join(base_path,"./rbf_exp_var_stdsel"),
    sub_exp_folder="rbf_exp_var_stdsel",
)
eval_res = dc.load_exp()

scatter_rmse_over_meas(ax, eval_res, exp_quant_name=r"$\textit{CAL}\:\mathbf{v_{target}\in[10^{-2,5}, 10^{-1}]}$", color=appr_style["CAL"][0], marker=appr_style["CAL"][2])

dc =ScatterComputer(
    base_path=os.path.join(base_path,"./brown_exp_var_stdsel"),
    sub_exp_folder="brown_exp_var_stdsel",
)
our_eval_res = dc.load_exp()

scatter_rmse_over_meas(ax, our_eval_res, exp_quant_name=r"$\textbf{DEAL}\:\mathbf{v_{target}\in[10^{-2,5}, 10^{-1}]}$", color=appr_style["BR"][0], marker=appr_style["BR"][2])

dc =ScatterComputer(
    base_path=os.path.join(base_path,"./behavior_change"),
    sub_exp_folder="behavior_change",
)
eval_res = dc.load_exp()

scatter_rmse_over_meas(ax, eval_res, exp_quant_name=r"$\textit{CI}$", color=appr_style["CI"][0])

ax.set_title("Measurements vs RMSE for all Configurations")
ax.set_xlim(5,8000)
ax.set_ylim(0,3)
ax.set_xscale("log")

plt.legend(numpoints=1)#fontsize="5", borderpad=0.1, loc='upper right')#, ncol=5, bbox_transform=fig.transFigure)
#ax.legend(loc='best', bbox_to_anchor=(0.45, 0.2, 0.0, 0.0),frameon=False)#fontsize="5", borderpad=0.1, loc='upper right')#, ncol=5, bbox_transform=fig.transFigure)
#ax.legend(numpoints=1, bbox_to_anchor=(0.2, 0.1))

save(fig= fig, name="Required Measurements for Convergence Scatter",path=save_path)



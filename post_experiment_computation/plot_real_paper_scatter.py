import os
import numpy as np
from utils import scatter_rmse_over_meas, save, ScatterComputer, base_path, save_path, create_fig, appr_style
from matplotlib import pyplot as plt

fig, ax = create_fig(subplots=(1,1), width="paper_2c", fraction=1)

dc =ScatterComputer(
    base_path=os.path.join(base_path,"./data_seismic_fixed_intervals_var_int"),
    sub_exp_folder="data_seismic_fixed_intervals_var_int",
)
base_eval_res = dc.load_exp()

scatter_rmse_over_meas(ax, base_eval_res, exp_quant_name=r"$\textit{CM}\:\mathbf{\delta t_{meas}\in[1,20]}$", color=appr_style["CM"][0], marker=appr_style["CM"][2])

dc =ScatterComputer(
    base_path=os.path.join(base_path,"./data_seismic_brown_exp_var_stdsel"),
    sub_exp_folder="data_seismic_brown_exp_var_stdsel",
)
our_eval_res = dc.load_exp()

scatter_rmse_over_meas(ax, our_eval_res, exp_quant_name=r"$\textbf{DEAL}\:\mathbf{v_{target}\in[10^{-2,5}, 10^{-1}]}$", color=appr_style["BR"][0], marker=appr_style["BR"][2])

ax.set_title("Measurements vs RMSE for all Configurations")
ax.set_xlim(5,8000)
ax.set_ylim(0,3)
ax.set_xscale("log")

plt.legend(numpoints=1)#fontsize="5", borderpad=0.1, loc='upper right')#, ncol=5, bbox_transform=fig.transFigure)
#ax.legend(loc='best', bbox_to_anchor=(0.45, 0.2, 0.0, 0.0),frameon=False)#fontsize="5", borderpad=0.1, loc='upper right')#, ncol=5, bbox_transform=fig.transFigure)
#ax.legend(numpoints=1, bbox_to_anchor=(0.2, 0.1))

save(fig= fig, name="Required Measurements for Convergence Seismic Data Scatter",path=save_path)



import os
import numpy as np
from utils import plot_rmse_over_exp_quant, plot_rmse_over_mean_meas, DataComputer, save, plot_pred_vs_gt, RunRes
from matplotlib import pyplot as plot

class RunComputer(DataComputer):
    
    def comp_run(self, run_res: RunRes):
        plot_pred_vs_gt(run_res)
        save(path=run_res.run_path,name=f"{run_res.run_name}_pred_vs_gt")
        return super().comp_run(run_res)

dc = RunComputer(
    base_path="/home/bela/Cloud/code/Git/alsbts-experiments/eval/rbf_stdsel_exp_var_brown",
    sub_exp_folder="rbf_stdsel_exp_var_brown",
)
eval_res = dc.load_exp()
print(eval_res)
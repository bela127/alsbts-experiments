import os
import numpy as np
from utils import plot_rmse_over_exp_quant, plot_rmse_over_mean_meas, DataComputer, save, plot_pred_vs_gt, RunRes
from matplotlib import pyplot as plot

class RunComputer(DataComputer):
    
    def comp_run(self, run_res: RunRes):
        run_res.estimation = run_res.estimation / 5
        plot_pred_vs_gt(run_res)
        save(path=run_res.run_path,name=f"{run_res.run_name}_pred_vs_gt")
        return super().comp_run(run_res)

dc = RunComputer(
    base_path="/home/bela/Cloud/code/Git/alsbts-experiments/eval/int_brown_exp_var_stdsel",
    sub_exp_folder="int_brown_exp_var_stdsel",
)
eval_res = dc.load_exp()
print(eval_res)
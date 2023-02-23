import os
import numpy as np
from utils import plot_rmse_over_exp_quant, plot_rmse_over_mean_meas, DataComputer, save, plot_pred_vs_gt, RunRes
from matplotlib import pyplot as plot


for i in np.arange(0, 5, 1):

    class RunComputer(DataComputer):
    
        def comp_run(self, run_res: RunRes):
            run_res.estimation = run_res.estimation / (i+1)
            plot_pred_vs_gt(run_res)
            save(path=run_res.run_path,name=f"{run_res.run_name}_pred_vs_gt")
            return super().comp_run(run_res)

    dc = RunComputer(
        base_path=f"/home/bela/Cloud/code/Git/alsbts-experiments/eval/int_brown_est_brown_var_int{i}_stdsel",
        sub_exp_folder=f"int_brown_est_brown_var_int{i}_stdsel",
    )
    eval_res = dc.load_exp()


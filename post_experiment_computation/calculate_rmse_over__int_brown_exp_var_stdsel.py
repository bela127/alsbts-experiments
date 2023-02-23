import os
import numpy as np
from utils import plot_rmse_over_exp_quant, plot_rmse_over_mean_meas, DataComputer, save, RunRes
from matplotlib import pyplot as plot

class RunComputer(DataComputer):
    
    def comp_run(self, run_res: RunRes):
        run_res.estimation = run_res.estimation / 6
        return super().comp_run(run_res)

dc =RunComputer(
    base_path="/home/bela/Cloud/code/Git/alsbts-experiments/eval/int_brown_exp_var_stdsel",
    sub_exp_folder="int_brown_exp_var_stdsel",
)
eval_res = dc.load_exp()

plot_rmse_over_exp_quant(eval_res, exp_quant_name="sel_var_int_brown")
save(name="Impact of selection variance on rmse",path=dc.base_path)

plot_rmse_over_mean_meas(eval_res)
save(name="rmse over nr of measurements",path=dc.base_path)
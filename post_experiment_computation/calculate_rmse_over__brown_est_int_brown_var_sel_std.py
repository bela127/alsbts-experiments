import os
import numpy as np
from utils import plot_rmse_over_exp_quant, plot_rmse_over_mean_meas, DataComputer, save, RunRes
from matplotlib import pyplot as plot

dc = DataComputer(
    base_path="/home/bela/Cloud/code/Git/alsbts-experiments/eval/brown_est_int_brown_var_sel_std",
    sub_exp_folder="brown_est_int_brown_var_sel_std",
)
eval_res = dc.load_exp()

plot_rmse_over_exp_quant(eval_res, exp_quant_name="sel_var_int_brown")
save(name="Impact of selection variance on rmse",path=dc.base_path)

plot_rmse_over_mean_meas(eval_res)
save(name="rmse over nr of measurements",path=dc.base_path)
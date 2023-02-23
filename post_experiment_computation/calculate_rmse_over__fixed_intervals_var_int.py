import os
import numpy as np
from utils import plot_rmse_over_exp_quant, plot_rmse_over_mean_meas, DataComputer, save
from matplotlib import pyplot as plot

dc =DataComputer(
    base_path="/home/bela/Cloud/code/Git/alsbts-experiments/eval/fixed_intervals_var_int",
    sub_exp_folder="fixed_intervals_var_int",
)
eval_res = dc.load_exp()

plot_rmse_over_exp_quant(eval_res)
save(name="Impact of measurement frequency on rmse",path=dc.base_path)

plot_rmse_over_mean_meas(eval_res)
save(name="rmse over nr of measurements",path=dc.base_path)
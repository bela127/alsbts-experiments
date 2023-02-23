import os
import numpy as np
from utils import plot_rmse_over_exp_quant, plot_rmse_over_mean_meas, DataComputer, save
from matplotlib import pyplot as plot

dc =DataComputer(
    base_path="/home/bela/Cloud/code/Git/alsbts-experiments/eval/learn_brown_exp_var_stdsel",
    sub_exp_folder="learn_brown_exp_var_stdsel",
)
eval_res = dc.load_exp()

plot_rmse_over_exp_quant(eval_res, exp_quant_name="missed detections")
save(name="Impact of nr of missed detections on rmse",path=dc.base_path)

plot_rmse_over_mean_meas(eval_res)
save(name="rmse over nr of measurements",path=dc.base_path)
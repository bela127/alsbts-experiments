import os
import numpy as np
from utils import plot_rmse_over_exp_quant, plot_rmse_over_mean_meas, DataComputer, save
from matplotlib import pyplot as plot

dc =DataComputer(
    base_path="/home/bela/Cloud/code/Git/alsbts-experiments/eval/rbf_stdsel_exp_var_brown",
    sub_exp_folder="rbf_stdsel_exp_var_brown",
)
eval_res = dc.load_exp()

plot_rmse_over_exp_quant(eval_res, exp_quant_name="missed detections")
save(name="Impact of nr of missed detections on rmse",path=dc.base_path)

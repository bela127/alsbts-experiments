import os
import numpy as np
from utils import plot_rmse_over_exp_quant, plot_rmse_over_mean_meas, DataComputer, save
from matplotlib import pyplot as plot

dc =DataComputer(
    base_path="/home/bela/Cloud/code/Git/alsbts-experiments/eval/brown_exp_var_stdsel",
    sub_exp_folder="brown_exp_var_stdsel",
)
eval_res = dc.load_exp()

eval_res.exp_quantity = np.sqrt(eval_res.exp_quantity) / 1.96

plot_rmse_over_exp_quant(eval_res, exp_quant_name="allowed std")
plot.xlim(0.025,0.2)
plot.ylim(0.025,0.2)
save(name="Allowed std vs rmse",path=dc.base_path)

plot_rmse_over_mean_meas(eval_res)
save(name="rmse over nr of measurements",path=dc.base_path)
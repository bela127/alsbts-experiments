from dataclasses import dataclass
import os
import numpy as np
from utils import MeasComputer, save, plot_meas_over_time
from matplotlib import pyplot as plot


dc =MeasComputer(
    base_path="/home/bela/Cloud/code/Git/alsbts-experiments/eval/rbf_exp_var_stdsel",
    sub_exp_folder="rbf_exp_var_stdsel",
)
meas_res = dc.load_exp()


plot_meas_over_time(meas_res, exp_quant_name="selvar")

plot.xlim(0,1000)
plot.ylim(0,300)
plot.legend()

save(name="measurements over time",path=dc.base_path)

plot_meas_over_time(meas_res, exp_quant_name="selvar")

plot.xlim(0,100)
plot.ylim(0,15)
plot.legend()

save(name="init phase",path=dc.base_path)
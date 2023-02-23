from dataclasses import dataclass
import os
import numpy as np
from utils import MeasComputer, save, plot_meas_over_time, plot_sum_meas_over_time, plot_meas_per_step_vs_exp_quant
from matplotlib import pyplot as plot


dc =MeasComputer(
    base_path="/home/bela/Cloud/code/Git/alsbts-experiments/eval/brown_exp_var_stdsel",
    sub_exp_folder="brown_exp_var_stdsel",
)
meas_res = dc.load_exp()


plot_meas_over_time(meas_res, exp_quant_name="selvar")

plot.xlim(0,1000)
plot.ylim(0,1)
plot.legend()

save(name="measurements over time",path=dc.base_path)

plot_meas_over_time(meas_res, exp_quant_name="selvar")

plot.xlim(0,50)
plot.ylim(0,1.1)
plot.legend()

save(name="init phase",path=dc.base_path)


plot_sum_meas_over_time(meas_res, exp_quant_name="selvar")

plot.xlim(0,1000)
plot.ylim(0,350)
plot.legend()

save(name="sum measurements over time",path=dc.base_path)

plot_sum_meas_over_time(meas_res, exp_quant_name="selvar")

plot.xlim(0,15)
plot.ylim(0,8)
plot.legend()

save(name="sum init phase",path=dc.base_path)

plot_meas_per_step_vs_exp_quant(meas_res, exp_quant_name="selvar")
plot.ylim(0,0.3)
save(name="meas_over_selvar",path=dc.base_path)
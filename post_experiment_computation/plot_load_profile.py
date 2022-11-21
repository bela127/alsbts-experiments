from matplotlib import pyplot as plot
import numpy as np
import os
from utils import save, norm_time, plot_pred_vs_gt, load_data

path = "/home/bela/Cloud/code/Git/alvsts-experiments/eval/estimation_brown/log/exp_0"
file = "all_data_source_data.npy"

sim_res = load_data(path, skip_points= int(1.5/0.2))

sim_frame = norm_time(sim_res.time)
plot.xlabel("sim period's")
plot.ylabel("voltage sensitivity")
plot.plot(sim_frame, sim_res.gt_vs)
save("vs_profile", path=path)

x= 10
plot.xlim(0, 2.5)
plot.xlabel("time [s]")
plot.ylabel("voltage [per unit]")
plot.plot(sim_res.time-x, sim_res.voltage)
save("v_disturbance", path=path)

plot.xlabel("time [s]")
plot.ylabel("voltage [per unit]")
plot.plot(sim_res.time, sim_res.voltage)
save("v", path=path)

plot.xlabel("time [s]")
plot.ylabel("relative voltage [per unit]")
plot.plot(sim_res.time, sim_res.v_change)
save("v_change", path=path)

plot.plot(sim_res.time, sim_res.active_power)
save("active_power", path=path)

plot.plot(sim_res.time, sim_res.reactive_power)
save("reactive_power", path=path)

plot.plot(sim_res.time, sim_res.vs_estimation)
save("vs_estimation", path=path)

plot.plot(sim_res.time, sim_res.measurement_trigger)
save("measurement_trigger", path=path)

plot.plot(sim_res.time, sim_res.rvs)
save("rvs", path=path)

plot.plot(sim_res.time, sim_res.change)
save("change_point", path=path)

plot.plot(sim_res.time, sim_res.measurement_active)
save("measurement_active", path=path)

plot.plot(sim_res.time, sim_res.vs_measurement)
save("vs_measurement", path=path)


plot.xlabel("time")
plot.ylabel("voltage sensitivity")
plot.ylim(-0.5,2.5)
plot.plot(sim_res.time, sim_res.gt_vs, label = "GT_VS")
plot.plot(sim_res.time, sim_res.vs_estimation, label = "VS_estimate")
plot.legend()
save("vs_gt_vs_estimation", path=path)


plot.xlabel("time")
plot.ylabel("voltage sensitivity")
plot.ylim(-0.5,2.5)
plot.plot(sim_res.time, sim_res.gt_vs, label = "GT_VS")
plot.plot(sim_res.time, sim_res.vs_measurement, label = "vs_measurement")
plot.legend()
save("vs_gt_vs_measure", path=path)
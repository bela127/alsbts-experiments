from matplotlib import pyplot as plot
import numpy as np
import os
from utils import save, norm_time

sim_time = 600
dist = 120

path = "/home/bela/Cloud/code/Git/alvsts-experiments/eval/change_offset/eval/change_offset0/log/exp_0"
file = "all_data_source_data.npy"

data = np.load(os.path.join(path,file))

time, voltage, new_v, active_power, reactive_power, gt_vs, estimation, measurement, rvs, change , measurement_active, vs_measurement= np.transpose(data)


sim_frame = norm_time(time)
plot.xlabel("sim period's")
plot.ylabel("voltage sensitivity")
plot.plot(sim_frame, gt_vs)
save("vs_profile", path=path)

x= 10
plot.xlim(0, 2.5)
plot.xlabel("time [s]")
plot.ylabel("voltage [per unit]")
plot.plot(time-x, voltage)

save("v_disturbance", path=path)

plot.plot(time, active_power)
save("active_power", path=path)

plot.plot(time, reactive_power)
save("reactive_power", path=path)

plot.plot(time, estimation)
save("vs_estimation", path=path)

plot.plot(time, measurement)
save("measurement_trigger", path=path)

plot.plot(time, rvs)
save("rvs", path=path)

plot.plot(time, change)
save("change_point", path=path)

plot.plot(time, measurement_active)
save("measurement_active", path=path)

plot.plot(time, vs_measurement)
save("vs_measurement", path=path)

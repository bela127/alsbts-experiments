from cProfile import label
from matplotlib import pyplot as plot
import numpy as np
import os
from utils import save, norm_time, plot_pred_vs_gt, load_data

path = "/home/bela/Cloud/code/Git/alvsts-experiments/eval/fixed_interval/log/exp_0"
file = "all_data_source_data.npy"

measure_int = 10
measure_time = 2.5
total_time=600
step_time = 0.2
start_time = 500

fig_path = os.path.join(path, "steps")
os.makedirs(fig_path,exist_ok=True)



sim_res = load_data(path, skip_points= int(start_time/0.2))


dp=(sim_res.active_power[:-1]-sim_res.active_power[1:])/sim_res.active_power[1:]
dv=(sim_res.voltage[:-1]-sim_res.voltage[1:])/sim_res.voltage[1:]
kp=dp/dv
#kp = np.nan_to_num(kp, neginf=0, posinf=0)

window_size = int(measure_int/step_time)
for i in range(int((total_time-start_time)/measure_int)):
    start = window_size * i
    end = start + int(measure_time/step_time)

    plot.plot(sim_res.time[start:end], sim_res.active_power[start:end])
    save(f"active_power_{i}", path= fig_path)

    plot.plot(sim_res.time[start:end], sim_res.voltage[start:end])
    save(f"voltage_{i}", path= fig_path)

    plot.plot(sim_res.time[start:end], sim_res.gt_vs[start:end])
    plot.plot(sim_res.time[start:end], kp[start:end])
    save(f"gt_vs_vs_kp_{i}", path= fig_path)

    plot.plot(sim_res.time[start:end], sim_res.active_power[start:end]/50000*100-97,label="p")
    plot.plot(sim_res.time[start:end], sim_res.voltage[start:end]*100-97,label="v")
    plot.plot(sim_res.time[start:end], sim_res.gt_vs[start:end],label="vs_gt")
    plot.plot(sim_res.time[start:end], kp[start:end],label="vs")
    plot.legend()
    save(f"gt_vs_vs_kp_v_p_{i}", path= fig_path)


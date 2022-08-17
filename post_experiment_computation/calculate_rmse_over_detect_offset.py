import os
import numpy as np
from utils import calc_measurements, check_sim_data_ok, load_data, calc_rmse, SimRes, norm_measurements, plot_pred_vs_gt, save, walk_dirs, walk_files
from matplotlib import pyplot as plot

path = "/home/bela/Cloud/code/Git/alvsts-experiments/eval/change_offset/eval"

def rmse_worker(dirpath, file):
    sim_res = load_data(skip_points = 0, path=dirpath)

    if not check_sim_data_ok(sim_res):
        print("Corrupt Sim Data: ", dirpath)
        print("Min = ", np.min(sim_res.vs_measurement))
        #plot_pred_vs_gt(sim_res)
        sim_res.vs_measurement[sim_res.vs_measurement < 0] = 0
        sim_res.vs_estimation[sim_res.vs_estimation < 0] = 0

    rmse = calc_rmse(sim_res)
    nr_measurements = calc_measurements(sim_res)
    return rmse, nr_measurements


def dir_worker(dirpath, dir: str):
    if dir.startswith("change_offset"):
        change_offset_str = dir.lstrip("change_offset")
        change_offset = float(change_offset_str)

        rmse_meas = [rmse_meas for rmse_meas in walk_files(path=os.path.join(dirpath, dir), worker=rmse_worker)]
        rmse , meas = np.asarray(rmse_meas).T

        mean_rmse = np.mean(rmse)
        std_rmse = np.std(rmse)

        mean_meas = np.mean(meas)
        std_meas = np.std(meas)

        change_rmse_std_meas_std = (change_offset, mean_rmse, std_rmse, mean_meas, std_meas)
        return change_rmse_std_meas_std


change_rmse_std_meas_std = [i_r_v for i_r_v in walk_dirs(path=path, worker=dir_worker) if i_r_v is not None]

def plot_rmse_over_meas():
    chan_rmse_std_meas_std = np.asarray(change_rmse_std_meas_std)
    chan_rmse_std_meas_std = chan_rmse_std_meas_std[chan_rmse_std_meas_std[:, 3].argsort()]
    change_offset, mean_rmse, std_rmse, mean_meas, std_meas = chan_rmse_std_meas_std.T
    
    rel_meas = norm_measurements(mean_meas)
    plot.plot(rel_meas, mean_rmse, color="black", label="Mean")
    plot.fill_between(
        rel_meas,
        mean_rmse - std_rmse,
        mean_rmse + std_rmse,
        alpha=0.1,
        color="black",
        label=r"$\pm$ 1 std. dev.",
    )
    plot.xlabel("measurement per load change")
    plot.ylabel("rmse")

plot_rmse_over_meas()
save(name="Impact of measurement frequency on rmse",path=path)


def plot_rmse_over_offset():
    chan_rmse_std_meas_std = np.asarray(change_rmse_std_meas_std)
    chan_rmse_std_meas_std = chan_rmse_std_meas_std[chan_rmse_std_meas_std[:, 0].argsort()]
    change_offset, mean_rmse, std_rmse, mean_meas, std_meas = chan_rmse_std_meas_std.T
    
    rel_meas = norm_measurements(mean_meas)
    plot.plot(change_offset, mean_rmse, color="black", label="Mean")
    plot.fill_between(
        change_offset,
        mean_rmse - std_rmse,
        mean_rmse + std_rmse,
        alpha=0.1,
        color="black",
        label=r"$\pm$ 1 std. dev.",
    )
    plot.xlabel("change offset in time steps")
    plot.ylabel("rmse")

plot_rmse_over_offset()
save(name="Impact of change_offset on rmse",path=path)
import os
import numpy as np
from utils import calc_measurements, check_sim_data_ok, load_data, calc_rmse, SimRes, norm_measurements, plot_pred_vs_gt, save, walk_dirs, walk_files
from matplotlib import pyplot as plot

path = "/home/bela/Cloud/code/Git/alvsts-experiments/eval/fixed_interval_dif_ints/eval/"

def rmse_worker(dirpath, file):
    sim_res = load_data(skip_points = 0, path=dirpath)

    if not check_sim_data_ok(sim_res):
        print("Corrupt Sim Data: ", dirpath)
        print("Min = ", np.min(sim_res.vs_measurement))
        plot_pred_vs_gt(sim_res)

    rmse = calc_rmse(sim_res)
    nr_measurements = calc_measurements(sim_res)
    print(rmse, nr_measurements)
    return rmse, nr_measurements


def dir_worker(dirpath, dir: str):
    if dir.startswith("fixed_interval_t"):
        measure_time_str = dir.lstrip("fixed_interval_t")
        measure_time = float(measure_time_str)

        print(measure_time)


        rmse_meas = [rmse_meas for rmse_meas in walk_files(path=os.path.join(dirpath, dir), worker=rmse_worker)]
        rmse , meas = np.asarray(rmse_meas).T

        mean_rmse = np.mean(rmse)
        std_rmse = np.std(rmse)

        mean_meas = np.mean(meas)
        std_meas = np.std(meas)

        rmse_std_meas_std = (measure_time, mean_rmse, std_rmse, mean_meas, std_meas)
        return rmse_std_meas_std


rmse_std_meas_std = [i_r_v for i_r_v in walk_dirs(path=path, worker=dir_worker) if i_r_v is not None]

def plot_rmse_over_inter():
    inter_rmse_std_meas_std = np.asarray(rmse_std_meas_std)
    inter_rmse_std_meas_std = inter_rmse_std_meas_std[inter_rmse_std_meas_std[:, 0].argsort()]
    inter, mean_rmse, std_rmse, mean_meas, std_meas = inter_rmse_std_meas_std.T
    
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

plot_rmse_over_inter()
save(name="Impact of measurement frequency on rmse",path=path)
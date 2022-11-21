import os
import numpy as np
from utils import calc_measurements, check_sim_data_ok, load_data, calc_rmse, SimRes, norm_measurements, plot_pred_vs_gt, save, walk_dirs, walk_files
from matplotlib import pyplot as plot

path = "/home/bela/Cloud/code/Git/alvsts-experiments/eval/missed_detection/eval"

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
    if dir.startswith("missed_detection"):
        wrong_detection_str = dir.lstrip("missed_detection")
        wrong_detection = float(wrong_detection_str)

        rmse_meas = [rmse_meas for rmse_meas in walk_files(path=os.path.join(dirpath, dir), worker=rmse_worker)]
        rmse , meas = np.asarray(rmse_meas).T

        mean_rmse = np.mean(rmse)
        std_rmse = np.std(rmse)

        mean_meas = np.mean(meas)
        std_meas = np.std(meas)

        wrong_rmse_std_meas_std = (wrong_detection, mean_rmse, std_rmse, mean_meas, std_meas)
        return wrong_rmse_std_meas_std


wrong_rmse_std_meas_std = [i_r_v for i_r_v in walk_dirs(path=path, worker=dir_worker) if i_r_v is not None]

def plot_rmse_over_meas():
    wro_rmse_std_meas_std = np.asarray(wrong_rmse_std_meas_std)
    wro_rmse_std_meas_std = wro_rmse_std_meas_std[wro_rmse_std_meas_std[:, 3].argsort()]
    wrong_detection, mean_rmse, std_rmse, mean_meas, std_meas = wro_rmse_std_meas_std.T
    
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


def plot_rmse_over_wrong():
    wro_rmse_std_meas_std = np.asarray(wrong_rmse_std_meas_std)
    wro_rmse_std_meas_std = wro_rmse_std_meas_std[wro_rmse_std_meas_std[:, 0].argsort()]
    wrong_detection, mean_rmse, std_rmse, mean_meas, std_meas = wro_rmse_std_meas_std.T
    
    plot.plot(wrong_detection, mean_rmse, color="black", label="Mean")
    plot.fill_between(
        wrong_detection,
        mean_rmse - std_rmse,
        mean_rmse + std_rmse,
        alpha=0.1,
        color="black",
        label=r"$\pm$ 1 std. dev.",
    )
    plot.xlabel("fraction of missed detections")
    plot.ylabel("rmse")

plot_rmse_over_wrong()
save(name="Impact of missed detection ratio on rmse", path=path)


def plot_meas_over_wrong():
    wro_rmse_std_meas_std = np.asarray(wrong_rmse_std_meas_std)
    wro_rmse_std_meas_std = wro_rmse_std_meas_std[wro_rmse_std_meas_std[:, 0].argsort()]
    wrong_detection, mean_rmse, std_rmse, mean_meas, std_meas = wro_rmse_std_meas_std.T
    
    rel_meas = norm_measurements(mean_meas)
    rel_meas_std = norm_measurements(std_meas)
    plot.plot(wrong_detection, rel_meas, color="black", label="Mean")
    plot.fill_between(
        wrong_detection,
        rel_meas - rel_meas_std,
        rel_meas + rel_meas_std,
        alpha=0.1,
        color="black",
        label=r"$\pm$ 1 std. dev.",
    )
    plot.xlabel("fraction of missed detections")
    plot.ylabel("number of measurements per load change")

plot_meas_over_wrong()
save(name="Impact of missed detection ratio on measurement count",path=path)


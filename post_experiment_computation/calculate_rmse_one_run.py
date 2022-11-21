import os
import numpy as np
from utils import calc_mean_meas, calc_mean_rmse, calc_measurements, check_sim_data_ok, load_data, calc_rmse, SimRes, norm_measurements, plot_pred_vs_gt, save, walk_dirs, walk_files
from matplotlib import pyplot as plot

path = "/home/bela/Cloud/code/Git/alvsts-experiments/eval/estimation_brown/log"

def rmse_worker(dirpath, file):
    sim_res = load_data(skip_points = int(299.5/0.2), path=dirpath)

    if not check_sim_data_ok(sim_res):
        print("Corrupt Sim Data: ", dirpath)
        print("Min = ", np.min(sim_res.vs_measurement))
        plot_pred_vs_gt(sim_res)

    rmse = calc_rmse(sim_res)
    nr_measurements = calc_measurements(sim_res)
    return rmse, nr_measurements


def calc_mean_rmse_meas(path):

    rmse_meas = [rmse_meas for rmse_meas in walk_files(path=path, worker=rmse_worker)]
    rmse , meas = np.asarray(rmse_meas).T

    mean_rmse, std_rmse = calc_mean_rmse(rmse)

    mean_meas, std_meas = calc_mean_meas(meas)

    rmse_std_meas_std = (mean_rmse, std_rmse, mean_meas, std_meas)

    return rmse_std_meas_std, rmse_meas


rmse_std_meas_std, rmse_meas = calc_mean_rmse_meas(path=path)

print(rmse_std_meas_std)
for rmse, meas in rmse_meas:
    print(rmse, meas)
import os
from click import style
import numpy as np
from utils import  exp_dir_worker,  norm_measurements, save, walk_dirs
from matplotlib import pyplot as plot

path = "/home/bela/Cloud/code/Git/alvsts-experiments/eval/fixed_interval_dif_ints/eval/"
int_rmse_std_meas_std = [i_r_v for i_r_v in walk_dirs(path=path, worker=exp_dir_worker("fixed_interval_t")) if i_r_v is not None]


path = "/home/bela/Cloud/code/Git/alvsts-experiments/eval/missed_detection/eval"
miss_rmse_std_meas_std = [i_r_v for i_r_v in walk_dirs(path=path, worker=exp_dir_worker("missed_detection")) if i_r_v is not None]



def plot_rmse_over_wrong_vs_baseline():
    wro_rmse_std_meas_std = np.asarray(miss_rmse_std_meas_std)
    wro_rmse_std_meas_std = wro_rmse_std_meas_std[wro_rmse_std_meas_std[:, 3].argsort()]
    wrong_detection, mean_rmse, std_rmse, mean_meas, std_meas = wro_rmse_std_meas_std.T
    
    fig = plot.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()

    handel1_1, = ax1.plot(wrong_detection, mean_rmse, color="black", label="Mean")
    handel1_2 = ax1.fill_between(
        wrong_detection,
        mean_rmse - std_rmse,
        mean_rmse + std_rmse,
        alpha=0.1,
        color="black",
        label=r"$\pm$ 1 std. dev.",
    )
    ax1.set_xlim(0, 0.5)
    ax1.set_xlabel("fraction of missed detections")
    ax1.set_ylabel("rmse")

    inter_rmse_std_meas_std = np.asarray(int_rmse_std_meas_std)
    inter_rmse_std_meas_std = inter_rmse_std_meas_std[inter_rmse_std_meas_std[:, 0].argsort()]
    inter, mean_rmse, std_rmse, mean_meas, std_meas = inter_rmse_std_meas_std.T
    
    rel_meas = norm_measurements(mean_meas)
    handel2, = ax2.plot(rel_meas, mean_rmse, linestyle='--', color="black", label="Baseline")

    # handel1_2 = ax2.fill_between(
    #     rel_meas,
    #     mean_rmse - std_rmse,
    #     mean_rmse + std_rmse,
    #     alpha=0.1,
    #     color="black",
    #     label=r"$\pm$ 1 std. dev.",
    # )

    ax2.set_xlim(1, 0.5)
    ax2.set_xlabel("measurement per load change")
    ax2.set_ylabel("rmse")
    ax1.legend(handles=[handel2, handel1_1, handel1_2])


plot_rmse_over_wrong_vs_baseline()
save(name="Impact of missed detection ratio on rmse vs baseline", path=path)

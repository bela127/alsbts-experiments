from dataclasses import dataclass
from typing import Any, Callable, Tuple
from typing_extensions import Protocol
from matplotlib import pyplot as plot
from nptyping import Float, NDArray, Shape
import numpy as np
import os


sim_time = 600
dist = 120


def load_data(path, file = "all_data_source_data.npy", skip_points = 0):
    data = np.load(os.path.join(path, file))
    sim_res = SimRes(*np.transpose(data[skip_points:]))
    return sim_res

@dataclass
class SimRes():
    time: NDArray[Shape["Batch, 1"], Float]
    voltage: NDArray[Shape["Batch, 1"], Float]
    v_change: NDArray[Shape["Batch, 1"], Float]
    active_power: NDArray[Shape["Batch, 1"], Float]
    reactive_power: NDArray[Shape["Batch, 1"], Float]
    gt_vs: NDArray[Shape["Batch, 1"], Float]
    vs_estimation: NDArray[Shape["Batch, 1"], Float]
    measurement_trigger: NDArray[Shape["Batch, 1"], Float]
    rvs: NDArray[Shape["Batch, 1"], Float]
    change: NDArray[Shape["Batch, 1"], Float]
    measurement_active: NDArray[Shape["Batch, 1"], Float]
    vs_measurement: NDArray[Shape["Batch, 1"], Float]

def plot_pred_vs_gt(sim_res: SimRes):
    plot.plot(sim_res.time, sim_res.gt_vs)
    plot.plot(sim_res.time, sim_res.vs_estimation)
    plot.show()

def calc_measurements(sim_res: SimRes)-> float:
    measurements = np.sum(sim_res.measurement_trigger)
    return measurements

def calc_rmse(sim_res: SimRes)-> float:
    rmse = np.sqrt(np.sum((sim_res.gt_vs - sim_res.vs_estimation)**2)/len(sim_res.gt_vs))
    return rmse

def calc_mean_rmse(rmses):

    mean_rmse = np.mean(rmses)
    std_rmse = np.std(rmses)

    return mean_rmse, std_rmse

def calc_mean_meas(meas):

    mean_meas = np.mean(meas)
    std_meas = np.std(meas)

    return mean_meas, std_meas

def check_sim_data_ok(sim_res: SimRes):
    return not np.any(sim_res.vs_measurement < 0)

class FileWorker(Protocol):
    def __call__(self, dirpath: str, file: str) -> Any: ...

def walk_files(path: str, worker: FileWorker):
    for dirpath, dnames, fnames in os.walk(path):
        f: str
        for f in fnames:
            if f.endswith(".npy"):
                yield worker(dirpath = dirpath, file = f)

class DirWorker(Protocol):
    def __call__(self, dirpath: str, dir: str) -> Any: ...

def walk_dirs(path: str, worker: DirWorker):
    for dirpath, dnames, fnames in os.walk(path):
        f: str
        for d in dnames:
                yield worker(dirpath = dirpath, dir = d)


def rmse_file_worker(dirpath, file):
    sim_res = load_data(skip_points = 0, path=dirpath)

    if not check_sim_data_ok(sim_res):
        print("Corrupt Sim Data: ", dirpath)
        print("Min = ", np.min(sim_res.vs_measurement))
        sim_res.vs_measurement[sim_res.vs_measurement < 0] = 0
        sim_res.vs_estimation[sim_res.vs_estimation < 0] = 0

    rmse = calc_rmse(sim_res)
    nr_measurements = calc_measurements(sim_res)
    return rmse, nr_measurements

def exp_dir_worker(sub_exp_folder: str):
    def dir_worker(dirpath, dir: str):
        if dir.startswith(sub_exp_folder):
            exp_quantity_str = dir.lstrip(sub_exp_folder)
            exp_quantity = float(exp_quantity_str)

            rmse_meas = [rmse_meas for rmse_meas in walk_files(path=os.path.join(dirpath, dir), worker=rmse_file_worker)]
            rmse , meas = np.asarray(rmse_meas).T

            mean_rmse = np.mean(rmse)
            std_rmse = np.std(rmse)

            mean_meas = np.mean(meas)
            std_meas = np.std(meas)

            quant_rmse_std_meas_std = (exp_quantity, mean_rmse, std_rmse, mean_meas, std_meas)
            return quant_rmse_std_meas_std
    return dir_worker

def norm_time(time):
    return time / sim_time

def norm_measurements(measurements):
    return measurements / dist

def save(name, path):
    plot.title(name)
    loc = os.path.join(path,f"{name}.svg")
    plot.savefig(loc, format="svg")
    plot.clf()
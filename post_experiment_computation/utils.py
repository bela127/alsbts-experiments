from dataclasses import dataclass
from typing import Any, Callable
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

def check_sim_data_ok(sim_res: SimRes):
    return not np.any(sim_res.vs_measurement < 0)

def walk_files(path: str, worker: Callable[[str, str], Any]):
    for dirpath, dnames, fnames in os.walk(path):
        f: str
        for f in fnames:
            if f.endswith(".npy"):
                yield worker(dirpath = dirpath, file = f)

            
def walk_dirs(path: str, worker: Callable[[str, str], Any]):
    for dirpath, dnames, fnames in os.walk(path):
        f: str
        for d in dnames:
                yield worker(dirpath = dirpath, dir = d)

def norm_time(time):
    return time / sim_time

def norm_measurements(measurements):
    return measurements / dist

def save(name, path):
    plot.title(name)
    loc = os.path.join(path,f"{name}.svg")
    plot.savefig(loc, format="svg")
    plot.clf()
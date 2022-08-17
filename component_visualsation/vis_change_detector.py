import numpy as np
from matplotlib import pyplot as plot # type: ignore

from alvsts.modules.change_detector import NoisyChangeDetector
from alvsts.modules.change_detector import OptimalChangeDetector
from alvsts.modules.rvs_estimator import OptimalRVSEstimator

from utils import generate_data

rvs_est = OptimalRVSEstimator()()
cd = NoisyChangeDetector()()
opt_cd = OptimalChangeDetector()()

time = []
gt_changes = []
changes = []
for data in generate_data(rvs_est, cd, time=60):
    vs_gt, timeOutput, voltageOutput, knewVOutput, activePowerOutput, reactivePowerOutput, rvs, change = data
    gt_change = opt_cd.detect(vs_gt, timeOutput, voltageOutput, knewVOutput, activePowerOutput, reactivePowerOutput, rvs)
    time.append(timeOutput)
    gt_changes.append(gt_change)
    changes.append(change)


time = np.asarray(time)
gt_changes = np.asarray(gt_changes)
changes = np.asarray(changes)
kinds = cd.change_kinds

plot.figure(figsize=(10, 2))
mask = gt_changes > 0
err_time = time[mask]
err_change = gt_changes[mask]
plot.errorbar(err_time, err_change, yerr=5, fmt=" ",label='GT changes')

unique_kinds = set(kinds)  # or yo can use: np.unique(m)
markers = {0: "o", 1: "o", -1: "x", -2: "+"}
colors = {0: "blue", 1: "lightgreen", -1: "red", -2: "grey"}
labels = {0: "blue", 1: "correct detection", -1: "false detection", -2: "missed detection"}
for uk in unique_kinds:
    mask = kinds == uk
    if uk != 0:
        # mask is now an array of booleans that can be used for indexing
        masked_time = time[mask]
        change_pos = np.ones_like(masked_time)
        plot.scatter(masked_time, change_pos, marker=markers[uk], c=colors[uk], label=labels[uk])

plot.ylim(0.8,1.2)
plot.yticks([])
plot.legend()

plot.show()
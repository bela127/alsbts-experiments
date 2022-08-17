
import numpy as np
from alvsts.modules.rvs_estimator import RVSEstimator
from alvsts.modules.change_detector import ChangeDetector


def generate_data(rvs_estimator: RVSEstimator, cd: ChangeDetector, vs_min = 0, vs_max = 2, time = 600):
    
    change_times = np.arange(0, time, 5)
    vs_gt_changes = np.random.uniform(vs_min, vs_max, size=len(change_times))

    vs_gts = []
    run_times = np.arange(0, time, 0.2)
    for run_time in run_times:
        mask = change_times < run_time
        last_vs_gt = vs_gt_changes[mask]
        if last_vs_gt.size == 0:
            vs_gt = 0
        else:
            vs_gt = last_vs_gt[-1]
        vs_gts.append(vs_gt)
    vs_gts = np.asarray(vs_gts)

    voltageOutput_s = np.random.normal(loc = 1, scale= 0.1, size = len(run_times))
    knewVOutput_s = np.random.normal(loc = 0, scale= 0.1, size = len(run_times))
    activePowerOutput_s = np.random.normal(loc = 5000, scale= 1, size = len(run_times))
    reactivePowerOutput_s = np.random.normal(loc = 4900, scale= 1, size = len(run_times))

    for vs_gt, timeOutput, voltageOutput, knewVOutput, activePowerOutput, reactivePowerOutput in zip(vs_gts, run_times, voltageOutput_s, knewVOutput_s, activePowerOutput_s, reactivePowerOutput_s):
        rvs = rvs_estimator.estimate(vs_gt, timeOutput, voltageOutput, knewVOutput, activePowerOutput, reactivePowerOutput)
        change = cd.detect(vs_gt, timeOutput, voltageOutput, knewVOutput, activePowerOutput, reactivePowerOutput, rvs)
        yield vs_gt, timeOutput, voltageOutput, knewVOutput, activePowerOutput, reactivePowerOutput, rvs, change
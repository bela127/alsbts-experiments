import numpy as np
from matplotlib import pyplot as plot # type: ignore

from alvsts.modules.rvs_estimator import NoisyGaussianRVSEstimator

rvs_estimator = NoisyGaussianRVSEstimator(noise_var=0.001, length_scale=0.2 , probability_of_function_change=-1, change_size_proportion=0)()

vs_min = 0
vs_max = 2

timeOutput_s = np.arange(0, 200, 2)
vs_gt_s = np.sort(np.random.uniform(vs_min, vs_max, size=len(timeOutput_s)))
voltageOutput_s = np.random.normal(loc = 1, scale= 0.1, size = len(timeOutput_s))
knewVOutput_s = np.random.normal(loc = 0, scale= 0.1, size = len(timeOutput_s))
activePowerOutput_s = np.random.normal(loc = 5000, scale= 1, size = len(timeOutput_s))
reactivePowerOutput_s = np.random.normal(loc = 4900, scale= 1, size = len(timeOutput_s))

rvs_s = []
for vs_gt, timeOutput, voltageOutput, knewVOutput, activePowerOutput, reactivePowerOutput in zip(vs_gt_s, timeOutput_s, voltageOutput_s, knewVOutput_s, activePowerOutput_s, reactivePowerOutput_s):
    rvs = rvs_estimator.estimate(vs_gt, timeOutput, voltageOutput, knewVOutput, activePowerOutput, reactivePowerOutput)
    rvs_s.append(rvs)

plot.xlim(0,2)
plot.plot(vs_gt_s, rvs_s)
plot.show()

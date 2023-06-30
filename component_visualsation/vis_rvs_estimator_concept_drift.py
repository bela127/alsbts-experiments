import numpy as np
from matplotlib import pyplot as plot # type: ignore

from alsbts.modules.change_detector import NoisyChangeDetector
from alsbts.modules.change_detector import OptimalChangeDetector
from alsbts.modules.rvs_estimator import NoisyGaussianRVSEstimator

from utils import generate_data

rvs_est = NoisyGaussianRVSEstimator(noise_var=0.0,length_scale=0.2,probability_of_function_change=0.02,change_size_proportion=0.50)()
opt_cd = OptimalChangeDetector()()


vs_gts = []
rvss = []
times = []
for data in generate_data(rvs_est, opt_cd, time=1000):
    vs_gt, timeOutput, voltageOutput, knewVOutput, activePowerOutput, reactivePowerOutput, rvs, change = data
    vs_gts.append(vs_gt)
    rvss.append(rvs)
    times.append(timeOutput)

time = np.asarray(times)
vs_gts = np.asarray(vs_gts)
rvss = np.asarray(rvss)
change_times = rvs_est.change_times

last_time = 0
for change_time in change_times:
    m1 = time >= last_time
    m2 = time < change_time
    mask = m1 & m2
    

    vs_gts_profile = vs_gts[mask]
    rvss_profile = rvss[mask]

    index = vs_gts_profile.argsort()

    plot.plot(vs_gts_profile[index], rvss_profile[index], label=f"mapping at t={last_time}")
    last_time = change_time

mask = time >= last_time

vs_gts_profile = vs_gts[mask]
rvss_profile = rvss[mask]

index = vs_gts_profile.argsort()

plot.plot(vs_gts_profile[index], rvss_profile[index], label=f"mapping at t={last_time}")

plot.title("Evolvement of rVS mapping over time")
plot.xlabel("GT VS")
plot.ylabel("rVS")

plot.legend()
plot.show()

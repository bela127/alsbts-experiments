# vs=(dp/p)/(dv/v)
# vs*(dv/v)=(dp/p)
# dv/v=(dp/p)/vs
# dv=(dp/p)/vs*v

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

min_vs = 0.01
max_vs = 2
v_std = 0.0001
v=1
p_std = 1
p=5000

plot_points = 10000

rng = np.random.RandomState(None)

vs = rng.uniform(min_vs, max_vs, plot_points)
dp = rng.normal(0, p_std, plot_points)
dv = (dp/p)/vs*v


bins = np.linspace(min_vs, max_vs, 100)
bin_centers = (bins[:-1] + bins[1:])/2
digitized = np.digitize(vs, bins)
dv_means = np.asarray([dv[digitized == i].mean() for i in range(1, len(bins))])
dv_std = np.asarray([dv[digitized == i].std() for i in range(1, len(bins))])

plt.scatter(vs, dv)
plt.plot(bin_centers, dv_means)
plt.plot(bin_centers, dv_std*2)
plt.show()

import GPy
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from alts.modules.oracle.data_source import BrownianDriftDataSource
from alsbts.modules.estimator import Combined

min_x = -1
max_x = 1
stop_time =1000

nr_plot_points = 500
nr_time_steps = 6

rng = np.random.RandomState(None)

#ds = BrownianDriftDataSource(brown_var = 0.005, rbf_var = 0.25 ,rbf_leng = 0.1)() # brown_var = 0.05, rbf_var = 0.25 ,rbf_leng = 0.1
ds = BrownianDriftDataSource()() # brown_var = 0.05, rbf_var = 0.25 ,rbf_leng = 0.1


x_train = np.linspace(min_x, max_x, nr_plot_points)
t_train = np.linspace(0, stop_time, num=nr_time_steps)

x_grid, t_grid = np.meshgrid(x_train, t_train)

x = np.reshape(x_grid,(-1,1))
t = np.reshape(t_grid,(-1,1))

data_train = np.concatenate((t, x), axis=1)

queries, y = ds.query(data_train)


y_train = np.reshape(y,(6,-1))

for i in range(nr_time_steps):
    plt.plot(x_train, y_train[i,:], label=f"mapping at t={t_train[i]}")

plt.title("Evolvement of concept over time")
plt.xlabel("x")
plt.ylabel("y")

plt.legend()
path = "./concept_drift.svg"
plt.savefig(path, format="svg")
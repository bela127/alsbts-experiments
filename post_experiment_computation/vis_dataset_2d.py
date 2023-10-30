from alts.modules.oracle.data_source import MixedBrownDriftDataSource, MixedDriftDataSource, RBFDriftDataSource, BrownianDriftDataSource, SinDriftDataSource, GaussianProcessDataSource
import numpy as np
from matplotlib import pyplot as plt # type: ignore
from utils import create_fig, save

dss = [SinDriftDataSource, MixedDriftDataSource, RBFDriftDataSource, BrownianDriftDataSource, ]

def compute_data():
    x = np.linspace(-1,1,1000)
    t = np.linspace(0,1000,1000)

    tx_grid = np.meshgrid(t,x)
    ts, xs = tx_grid

    tx = np.reshape(tx_grid, (2, -1)).T

    np.save(f"./vis_data/data_ts.npy", ts)
    np.save(f"./vis_data/data_xs.npy", xs)

    for dsc in dss:

        ds: GaussianProcessDataSource = dsc()

        ds: GaussianProcessDataSource = ds()

        tx, y = ds.query(tx)
        y_vis = np.reshape(y.T, (1000,1000))

        np.save(f"./vis_data/data_{ds.__class__.__name__}_y_vis.npy", y_vis)

plot_names = [r"$C_{sin}(x(t),t))$",r"$C_{rbf}(x(t),t))$", r"$C_{mix}(x(t),t))$", r"$C_{b}(x(t),t))$"]
ds_names = {dsc.__name__: value for dsc, value in zip(dss, plot_names)}

def plot_data():
    fig, axs = create_fig(subplots=(2,2), width="paper", fraction=1, hfrac=2/3)

    ts = np.load(f"./vis_data/data_ts.npy")
    xs = np.load(f"./vis_data/data_xs.npy")


    for ax, dsc in zip(axs.flat, dss):

        y_vis = np.load(f"./vis_data/data_{dsc.__name__}_y_vis.npy")
        y_min = np.min(y_vis)
        y_max = np.max(y_vis)
        y_vis = (y_vis - (y_min + y_max) / 2) / (y_max - y_min) * 2

    # Plot the contour.
        pc = ax.pcolor(ts, xs, y_vis, cmap="plasma", antialiased=True, rasterized=True)
        ax.set_ylabel("x")
        ax.set_xlabel("time t")
        ax.set_title(f"Sampled time series from {ds_names[dsc.__name__]}")

    fig.suptitle("Time series sampled from different priors")
    fig.colorbar(pc, ax=axs.ravel().tolist(),label=r"$c(x(t),t))$")

    save(fig, "data_processes", "./vis_data", format="svg")

#compute_data()
plot_data()
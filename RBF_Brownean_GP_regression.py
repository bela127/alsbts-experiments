import GPy
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from alts.modules.oracle.data_source import BrownianDriftDataSource
from alsbts.modules.estimator import Combined

min_p = 0
max_p = 1
stop_time =900

nr_plot_points = 100
number_of_train_points = 150

rng = np.random.RandomState(None)

ds = BrownianDriftDataSource(brown_var = 0.005, rbf_var = 0.25 ,rbf_leng = 0.1)()



p_train = rng.uniform(min_p, max_p, number_of_train_points)
t_train = np.linspace(0, stop_time, num=p_train.shape[0])

x_train = np.concatenate((t_train[:,None], p_train[:,None]), axis=1)

queries, y_train = ds.query(x_train)



from GPy.kern import Kern
from GPy.core.parameterization import Param
from paramz.transformations import Logexp
from paramz.caching import Cache_this


k = Combined(rbf_lengthscale=0.1 ,rbf_variance=0.25, brown_variance=0.005)
m = GPy.models.GPRegression(x_train, y_train, k, noise_var=0.00001)
print(m)

m.Gaussian_noise.variance.fix()
#m.Combined.rbf_lengthscale = 0.1
#m.Combined.rbf_lengthscale.fix()
m.Combined.rbf_variance.fix()
print(m)


def plot_3d(gp: GPy.models.GPRegression, title='Estimated Model', random = False):
    X = np.linspace(min_p, max_p, num=nr_plot_points)
    t = np.linspace(0, stop_time*1.2, num=nr_plot_points)
    X_pred, t_pred = np.meshgrid(X, t)
    
    pred = np.concatenate((t_pred.reshape((-1,1)), X_pred.reshape((-1,1))), axis=1)

    Ypred,YpredCov = gp.predict_noiseless(pred)
    SE = np.sqrt(YpredCov)[:,0]
    vs_pred = Ypred.reshape((X_pred.shape))
    SE_pred = SE.reshape(X_pred.shape)

    color_dimension = SE_pred*1.96
    minn, maxx = color_dimension.min(), color_dimension.max()
    norm = matplotlib.colors.Normalize(minn, maxx)
    m = plt.cm.ScalarMappable(norm=norm, cmap='jet')
    m.set_array([])
    fcolors = m.to_rgba(color_dimension)
    fcolors[:,:,3] = 0.2

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')




    ax.plot_surface(t_pred, X_pred, vs_pred, rstride=1, cstride=1, facecolors=fcolors, vmin=minn, vmax=maxx, shade=False)

    ax.scatter(x_train[:,0], x_train[:,1], y_train,label='GT Data', )
    for t in x_train:
        ax.plot([t[0],t[0]],[t[1],t[1]],[0,2])
    ax.scatter(stop_time*1.2 * np.ones_like(x_train[:,0]), x_train[:,1], y_train,label='GT Data', )

    ax.set_xlabel('time')
    ax.set_ylabel('rvs')
    ax.set_zlabel('vs')


    plt.title(title)
    plt.legend()
    plt.colorbar(m)


m.optimize_restarts(num_restarts=5, max_iters=1000, messages=True, ipython_notebook=False)
print(m)

plot_3d(m)
plt.show()

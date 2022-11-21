import GPy
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

min_vs = 0
max_vs = 2
std = 0.0001
stop_time =900

nr_plot_points = 100
number_of_train_points = 70

n_samples = 5 #Number of function realizations

rng = np.random.RandomState(None)


vs_train = rng.uniform(min_vs, max_vs, number_of_train_points)
t_train = np.linspace(0, stop_time, num=vs_train.shape[0])
noise_train =  rng.normal(0, t_train*std)
rvs_train = vs_train + 0.2*np.sin(vs_train*10) + noise_train #+ t_train

x_train = np.concatenate((t_train[:,None], rvs_train[:,None]), axis=1)


from GPy.kern import Kern
from GPy.core.parameterization import Param
from paramz.transformations import Logexp
from paramz.caching import Cache_this

class Combined(Kern):
    """
    Abstract class for change kernels
    """
    def __init__(self, input_dim = 2, active_dims=None,  rbf_variance = 10, rbf_lengthscale = 0.4, brown_variance = 10, name = 'Combined'):

        super(Combined, self).__init__(input_dim, active_dims, name)

        self.brown = GPy.kern.Brownian(variance=brown_variance, active_dims=[0])
        self.rbf = GPy.kern.RBF(variance=rbf_variance,lengthscale=rbf_lengthscale, input_dim=1, active_dims=[1])
        self.rbf_add = GPy.kern.RBF(variance=rbf_variance,lengthscale=rbf_lengthscale, input_dim=1, active_dims=[1])

        self.rbf_variance = Param('rbf_variance', rbf_variance, Logexp())
        self.link_parameter(self.rbf_variance)
        self.rbf_lengthscale = Param('rbf_lengthscale', rbf_lengthscale, Logexp())
        self.link_parameter(self.rbf_lengthscale)
        self.brown_variance = Param('brown_variance', brown_variance, Logexp())
        self.link_parameter(self.brown_variance)

    def parameters_changed(self):
        self.rbf.variance = self.rbf_add.variance = self.rbf_variance
        self.rbf.lengthscale = self.rbf_add.lengthscale = self.rbf_lengthscale
        self.brown.variance = self.brown_variance

    @Cache_this(limit = 3)
    def K(self, X, X2 = None):
        return self.rbf_add.K(X, X2) + self.brown.K(X, X2) * self.rbf.K(X, X2)

    @Cache_this(limit = 3)
    def Kdiag(self, X):
        return self.rbf_add.Kdiag(X) + self.brown.Kdiag(X) * self.rbf.Kdiag(X)

    # NOTE ON OPTIMISATION:
    #   Should be able to get away with only optimising the parameters of one sigmoidal kernel and propagating them

    def update_gradients_full(self, dL_dK, X, X2 = None): # See NOTE ON OPTIMISATION
        self.brown.update_gradients_full(dL_dK * self.rbf.K(X, X2), X, X2)
        self.rbf.update_gradients_full(dL_dK * self.brown.K(X, X2), X, X2)

        self.rbf_add.update_gradients_full(dL_dK, X, X2)

        self.rbf_variance.gradient = self.rbf.variance.gradient + self.rbf_add.variance.gradient
        self.rbf_lengthscale.gradient = self.rbf.lengthscale.gradient + self.rbf_add.lengthscale.gradient
        self.brown_variance.gradient = self.brown.variance.gradient


    def update_gradients_diag(self, dL_dK, X):
        self.brown.update_gradients_diag(dL_dK * self.rbf.Kdiag(X), X)
        self.rbf.update_gradients_diag(dL_dK * self.brown.Kdiag(X), X)

        self.rbf_add.update_gradients_diag(dL_dK, X)

        self.rbf_variance.gradient = self.rbf.variance.gradient + self.rbf_add.variance.gradient
        self.rbf_lengthscale.gradient = self.rbf.lengthscale.gradient + self.rbf_add.lengthscale.gradient
        self.brown_variance.gradient = self.brown.variance.gradient

k = Combined() + GPy.kern.Linear(input_dim=1, active_dims=[1], variances=0.01)
m = GPy.models.GPRegression(x_train, vs_train[:,None], k, noise_var=0.01)
print(m)

m.Gaussian_noise.variance.fix()
m.sum.linear.variances.fix()
#m.sum.Combined.rbf_lengthscale = 2
#m.sum.Combined.rbf_lengthscale.fix()
print(m)


def plot_3d(gp: GPy.models.GPRegression, title='Estimated Model'):
    X = np.linspace(min_vs, max_vs, num=nr_plot_points)
    t = np.linspace(0, stop_time*1.2, num=nr_plot_points)
    X_pred, t_pred = np.meshgrid(X, t)
    pred = np.concatenate((t_pred.reshape((-1,1)), X_pred.reshape((-1,1))), axis=1)

    Ypred,YpredCov = gp.predict_noiseless(pred)
    SE = np.sqrt(YpredCov)[:,0]
    vs_pred = Ypred.reshape(X_pred.shape)
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

    ax.scatter(x_train[:,0], x_train[:,1], vs_train,label='GT Data', )
    for t in x_train:
        ax.plot([t[0],t[0]],[t[1],t[1]],[0,2])
    ax.scatter(stop_time*1.2 * np.ones_like(x_train[:,0]), x_train[:,1], vs_train,label='GT Data', )

    ax.set_xlabel('time')
    ax.set_ylabel('rvs')
    ax.set_zlabel('vs')


    plt.title(title)
    plt.legend()
    plt.colorbar(m)


#grad_correct = m.checkgrad()
m.optimize_restarts(num_restarts=5, max_iters=10000, messages=True, ipython_notebook=False)
print(m)


plot_3d(m)
plt.show()
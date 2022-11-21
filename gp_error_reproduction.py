import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Hyperparameter,  Kernel


min_x = 0
max_x = 50
std = 0.2
stop_time = 50

nr_plot_points = 20
number_of_train_points = 5

class MinT(Kernel):

    def __init__(self, sigma_0=1.0, sigma_0_bounds=(0.01, 10)):
        self.sigma_0 = sigma_0
        self.sigma_0_bounds = sigma_0_bounds

    @property
    def hyperparameter_sigma_0(self):
        return Hyperparameter("sigma_0", "numeric", self.sigma_0_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : ndarray of shape (n_samples_Y, n_features), default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.
            Only supported when Y is None.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims),\
                optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when `eval_gradient`
            is True.
        """
        X = np.atleast_2d(X)
        ones_x = np.ones_like(X)
        if Y is None:
            Kc = X*ones_x.T
            Kr = ones_x * X.T
            Kcr = np.concatenate((Kc[...,None], Kr[...,None]), axis=-1)
            K = np.min(Kcr, axis=-1)
        else:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            
            ones_y = np.ones_like(Y)
            Kc = X*ones_y.T
            Kr = ones_x * Y.T
            Kcr = np.concatenate((Kc[...,None], Kr[...,None]), axis=-1)
            K = np.min(Kcr, axis=-1)


        if eval_gradient:
            if not self.hyperparameter_sigma_0.fixed:
                K_gradient = np.empty((K.shape[0], K.shape[1], 1))
                K_gradient[..., 0] = self.sigma_0
                return K, K_gradient
            else:
                return K, np.empty((X.shape[0], X.shape[0], 0))
        else:
            return K

    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y).

        Returns
        -------
        K_diag : ndarray of shape (n_samples_X,)
            Diagonal of kernel k(X, X).
        """
        return X[:,0] #np.copy(X[:,0])

    def is_stationary(self):
        """Returns whether the kernel is stationary."""
        return False

    def __repr__(self):
        return "{0}(sigma_0={1:.3g})".format(self.__class__.__name__, self.sigma_0)


rng = np.random.RandomState(None)


t_train = np.linspace(0, stop_time, num=number_of_train_points)
y_train =  rng.normal(0, t_train*std)

gpr_min_t = GaussianProcessRegressor(kernel=MinT(), random_state=None, n_restarts_optimizer=5)

gpr_min_t.fit(t_train.reshape(-1, 1), y_train)
x = np.linspace(min_x, max_x, nr_plot_points)
X = x.reshape(-1, 1)

X_copy = np.copy(X)

y_mean, y_std = gpr_min_t.predict(X, return_std=True)

assert np.all(X_copy == X)

import sklearn; sklearn.show_versions()
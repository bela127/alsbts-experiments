import matplotlib.pyplot as plot
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Hyperparameter, StationaryKernelMixin, NormalizedKernelMixin, Kernel, DotProduct, RBF

from scipy.spatial.distance import pdist, cdist, squareform

min_x = 0
max_x = 50
std = 0.2
stop_time = 50

nr_plot_points = 100
number_of_train_points = 10

n_samples = 5 #Number of function realizations

def _num_samples(x):
    """Return number of samples in array-like x."""
    message = "Expected sequence or array-like, got %s" % type(x)
    if hasattr(x, "fit") and callable(x.fit):
        # Don't get num_samples from an ensembles length!
        raise TypeError(message)

    if not hasattr(x, "__len__") and not hasattr(x, "shape"):
        if hasattr(x, "__array__"):
            x = np.asarray(x)
        else:
            raise TypeError(message)

    if hasattr(x, "shape") and x.shape is not None:
        if len(x.shape) == 0:
            raise TypeError(
                "Singleton array %r cannot be considered a valid collection." % x
            )
        # Check that shape is returning an integer or default to len
        # Dask dataframes may not return numeric shape[0] value
        if isinstance(x.shape[0], np.integer):
            return x.shape[0]

    try:
        return len(x)
    except TypeError as type_error:
        raise TypeError(message) from type_error

def _check_divergence(X, divergence):
    divergence = np.squeeze(divergence).astype(float)
    if np.ndim(divergence) > 1:
        raise ValueError("divergence cannot be of dimension greater than 1")
    if np.ndim(divergence) == 1 and X.shape[1] != divergence.shape[0]:
        raise ValueError(
            "Anisotropic kernel must have the same number of "
            "dimensions as data (%d!=%d)" % (divergence.shape[0], X.shape[1])
        )
    return divergence

class KF(Kernel):
    r"""Dot-Product kernel.

    The DotProduct kernel is non-stationary and can be obtained from linear
    regression by putting :math:`N(0, 1)` priors on the coefficients
    of :math:`x_d (d = 1, . . . , D)` and a prior of :math:`N(0, \sigma_0^2)`
    on the bias. The DotProduct kernel is invariant to a rotation of
    the coordinates about the origin, but not translations.
    It is parameterized by a parameter sigma_0 :math:`\sigma`
    which controls the inhomogenity of the kernel. For :math:`\sigma_0^2 =0`,
    the kernel is called the homogeneous linear kernel, otherwise
    it is inhomogeneous. The kernel is given by

    .. math::
        k(x_i, x_j) = \sigma_0 ^ 2 + x_i \cdot x_j

    The DotProduct kernel is commonly combined with exponentiation.

    See [1]_, Chapter 4, Section 4.2, for further details regarding the
    DotProduct kernel.

    Read more in the :ref:`User Guide <gp_kernels>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    sigma_0 : float >= 0, default=1.0
        Parameter controlling the inhomogenity of the kernel. If sigma_0=0,
        the kernel is homogeneous.

    sigma_0_bounds : pair of floats >= 0 or "fixed", default=(1e-5, 1e5)
        The lower and upper bound on 'sigma_0'.
        If set to "fixed", 'sigma_0' cannot be changed during
        hyperparameter tuning.

    References
    ----------
    .. [1] `Carl Edward Rasmussen, Christopher K. I. Williams (2006).
        "Gaussian Processes for Machine Learning". The MIT Press.
        <http://www.gaussianprocess.org/gpml/>`_

    Examples
    --------
    >>> from sklearn.datasets import make_friedman2
    >>> from sklearn.gaussian_process import GaussianProcessRegressor
    >>> from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
    >>> X, y = make_friedman2(n_samples=500, noise=0, random_state=0)
    >>> kernel = DotProduct() + WhiteKernel()
    >>> gpr = GaussianProcessRegressor(kernel=kernel,
    ...         random_state=0).fit(X, y)
    >>> gpr.score(X, y)
    0.3680...
    >>> gpr.predict(X[:2,:], return_std=True)
    (array([653.0..., 592.1...]), array([316.6..., 316.6...]))
    """

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
        if Y is None:
            #K = np.inner(X, X) #+ self.sigma_0**2
            #K = np.abs(X-X.T) + self.sigma_0**2
            #K = K - np.min(K) + self.sigma_0**2
            K = X  * np.eye(_num_samples(X)) * self.sigma_0
        else:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            #K = np.inner(X, Y) + self.sigma_0**2
            #K = np.abs(X-X.T) + self.sigma_0**2
            #K = K - np.min(K) + self.sigma_0**2
            #K = np.eye(_num_samples(X))
            K = np.zeros((_num_samples(X), _num_samples(Y)))
            #ones = np.ones_like(Y)
            #K = X*ones.T - np.concatenate((np.asarray([[0]]), Y))[:-1,:].T
            #K = K * (K > 0).astype(float) * self.sigma_0**2 


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
        return X[:,0] * self.sigma_0

    def is_stationary(self):
        """Returns whether the kernel is stationary."""
        return False

    def __repr__(self):
        return "{0}(sigma_0={1:.3g})".format(self.__class__.__name__, self.sigma_0)


gpr = GaussianProcessRegressor(kernel=KF(), random_state=None, n_restarts_optimizer=5)

def plot_gpr_samples(gpr_model, n_samples, ax):
    """Plot samples drawn from the Gaussian process model.

    If the Gaussian process model is not trained then the drawn samples are
    drawn from the prior distribution. Otherwise, the samples are drawn from
    the posterior distribution. Be aware that a sample here corresponds to a
    function.

    Parameters
    ----------
    gpr_model : `GaussianProcessRegressor`
        A :class:`~sklearn.gaussian_process.GaussianProcessRegressor` model.
    n_samples : int
        The number of samples to draw from the Gaussian process distribution.
    ax : matplotlib axis
        The matplotlib axis where to plot the samples.
    """
    x = np.linspace(min_x, max_x, nr_plot_points)
    X = x.reshape(-1, 1)

    y_mean, y_std = gpr_model.predict(X, return_std=True)
    y_samples = gpr_model.sample_y(X, n_samples)

    for idx, single_prior in enumerate(y_samples.T):
        ax.plot(
            x,
            single_prior,
            linestyle="--",
            alpha=0.7,
            label=f"Sampled function #{idx + 1}",
        )
    ax.plot(x, y_mean, color="black", label="Mean")
    ax.fill_between(
        x,
        y_mean - y_std,
        y_mean + y_std,
        alpha=0.1,
        color="black",
        label=r"$\pm$ 1 std. dev.",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    #ax.set_ylim([-3, 3])

rng = np.random.RandomState(None)


X_train = rng.uniform(min_x, max_x, number_of_train_points).reshape(-1, 1)
t_train = np.linspace(0, stop_time, num=X_train.shape[0])
y_train =  rng.normal(0, X_train[:, 0]*std)

gpr_t = GaussianProcessRegressor(kernel=1*RBF(length_scale=0.5),alpha=0.01, random_state=None, n_restarts_optimizer=5)


fig, axs = plot.subplots(nrows=2, sharex=True, sharey=True, figsize=(10, 8))

# plot prior
plot_gpr_samples(gpr_t, n_samples=n_samples, ax=axs[0])
axs[0].set_title("Samples from prior distribution")

# plot posterior
gpr_t.fit(X_train, t_train)
plot_gpr_samples(gpr_t, n_samples=n_samples, ax=axs[1])
axs[1].scatter(X_train[:, 0], t_train, color="red", zorder=10, label="Observations")
axs[1].legend(bbox_to_anchor=(1.05, 1.5), loc="upper left")
axs[1].set_title("Samples from posterior distribution")

fig.suptitle("Radial Basis Function kernel", fontsize=18)
plot.tight_layout()
plot.show()


fig, axs = plot.subplots(nrows=2, sharex=True, sharey=True, figsize=(10, 8))

# plot prior
plot_gpr_samples(gpr, n_samples=n_samples, ax=axs[0])
axs[0].set_title("Samples from prior distribution")

# plot posterior
gpr.fit(X_train, y_train)
plot_gpr_samples(gpr, n_samples=n_samples, ax=axs[1])
axs[1].scatter(X_train[:, 0], y_train, color="red", zorder=10, label="Observations")
axs[1].legend(bbox_to_anchor=(1.05, 1.5), loc="upper left")
axs[1].set_title("Samples from posterior distribution")

fig.suptitle("Radial Basis Function kernel", fontsize=18)
plot.tight_layout()
plot.show()

import GPy
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from alsbts.modules.estimator import IntegralBrown
from post_experiment_computation.utils import create_fig, save

print(plt.style.available)

std = 0.1
stop_time = 25

nr_plot_points = 10000
number_of_train_points = 100

n_samples = 5 #Number of function realizations

rng = np.random.RandomState(None)


t_train = np.linspace(0, stop_time, num=number_of_train_points+1)

t_train_low = t_train[:-1]
t_train_up = t_train[1:]

noise_train =  rng.normal(0, t_train_low*std)
y_train = noise_train

train = np.concatenate((t_train_low[:,None], t_train_up[:,None]), axis=1)


def plot_integral(ax, gp: GPy.models.GPRegression, title='Estimated Model'):
    Xtest = np.linspace(0, stop_time*1.2, num=nr_plot_points+1)
    Xpred = np.array([Xtest[:-1],Xtest[1:]])
    Ypred,YpredCov = gp.predict_noiseless(Xpred.T)
    SE = np.sqrt(YpredCov)

    ax.scatter((t_train_up + t_train_low)/2, y_train,label='Measurements')
    ax.plot((Xpred[1]+Xpred[0])/2, Ypred,'r-',label='Estimation $y_{est}$')
    ax.plot((Xpred[1]+Xpred[0])/2,Ypred+SE*1.96,'r:',label='$v_{est}$')
    ax.plot((Xpred[1]+Xpred[0])/2,Ypred-SE*1.96,'r:')
    ax.set_title(title)
    ax.set_xlabel('time $t$')
    ax.set_ylabel('target-variable $y$')

    ax.set_xlim(0,10)
    ax.set_ylim(-3,3)

    plt.legend()
    

res = (y_train* (t_train_up - t_train_low))[:,None]
k = IntegralBrown(variance=1)
m = GPy.models.GPRegression(train, res, k, noise_var=0.0)
#train [100,2] ;

fig_uc, axs_uc = create_fig(subplots=(1,2), width="paper")
#fig_c, axs_c = create_fig(subplots=(1,2), width="paper")

fig_c_b, ax_c_b = create_fig(subplots=(1,1), width="paper", fraction=0.5)
fig_c_ib, ax_c_ib = create_fig(subplots=(1,1), width="paper", fraction=0.5)

print(m)
plot_integral(axs_uc[1], m, "Uncalibrated Model: $Int\_Brown$")

m.Gaussian_noise.variance.fix()
print(m)


m.optimize_restarts(num_restarts=3, max_iters=1000, messages=True, ipython_notebook=False)
plot_integral(ax_c_ib, m, "Calibrated Model: $Int\_Brown$")


k = GPy.kern.Brownian(variance=0.1) #+ GPy.kern.Bias(input_dim=1, variance=0.1)
x_train = ((train[:,1]+train[:,0])/2)[:,None]
m = GPy.models.GPRegression(x_train, y_train[:,None], k, noise_var=0.0)
m.Gaussian_noise.variance.fix()
print(m)
plot_integral(axs_uc[0], m, "Uncalibrated Model: $Brown$")

m.optimize_restarts(num_restarts=5, max_iters=1000, messages=True, ipython_notebook=False)
print(m)
plot_integral(ax_c_b, m, "Calibrated Model: $Brown$")

save(fig=fig_c_b, name="Brown", path="/home/bela/Cloud/Arbeit/KIT/Planed_Paper/estimation under brownean drift/fig/exp_figures")

save(fig=fig_c_ib, name="Int_Brown", path="/home/bela/Cloud/Arbeit/KIT/Planed_Paper/estimation under brownean drift/fig/exp_figures")


#fig_c.tight_layout()
#plt.show()


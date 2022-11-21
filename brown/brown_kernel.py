import GPy
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

std = 0.1
stop_time = 25

nr_plot_points = 1000
number_of_train_points = 100

n_samples = 5 #Number of function realizations

rng = np.random.RandomState(None)


t_train = np.linspace(0, stop_time, num=number_of_train_points+1)

t_train_low = t_train[:-1]
t_train_up = t_train[1:]

noise_train =  rng.normal(0, t_train_low*std)
y_train = noise_train

train = np.concatenate((t_train_low[:,None], t_train_up[:,None]), axis=1)


import numpy as np
from GPy.kern import Kern
from GPy.core.parameterization import Param
from paramz.transformations import Logexp
import math



def plot_integral(gp: GPy.models.GPRegression, title='Estimated Model'):
    Xtest = np.linspace(0, stop_time*1.2, num=nr_plot_points+1)
    Xpred = np.array([Xtest[:-1],Xtest[1:]])
    Ypred,YpredCov = gp.predict_noiseless(Xpred.T)
    SE = np.sqrt(YpredCov)

    plt.scatter((t_train_up + t_train_low)/2, y_train,label='GT Data')
    plt.plot((Xpred[1]+Xpred[0])/2, Ypred,'r-',label='Mean')
    plt.plot((Xpred[1]+Xpred[0])/2,Ypred+SE*1.96,'r:',label='95% CI')
    plt.plot((Xpred[1]+Xpred[0])/2,Ypred-SE*1.96,'r:')
    plt.title(title)
    plt.xlabel('time')
    plt.ylabel('VS')
    plt.ylim(-7,7)
    plt.legend()


k = GPy.kern.Brownian(variance=0.1) + GPy.kern.Bias(input_dim=1, variance=0.1)
x_train = ((train[:,1]+train[:,0])/2)[:,None]
m = GPy.models.GPRegression(x_train, y_train[:,None], k, noise_var=0.001)
m.Gaussian_noise.variance.fix()
print(m)
plot_integral(m, "Uncalibrated Model")
plt.show()

m.optimize_restarts(num_restarts=5, max_iters=1000, messages=True, ipython_notebook=False)
print(m)
plot_integral(m,"Calibrated Model")
plt.show()




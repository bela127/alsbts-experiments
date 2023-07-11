# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 18:07:29 2022

@author: pl411
"""

import numpy as np
from copy import deepcopy
import GPy
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from GPy.kern import Kern
from GPy.core.parameterization import Param
from paramz.transformations import Logexp
        

#self-explanatory
def load_profile(string, start, length):
    
    df = np.genfromtxt(string, delimiter= ';')
    lp=np.zeros(length)
    for j in range(0,length):
        lp[j]=df[j+start][2]
    
    #the data is given as total energy consumption during that minute in kWh. 
    #multiplying with 60 yields the power
    
    return lp


def binning(lp, offset=0, size=15):
    #aim of function: binning of load profile in bins of size b
    #value w = shift, as measurements do not have to take place at the same times, 
    #only at same frequency
    #array's first dim: value, second dim: error (used later)

    S = np.arange(offset, lp.shape[0], size+1, dtype=np.int64)
    T = np.empty_like(S)
    bined_lp = np.empty_like(S,dtype=np.float64)
    for i, s in enumerate(S):
        t = s+size
        T[i] = t
        bin_mean = np.mean(lp[s:t])
        bined_lp[i] = bin_mean

    return bined_lp, S, T

def expand_bins(bined, S, T):
    up = np.max(T)
    low = np.min(S)
    exp_bins = np.empty((int(up-low)))
    exp_t = np.empty((int(up-low)))
    index = 0
    for bin, t in zip(bined, T):
        while index < t:
            exp_bins[index] = bin
            exp_t[index] = low + index
            index += 1
    return exp_bins, exp_t


          

nr_plot_points = 1000

rng = np.random.RandomState(None)


start_t = 5
#                      min*hour*day
end_t = 60*24*1


lp = load_profile("/home/bela/Cloud/code/Git/alvsts-experiments/brown/LP_H_2Erw_2Ki_wkB_mB_15-21_03_21_1.csv", start_t, end_t)
agre_lp, t_train_low, t_train_up,  = binning(lp)
exp_agr_lp, exp_time = expand_bins(agre_lp, t_train_low, t_train_up)

plt.scatter((t_train_low + t_train_up)/2, agre_lp, c="red")
plt.plot(np.arange(lp.shape[0]) ,lp)
plt.plot(exp_time, exp_agr_lp)
plt.show()


y_train = agre_lp

train = np.concatenate((t_train_low[:,None], t_train_up[:,None]), axis=1)



class IntegralBrown(Kern): 
    """
    Integral kernel. This kernel allows 1d histogram or binned data to be modelled.
    The outputs are the counts in each bin. The inputs (on two dimensions) are the start and end points of each bin.
    The kernel's predictions are the latent function which might have generated those binned results.
    """

    def __init__(self, input_dim = 2, variance=1.0, ARD=False, active_dims=None, name='integral'):
        """
        """
        super(IntegralBrown, self).__init__(input_dim, active_dims, name)
 
        self.variance = Param('variance', variance, Logexp()) #Logexp - transforms to allow positive only values...
        self.link_parameters(self.variance) #this just takes a list of parameters we need to optimise.


    def update_gradients_full(self, dL_dK, X, X2=None):
        if X2 is None:  #we're finding dK_xx/dTheta
            dK_dv = np.zeros([X.shape[0],X.shape[0]])
            for i,x in enumerate(X):
                for j,x2 in enumerate(X):
                    dK_dv[i,j] = self.k_FF(x[1],x2[1],x[0],x2[0])  #the gradient wrt the variance is k_xx.
            self.variance.gradient = np.sum(dK_dv * dL_dK)
        else:     #we're finding dK_xf/Dtheta
            raise NotImplementedError("Currently this function only handles finding the gradient of a single vector of inputs (X) not a pair of vectors (X and X2)")


    def xx(self, yl,yu,xl,xu):
        return 1/2*xu**2*yu-1/2*xl**2*yu-1/2*xu**2*yl+1/2*xl**2*yl

    def yy(self, yl,yu,xl,xu):
        return self.xx(xl,xu, yl,yu)
    
    def xy(self, l,u):
        #1/4 pyramid + the below cuboid
        return 1/3*(u-l)**3+(u-l)**2*l

    def x(self, x, l, u):
        return x*u-x*l
    
    def y(self, l, u):
        return 1/2*u**2-1/2*l**2

    def k_FF(self,t,tprime,s,sprime):
        """Covariance between observed values.

        s and t are one domain of the integral (i.e. the integral between s and t)
        sprime and tprime are another domain of the integral (i.e. the integral between sprime and tprime)

        We're interested in how correlated these two integrals are.

        Note: We've not multiplied by the variance, this is done in K."""

        if s <= sprime < tprime <= t:
            xx = self.xx(tprime, t, sprime, tprime)
            xy = self.xy(sprime,tprime)
            yy= self.yy(s,sprime,sprime,tprime)
            i = xx + xy + yy
        elif sprime < tprime <= s < t:
            i = self.xx(s,t,sprime,tprime)
        elif sprime <= s < tprime <= t:
            xx = self.xx(s, t, sprime, s) + self.xx(tprime, t, s, tprime)
            xy = self.xy(s, tprime)
            i = xx + xy
        elif s <= sprime < t <= tprime:
            yy = self.yy(s, sprime,sprime,tprime) + self.yy(sprime, t, t, tprime)
            xy = self.xy(sprime, t)
            i = yy + xy
        elif s < t <= sprime < tprime:
            i = self.yy(s, t, sprime, tprime)
        elif sprime <= s < t <= tprime:
            xx = self.xx(s, t, sprime, s)
            xy = self.xy(s,t)
            yy = self.yy(s,t, t, tprime)
            i= xx + xy + yy
        else:
            raise RuntimeError(f"This should never happen, i guess i should check the code, please report: {s,t,sprime, tprime}")
        return i


    def k_ff(self,x,y):
        """Doesn't need s or sprime as we're looking at the 'derivatives', so no domains over which to integrate are required"""
        return np.fmin(x,y)


    def k_Ff(self,t,x,s):
        """Covariance between the gradient (latent value) and the actual (observed) value.

        Note that sprime isn't actually used in this expression, presumably because the 'primes' are the gradient (latent) values which don't
        involve an integration, and thus there is no domain over which they're integrated, just a single value that we want."""
        #      First integral *-* + Second Integral *-*

        if x <= s < t:
            i = self.x(x, s, t)
        elif s < x < t:
            y = self.y(s,x)
            x = self.x(x,x,t)
            i= x + y
        elif s < t <= x:
            i = self.y(s,t)
        else:
            raise RuntimeError(f"This should never happen, i guess i should check the code, please report: {s,x,t}")
        return i


        #return 1/2*np.fmin(x,t)**2-1/2*np.fmin(s,x)**2 + np.fmin(x,t)*t-np.fmin(x,t)**2


    def K(self, X, X2=None):
        """Note: We have a latent function and an output function. We want to be able to find:
          - the covariance between values of the output function
          - the covariance between values of the latent function
          - the "cross covariance" between values of the output function and the latent function
        This method is used by GPy to either get the covariance between the outputs (K_xx) or
        is used to get the cross covariance (between the latent function and the outputs (K_xf).
        We take advantage of the places where this function is used:
         - if X2 is none, then we know that the items being compared (to get the covariance for)
         are going to be both from the OUTPUT FUNCTION.
         - if X2 is not none, then we know that the items being compared are from two different
         sets (the OUTPUT FUNCTION and the LATENT FUNCTION).
        
        If we want the covariance between values of the LATENT FUNCTION, we take advantage of
        the fact that we only need that when we do prediction, and this only calls Kdiag (not K).
        So the covariance between LATENT FUNCTIONS is available from Kdiag.        
        """
        if X2 is None:
            K_FF = np.zeros([X.shape[0],X.shape[0]])
            for i,x in enumerate(X):
                for j,x2 in enumerate(X):
                    K_FF[i,j] = self.k_FF(x[1],x2[1],x[0],x2[0])
            eigv = np.linalg.eig(K_FF)
            return K_FF * self.variance
        else:
            K_Ff = np.zeros([X.shape[0],X2.shape[0]])
            for i,x in enumerate(X):
                for j,x2 in enumerate(X2):
                    K_Ff[i,j] = self.k_Ff(x[1],x2[1],x[0]) #x2[1] unused, see k_xf docstring for explanation.
            return K_Ff * self.variance


    def Kdiag(self, X):
        """I've used the fact that we call this method during prediction (instead of K). When we
        do prediction we want to know the covariance between LATENT FUNCTIONS (K_ff) (as that's probably
        what the user wants).
        $K_{ff}^{post} = K_{ff} - K_{fx} K_{xx}^{-1} K_{xf}$"""
        K_ff = np.zeros(X.shape[0])
        for i,x in enumerate(X):
            K_ff[i] = self.k_ff(x[1],x[1])
        return K_ff * self.variance


def plot_integral(gp: GPy.models.GPRegression):
    Xtest = np.linspace(0, end_t*1.2, num=nr_plot_points+1)
    Xpred = np.array([Xtest[:-1],Xtest[1:]])
    Ypred,YpredCov = gp.predict_noiseless(Xpred.T)
    SE = np.sqrt(YpredCov)

    plt.scatter((t_train_low + t_train_up)/2, agre_lp, c="red",label='Data')
    plt.plot(np.arange(lp.shape[0]) ,lp,label='GT')
    plt.plot(exp_time, exp_agr_lp)

    plt.plot((Xpred[1]+Xpred[0])/2, Ypred,'r-',label='Mean')
    plt.plot((Xpred[1]+Xpred[0])/2,Ypred+SE*1.96,'r:',label='95% CI')
    plt.plot((Xpred[1]+Xpred[0])/2,Ypred-SE*1.96,'r:')
    plt.title('Estimated Model')
    plt.xlabel('time')
    plt.ylabel('y')

res = y_train[:,None]* (t_train_up - t_train_low)
k = IntegralBrown(variance=1)
m = GPy.models.GPRegression(train, res, k, noise_var=0.0001)


print(m)
plot_integral(m)
plt.show()

m.Gaussian_noise.variance.fix()
print(m)


m.optimize_restarts(num_restarts=5, max_iters=1000, messages=True, ipython_notebook=False)
print(m)

plot_integral(m)
plt.show()





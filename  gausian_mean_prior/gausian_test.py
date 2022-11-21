from statistics import variance
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from sklearn.preprocessing import scale

nr_samples = 1000
sample_size = 10
samples = np.random.normal(size=(nr_samples, sample_size))

means = np.mean(samples, axis=1)
stds = np.std(samples, axis=1)


x = np.linspace(norm.ppf(0.01),norm.ppf(0.99), 1000)
x = x[:,None]
densities = norm.pdf(x, loc=means[None,:], scale=stds[None,:])

plt.plot(x, densities)
plt.show()

mean_mean = np.mean(means)
std_std = np.std(stds)

densities = norm.pdf(x, loc=mean_mean, scale=std_std)

plt.hist(means, 50, density=True)
plt.plot(x, densities)
plt.show()
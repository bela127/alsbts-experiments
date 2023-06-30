import numpy as np

nr_samples = 100000
random_times = np.random.uniform(0,1000,(nr_samples, 200),)

sorted_times = np.sort(random_times, axis=1)

dists = np.abs(sorted_times[:,1:] - sorted_times[:,:-1])
mins = np.min(dists, axis=-1)
maxs = np.max(dists, axis=-1)
mean = np.mean(dists, axis=-1)
std = np.std(dists, axis=-1)

print(np.mean(mean), np.mean(std), min(mins), max(maxs))
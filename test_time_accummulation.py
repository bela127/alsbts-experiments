import numpy as np
times = np.arange(2,10)
time_interval = 3
last_querry_time = 1

print(times)
lq_times = (times - last_querry_time) % time_interval
print(lq_times)

mask = lq_times[:-1] >= lq_times[1:]
mask = np.concatenate((np.asarray([times[0] - last_querry_time >= time_interval]),mask))
print(mask)

last_querry_time = times[mask][-1]

#-----
times = np.arange(10,20)
print(times)

lq_times = (times - last_querry_time) % time_interval
print(lq_times)

mask = lq_times[:-1] >= lq_times[1:]
mask = np.concatenate((np.asarray([times[0] - last_querry_time >= time_interval]),mask))
print(mask)
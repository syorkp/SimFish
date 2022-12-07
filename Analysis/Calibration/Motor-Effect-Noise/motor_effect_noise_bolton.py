import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import seaborn as sns
import statsmodels as sm
import scipy.stats as stats

file = np.loadtxt(open("Bolton et al. (2019)/all_huntbouts_source.csv", "rb"), delimiter=",", skiprows=1)

dist = file[:, 5]
delta_yaw = file[:, 7]
bout_az = file[:, 3]

para_dist = file[:, 10]
para_az = file[:, 8]
postbout_para_dist = file[:, 19]

para_az = para_az * (180/np.pi)
delta_yaw = delta_yaw * (180/np.pi)
bout_az *= (180/np.pi)


plt.scatter(para_az, bout_az, alpha=0.01)
plt.show()

plt.scatter(para_az, delta_yaw, alpha=0.01)
plt.show()

plt.scatter(para_dist, postbout_para_dist)
plt.show()


# Model
model = np.polyfit(para_az, delta_yaw, 1)
p = np.poly1d(model)

# Estimate Residual standard deviation


predictions = p(para_az)
differences = delta_yaw - predictions
bins = range(-100, 100, 20)
new_bins = []

corresponding_sds = []

for bin in bins:
    indexes_within_bin = (bin < para_az) * (para_az < bin + 20)

    d = differences[indexes_within_bin]
    p = predictions[indexes_within_bin]
    new_bins.append(np.mean(delta_yaw[indexes_within_bin]))

    ressd = (np.sum(p ** 2)/(len(d)-1)) ** 0.5

    corresponding_sds.append(ressd)

plt.scatter(bins, corresponding_sds)
plt.xlabel("Prey azimuth (degrees)")
plt.ylabel("Bout angle std (degrees)")
plt.show()

plt.scatter(np.absolute(new_bins), np.absolute(corresponding_sds))
z = np.polyfit(np.absolute(new_bins), corresponding_sds, 1)
print(z)
p = np.poly1d(z)
bins_to_check = np.linspace(0, 50, 10)
plt.plot(bins_to_check, p(bins_to_check), color="r")
plt.xlabel("Mean bout angle (degrees)")
plt.ylabel("Bout angle std (degrees)")
plt.show()
# ax = sns.regplot(x=para_az, y=delta_yaw)#, x_ci="sd")
# plt.show()


# Finding only appropriate distances
# converting to distances from pixels
para_dist *= 0.0106
dist *= 0.0106

prey_close = (para_dist < 6) * (dist < 3)

para_distances_reduced = para_dist[prey_close]
dist_reduced = dist[prey_close]

model = np.polyfit(para_distances_reduced, dist_reduced, 5)
p = np.poly1d(model)

plt.scatter(para_distances_reduced, dist_reduced, alpha=0.1)
plt.plot(np.linspace(0, 6, 100), p(np.linspace(0, 6, 100)), color="r")
plt.show()


predictions = p(para_distances_reduced)
differences = para_distances_reduced - predictions
bins = range(0, 6)
new_bins = []

corresponding_sds = []

for bin in bins:
    indexes_within_bin = (bin < para_distances_reduced) * (para_distances_reduced < bin + 20)

    d = differences[indexes_within_bin]
    p = predictions[indexes_within_bin]
    mean_bout_distance = np.mean(dist_reduced[indexes_within_bin])
    new_bins.append(mean_bout_distance)

    ressd = (np.sum(p ** 2)/(len(d)-1)) ** 0.5

    corresponding_sds.append(ressd)

plt.scatter(bins, corresponding_sds)
plt.xlabel("Prey distance (mm)")
plt.ylabel("Bout distance std (mm)")
plt.show()

plt.scatter(new_bins, corresponding_sds)
z = np.polyfit(new_bins, corresponding_sds, 1)
p = np.poly1d(z)
print(z)

bins_to_check = np.linspace(0, 1, 10)
plt.plot(bins_to_check, p(bins_to_check), color="r")
plt.xlabel("Mean bout distance (mm)")
plt.ylabel("Bout distance std (mm)")
plt.show()

x = True








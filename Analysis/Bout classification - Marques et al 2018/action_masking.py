import numpy as np
import scipy.io
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from scipy.interpolate import griddata
import matplotlib
import h5py
# import tensorflow.compat.v1 as tf
# import tensorflow_probability as tfp
from sklearn.cluster import DBSCAN
import statsmodels.api as sm
import seaborn as sns

# tf.disable_v2_behavior()

# f = h5py.File('BoutMapCenters_kNN4_74Kins4dims_1.75Smooth_slow_3000_auto_4roc_merged11.mat', 'r')
# data = f.get('data/variable1')
# data = np.array(data) # For converting to a NumPy array

"""
1) The kinematic parameters are in the "BoutKinematicParametersFinalArray" array. It has 240 columns. We used 74 of 
these to create the PCA space and to cluster the bout types. 

2) The "EnumeratorBoutKinPar.m" enumerator tells which column corresponds to which kinematic parameter.

3) The bout categorization is in the "BoutInfFinalArray". This array has information about the bouts that are not 
kinematic parameters. It also has its enumerator, called "EnumeratorBoutInf.m". The bout categorization is in column 
134.

4) Each bout belongs to one of 13 types (1-13). Since I've done different categorizations over the years I use a vector 
that reorders the bout types always in the same order (idx = finalClustering.idx). The colors I use in the paper for 
each bout types are in: col = finalClustering.col follow the same order as in "idx". Anyway, for this categorization 
the ordering of the bouts is:

Important indices (subtract 1?):
  - boutAngle = 10 Angle turned during tail motion (degrees, clockwise negative)
  - distBoutAngle = 11 Angle turned during bout including glide (degrees, clockwise negative)
  - boutMaxAngle = 12 Maximum angle turned during bout (degrees, absolute value)
  - boutDistanceX = 15 Distance advanced in the tail to head direction during tail motion (forward positive, mm)
  - boutDistanceY = 16 Distance moved in left right direction during tail motion (left positive, mm)
  - distBoutDistanceX = 19 Distance advanced in the tail to head direction including glide (forward positive, mm)
  - distBoutDistanceY = 20 Distance moved in left right direction including glide (left positive, mm)
"""

mat = scipy.io.loadmat("bouts.mat")
bout_kinematic_parameters_final_array = mat["BoutKinematicParametersFinalArray"]
angles = bout_kinematic_parameters_final_array[:, 9]
dist_angles = bout_kinematic_parameters_final_array[:, 10]  # This one
max_angles = bout_kinematic_parameters_final_array[:, 11]
distance_x = bout_kinematic_parameters_final_array[:, 14]
distance_y = bout_kinematic_parameters_final_array[:, 15]
distance_x_inc_glide = bout_kinematic_parameters_final_array[:, 18]
distance_y_inc_glide = bout_kinematic_parameters_final_array[:, 19]


# plt.hist(max_angles, bins=1000)
# plt.show()
#
# plt.hist(distance_x, bins=1000)
# plt.show()
#
# plt.hist(distance_y, bins=1000)
# plt.show()

# Verification that these are the desired columns:
# plt.scatter(distance_x, distance_x_inc_glide)
# plt.show()
#
# plt.scatter(distance_y, distance_y_inc_glide)
# plt.show()


# Want actual distance moved - combination of both.
distance = (distance_x_inc_glide**2 + distance_y_inc_glide**2)**0.5

# Plot distance against angles in heatmap to get idea of distribution.
# plt.scatter(np.absolute(angles), distance, alpha=0.2)
# plt.xlabel("Angle (degrees)")
# plt.ylabel("Distance moved (mm)")
# plt.show()
#
# plt.scatter(np.absolute(dist_angles), distance, alpha=0.2)
# plt.xlabel("Full Angle (degrees)")
# plt.ylabel("Distance moved (mm)")
# plt.show()
#
# # Convert impulse for specific mass
impulse = (distance * 10 - (0.004644 * 140.0 + 0.081417)) / 1.771548
dist_angles_radians = (np.absolute(dist_angles)/180) * np.pi
# plt.scatter(dist_angles_radians, impulse, alpha=0.2)
# plt.xlabel("Full Angle (Radians)")
# plt.ylabel("Impulse")
# plt.show()

# Computing kernel density
# ang, imp = np.linspace(0, 500, 500), np.linspace(0, 20, 500)
# ang, imp = np.meshgrid(ang, imp)
# ang, imp = np.expand_dims(ang, 2), np.expand_dims(imp, 2)
# action_range = np.concatenate((ang, imp), axis=2)
# action_range = action_range.reshape(-1, action_range.shape[-1])
#
impulse = np.expand_dims(impulse, 1)
dist_angles_radians = np.expand_dims(dist_angles_radians, 1)
actions = np.concatenate((impulse, dist_angles_radians), axis=1)
kde = KernelDensity(bandwidth=20, kernel='gaussian').fit(actions)
log_density_of_original = kde.score_samples(actions)
params = kde.get_params()

#             OUTLIER IDENTIFICATION

# Z-score based
# z = np.abs(stats.zscore(actions, axis=None))
# # z = z[:, 0] + z[:, 1]
# outliers = actions[z[:, 0]>1.5 or z[:, 1]>1.5]


# for i, s in enumerate(range(1, 20)):
#     figure, axs = plt.subplots(3, 3)
#     figure.set_size_inches(18, 18)
#     for j, ep in enumerate(np.linspace(0.5, 1.0, 9)):
#         model = DBSCAN(eps=ep, min_samples=s).fit(actions)
#         colors = model.labels_
#
#         axs[j//3, j%3].scatter(actions[:, 0], actions[:, 1], c=colors, alpha=0.9)
#         # plt.scatter(outliers[:, 0], outliers[:, 1], alpha=0.2, color="r")
#         axs[j//3, j%3].set_ylabel("Full Angle (Radians)")
#         axs[j//3, j%3].set_xlabel(f"Impulse,Samples: {s}, eps: {ep}")
#     plt.show()


# Best parameters; 1.0, 5    -    0.775, 4   -    5, 0.6625       -      5, 0.8125
model = DBSCAN(eps=0.8125, min_samples=5).fit(actions)
colors = model.labels_
moutliers = actions[model.labels_ == -1]
plt.scatter(actions[:, 0], actions[:, 1], c=colors, alpha=0.9)
plt.show()

plt.scatter(actions[:, 0], actions[:, 1], alpha=0.3)
plt.scatter(moutliers[:, 0], moutliers[:, 1], color="r", alpha=0.3)
plt.show()

sorted_actions = actions[model.labels_ != -1]
plt.scatter(sorted_actions[:, 0], sorted_actions[:, 1], color="r", alpha=0.3)
plt.show()

# Extra step - cut off negative impulse values
sorted_actions = sorted_actions[sorted_actions[:, 0] >= 0]
sorted_actions = sorted_actions[sorted_actions[:, 1] >= 0]

#                      Final KDE Formation

kde_sorted = KernelDensity(bandwidth=20, kernel='exponential').fit(sorted_actions)

log_density_of_original = kde_sorted.score_samples(actions)
#
# xi = np.linspace(np.min(actions[:, 0]), np.max(actions[:, 0]), 100)
# yi = np.linspace(np.min(actions[:, 1]), np.max(actions[:, 1]), 100)
# xi, yi = np.meshgrid(xi, yi)
# zi = griddata(actions, log_density_of_original, (xi, yi), method="linear")
# plt.pcolormesh(xi, yi, zi)
# plt.show()

plt.scatter(impulse[:, 0], np.exp(log_density_of_original))
plt.show()

plt.scatter(dist_angles_radians[:, 0], np.exp(log_density_of_original))
plt.show()

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
ax.scatter(impulse[:, 0], dist_angles_radians[:, 0], np.exp(log_density_of_original))
plt.show()


# Jointplot

# action_data = {"x": actions[:, 0], "y": actions[:, 1]}
# g = sns.jointplot(x="x", y="y", data=action_data, kind="kde")
#
# g.plot_joint(plt.scatter, c="w")
# g.ax_joint.collections[0].set_alpha(0)
#
# plt.show()

# KDF 2

bw_ml_x = sm.nonparametric.KDEMultivariate(data=sorted_actions[:, 0], var_type='c', bw='cv_ml')
bw_ml_y = sm.nonparametric.KDEMultivariate(data=sorted_actions[:, 1], var_type='c', bw='cv_ml')
probs = bw_ml_x.pdf(actions[:, 0]) * bw_ml_y.pdf(actions[:, 1])
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
ax.scatter(impulse[:, 0], dist_angles_radians[:, 0], probs)
plt.show()



class TFTest:

    def __init__(self, distance_x_inc_glide, distance_y_inc_glide, dist_angles):
        distance = (distance_x_inc_glide ** 2 + distance_y_inc_glide ** 2) ** 0.5

        impulse = (distance * 10 - (0.004644 * 140.0 + 0.081417)) / 1.771548
        dist_angles_radians = (np.absolute(dist_angles) / 180) * np.pi

        impulse = np.expand_dims(impulse, 1)
        dist_angles_radians = np.expand_dims(dist_angles_radians, 1)
        actions = np.concatenate((impulse, dist_angles_radians), axis=1)
        self.kde = KernelDensity(bandwidth=20, kernel='gaussian').fit(actions)

    def value_through_kdf(self, x):
        log_prob = self.kde.score(x)
        prob = np.exp(log_prob)

# sess = tf.Session()
# with sess as sess:
#
#     init = tf.global_variables_initializer()
#     trainables = tf.trainable_variables()
#     sess.run(init)
# x = True

# plt.scatter(impulse[:, 0], log_density_of_original)
# plt.show()
#
# plt.scatter(dist_angles_radians[:, 0], log_density_of_original)
# plt.show()
#
#
# # # Visualise KDF for all original data points
# log_density_of_original = kde.score_samples(actions)
# #
# xi = np.linspace(np.min(actions[:, 0]),np.max(actions[:, 0]),100)
# yi = np.linspace(np.min(actions[:, 1]),np.max(actions[:, 1]),100)
# xi, yi = np.meshgrid(xi, yi)
# zi = griddata(actions, log_density_of_original, (xi, yi), method="linear")
# plt.pcolormesh(xi, yi, zi)
# plt.show()
#
# # Visualise KDF for larger range
# log_density = kde.score_samples(action_range)
#
# xi = np.linspace(np.min(action_range[:, 0]),np.max(action_range[:, 0]),100)
# yi = np.linspace(np.min(action_range[:, 1]),np.max(action_range[:, 1]),100)
# xi, yi = np.meshgrid(xi, yi)
# zi = griddata(action_range, log_density, (xi, yi), method="linear")
# plt.pcolormesh(xi, yi, zi)
# plt.show()
#


#                         TENSORFLOW Cutoff



#                         TENSORFLOW KDE




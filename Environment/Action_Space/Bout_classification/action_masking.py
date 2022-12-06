import copy
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
import scipy.stats as st

from Analysis.Training.tools import find_nearest
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


def produce_action_mask():
    res = 50   # Need to ensure resolution isn't too high as to drag the final mask down.
    kde, valid_bouts = get_action_mask()

    # Display KDE evaluation grid
    impulse_range = np.linspace(0, 50, res)
    angle_range = np.linspace(0, 5, res)
    X, Y = np.meshgrid(impulse_range, angle_range)
    X_, Y_ = np.expand_dims(X, 2), np.expand_dims(Y, 2)
    full_grid = np.concatenate((X_, Y_), axis=2).reshape((-1, 2))
    full_grid = np.swapaxes(full_grid, 0, 1)

    # Evaluate KDE Over Grid
    values = kde(full_grid)
    pdf = np.reshape(values, (res, res))
    pdf /= np.max(pdf)
    pdf = np.flip(pdf, 0)

    # Find which points in the grid should be considered true by found threshold
    nearest_i = np.array([find_nearest(impulse_range, v[0]) for v in valid_bouts])
    nearest_a = np.array([find_nearest(angle_range, v[1]) for v in valid_bouts])

    impulses = impulse_range[nearest_i]
    angles = angle_range[nearest_a]

    correct_bout_pairs = copy.copy(X)
    correct_bout_pairs[:, :] = False
    correct_bout_pairs[nearest_i, nearest_a] = True

    correct_bout_pairs[correct_bout_pairs == 0] = -1
    correct_bout_pairs = np.flip(correct_bout_pairs, 0)

    best_threshold = [0, 0]
    # Iterate over thresholds, determining how many points have successfully matched.
    for thres in np.linspace(0, 1, 100000):
        above_thresold = (pdf > thres)
        correctly_identified = (above_thresold * correct_bout_pairs) * 1
        print(f"Threshold: {thres}, Correct: {np.sum(correctly_identified)}")
        if np.sum(correctly_identified) > best_threshold[1]:
            best_threshold[0] = thres
            best_threshold[1] = np.sum(correctly_identified)

    print(best_threshold)

    plt.imshow(pdf, extent=[0, 50, 0, 5])
    plt.scatter(impulses, angles, alpha=0.01)
    plt.savefig("PDF vals 2.png")
    plt.clf()
    plt.close()

    plt.imshow(pdf, extent=[0, 50, 0, 5])
    plt.savefig("PDF vals.png")
    plt.clf()
    plt.close()

    pdf[pdf > best_threshold[0]] = 1
    pdf[pdf < 1] = 0
    plt.imshow(pdf,  extent=[0, 50, 0, 5])
    plt.savefig("PDF vals 3.png")
    plt.clf()
    plt.close()

def get_action_mask():
    try:
        mat = scipy.io.loadmat("./Environment/Action_Space/Bout_classification/bouts.mat")
    except FileNotFoundError:
        try:
            mat = scipy.io.loadmat("../../Environment/Action_Space/Bout_classification/bouts.mat")
        except FileNotFoundError:
            mat = scipy.io.loadmat("../../../Environment/Action_Space/Bout_classification/bouts.mat")

    bout_kinematic_parameters_final_array = mat["BoutKinematicParametersFinalArray"]
    dist_angles = bout_kinematic_parameters_final_array[:, 10]  # Angles including glide
    distance_x_inc_glide = bout_kinematic_parameters_final_array[:, 18]  # In mm
    distance_y_inc_glide = bout_kinematic_parameters_final_array[:, 19]  # In mm

    distance = (distance_x_inc_glide ** 2 + distance_y_inc_glide ** 2) ** 0.5

    impulse = distance * 10 * 0.34452532909386484
    dist_angles_radians = (np.absolute(dist_angles) / 180) * np.pi

    impulse = np.expand_dims(impulse, 1)
    dist_angles_radians = np.expand_dims(dist_angles_radians, 1)

    actions = np.concatenate((impulse, dist_angles_radians), axis=1)

    model = DBSCAN(eps=0.8125, min_samples=5).fit(actions)
    sorted_actions = actions[model.labels_ != -1]

    # Extra step - cut off negative impulse values
    sorted_actions = sorted_actions[sorted_actions[:, 0] >= 0]
    sorted_actions = sorted_actions[sorted_actions[:, 1] >= 0]

    print("Creating action mask...")
    sorted_actions = np.swapaxes(sorted_actions, 0, 1)
    kernel = st.gaussian_kde(sorted_actions)
    print("Action mask created")

    sorted_actions = np.swapaxes(sorted_actions, 0, 1)

    return kernel, sorted_actions


def create_action_mask_old():
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
    distance = (distance_x_inc_glide ** 2 + distance_y_inc_glide ** 2) ** 0.5

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
    dist_angles_radians = (np.absolute(dist_angles) / 180) * np.pi
    plt.scatter(dist_angles_radians, impulse, alpha=0.2)
    plt.xlabel("Angle (radians)")
    plt.ylabel("Impulse")
    plt.show()

    # Scale as scaled in PPO
    # impulse = impulse#/10.0
    # dist_angles_radians = dist_angles_radians# / (np.pi/5)

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

    # plt.scatter(actions[:, 0], actions[:, 1], c=colors, alpha=0.9)
    # plt.show()
    #
    plt.scatter(actions[:, 1], actions[:, 0], alpha=0.3)
    plt.scatter(moutliers[:, 1], moutliers[:, 0], color="r", alpha=0.3)
    plt.xlabel("Angle (radians)")
    plt.ylabel("Impulse")
    plt.show()
    #
    sorted_actions = actions[model.labels_ != -1]
    # plt.scatter(sorted_actions[:, 0], sorted_actions[:, 1], color="r", alpha=0.3)
    # plt.show()

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

    # plt.scatter(impulse[:, 0], np.exp(log_density_of_original))
    # plt.show()
    #
    # plt.scatter(dist_angles_radians[:, 0], np.exp(log_density_of_original))
    # plt.show()
    #
    # fig = plt.figure(figsize=(12, 12))
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(impulse[:, 0], dist_angles_radians[:, 0], np.exp(log_density_of_original))
    # plt.show()

    # Jointplot

    # action_data = {"x": actions[:, 0], "y": actions[:, 1]}
    # g = sns.jointplot(x="x", y="y", data=action_data, kind="kde")
    #
    # g.plot_joint(plt.scatter, c="w")
    # g.ax_joint.collections[0].set_alpha(0)
    #
    # plt.show()

    # KDF 2
    indices_of_valid = (model.labels_ != -1) * 1
    sorted_actions = np.array([sorted_actions])
    actions = np.array([actions])
    bw_ml_x = sm.nonparametric.KDEMultivariate(data=sorted_actions[:, :, 0], var_type='c', bw='cv_ml')
    bw_ml_y = sm.nonparametric.KDEMultivariate(data=sorted_actions[:, :, 1], var_type='c', bw='cv_ml')
    probs = bw_ml_x.pdf(actions[:, :, 0]) * bw_ml_y.pdf(actions[:, :, 1])
    probs2 = copy.copy(probs)

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(dist_angles_radians[:, 0], impulse[:, 0], probs)
    ax.set_ylabel("Impulse")
    ax.set_xlabel("Angle (radians)")
    ax.set_zlabel("Probability density")
    plt.show()

    # Thresholding                                       Best: [2.3733733733733735e-05, 8183]
    # print(f"Total inliers: {np.sum(indices_of_valid)}")
    # current_best = [0, 0]
    # for threshold in np.linspace(0.000005, 0.0001, 1000):
    #     probs2[probs < threshold] = 0
    #     probs2[probs > threshold] = 1
    #     match = np.sum((probs2 == indices_of_valid) * 1)
    #     if match > current_best[1]:
    #         current_best = [threshold, match]
    #         print(f"Computed inliers for {threshold}: {match}")
    #
    #     if np.array_equal(indices_of_valid, probs2):
    #         print(f"FOUND Threshold {threshold}")
    # #
    # print(f"Best: {current_best}")
    #
    # probs2[probs < 0.0000729] = 0
    # probs2[probs > 0.0000729] = 1
    # sames = (probs2 == indices_of_valid) * 1
    # print(np.sum(sames))
    # print(actions.shape[0])
    # # Multiply by integral.
    #
    # plt.scatter(impulse[:, 0], probs2)
    # plt.show()
    #
    # plt.scatter(dist_angles_radians[:, 0], probs2)
    # plt.show()
    #
    # fig = plt.figure(figsize=(12, 12))
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(impulse[:, 0], dist_angles_radians[:, 0], probs2)
    # plt.show()

    #                     Integrating distribution

    # Number of should have samples = 12 x 350
    possible_impulse = np.linspace(0, 350, 1000)  # Divide total by 100
    possible_angle = np.linspace(0, 12, 1000)  # Divide total by 100
    possible_impulse, possible_angle = np.meshgrid(possible_angle, possible_impulse)
    possible_impulse, possible_angle = possible_impulse.flatten(), possible_angle.flatten()

    # probs = bw_ml_x.pdf(possible_angle) * bw_ml_y.pdf(possible_impulse)

    probs2 = copy.copy(probs)
    t = 0.01587368

    for threshold in np.linspace(t, 0.1, 20):
        # probs2[probs < threshold] = 0
        # probs2[probs > threshold] = 1
        #
        # fig = plt.figure(figsize=(12, 12))
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(dist_angles_radians[:, 0], impulse[:, 0], probs2)
        # ax.set_ylabel("Impulse")
        # ax.set_xlabel("Angle (radians)")
        # ax.set_zlabel("Probability density")
        # plt.title(str(threshold))
        # plt.show()
        # print(np.sum(probs2))
        plt.scatter(dist_angles_radians[probs > threshold][:, 0], impulse[probs > threshold][:, 0])
        plt.show()

    x = True

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


if __name__ == "__main__":
    produce_action_mask()


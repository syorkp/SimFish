import numpy as np
import scipy.io
from sklearn.cluster import DBSCAN
import scipy.stats as st
import statsmodels.api as sm
import tensorflow_probability as tfp
import tensorflow.compat.v1 as tf

# tf.disable_v2_behavior()


class MaskedMultivariateNormal(tfp.distributions.MultivariateNormalDiag):

    def __init__(self, loc, scale_diag, impulse_scaling, angle_scaling):
        super().__init__(loc=loc, scale_diag=scale_diag, allow_nan_stats=False)

        self.mu_vals = loc
        self.kdf_threshold = 0.0014800148001480014  # As determined in Environment/Action_Space/Bout_classification/action_masking.py

        # Compute KDF here.
        try:
            mat = scipy.io.loadmat("./Environment/Action_Space/Bout_classification/bouts.mat")
        except FileNotFoundError:
            mat = scipy.io.loadmat("../../Environment/Action_Space/Bout_classification/bouts.mat")

        bout_kinematic_parameters_final_array = mat["BoutKinematicParametersFinalArray"]
        dist_angles = bout_kinematic_parameters_final_array[:, 10]  # Angles including glide
        distance_x_inc_glide = bout_kinematic_parameters_final_array[:, 18]  # In mm
        distance_y_inc_glide = bout_kinematic_parameters_final_array[:, 19]  # In mm

        distance = (distance_x_inc_glide ** 2 + distance_y_inc_glide ** 2) ** 0.5

        impulse = distance * 10 * 0.34452532909386484
        # impulse = (distance * 10 - (0.004644 * 140.0 + 0.081417)) / 1.771548
        dist_angles_radians = (np.absolute(dist_angles) / 180) * np.pi

        impulse = impulse / impulse_scaling

        dist_angles_radians = dist_angles_radians / angle_scaling

        impulse = np.expand_dims(impulse, 1)
        dist_angles_radians = np.expand_dims(dist_angles_radians, 1)
        actions = np.concatenate((impulse, dist_angles_radians), axis=1)

        model = DBSCAN(eps=0.8125, min_samples=5).fit(actions)
        sorted_actions = actions[model.labels_ != -1]

        # Extra step - cut off negative impulse values
        sorted_actions = sorted_actions[sorted_actions[:, 0] >= 0]
        sorted_actions = sorted_actions[sorted_actions[:, 1] >= 0]

        # self.kde_impulse = sm.nonparametric.KDEMultivariate(data=sorted_actions[:, 0], var_type='c', bw='cv_ml')
        # self.kde_angle = sm.nonparametric.KDEMultivariate(data=sorted_actions[:, 1], var_type='c', bw='cv_ml')
        print("Creating action mask...")
        # self.kde_impulse = sm.nonparametric.KDEMultivariate(data=sorted_actions[:, 0], var_type='c', bw='cv_ml')
        # self.kde_angle = sm.nonparametric.KDEMultivariate(data=sorted_actions[:, 1], var_type='c', bw='cv_ml')

        # New version requires swapping axes of inputs...
        sorted_actions = np.swapaxes(sorted_actions, 0, 1)
        self.kde = st.gaussian_kde(sorted_actions)
        print("Action mask created")

    def get_sample_masked_weights(self, actions, shape):
        if actions.shape[1] > 1:
            print("Action mask problem")
            print(actions.shape)


        actions = actions[:, 0, :]
        actions = np.swapaxes(actions, 0, 1)
        #probs = self.kde_impulse.pdf(actions[:, :, 0]) * self.kde_angle.pdf(np.absolute(actions[:, :, 1]))
        probs = self.kde(actions)
        actions = np.swapaxes(actions, 0, 1)
        # Dis-allow negative impulses
        # positive_impulses = ((actions[:, :, 0] >= 0) * 1)[:, 0]
        positive_impulses = ((actions[:, 0] >= 0) * 1)[0]
        probs = probs * positive_impulses
        probs = np.nan_to_num(probs)

        # # Step function on probs
        # probs[probs < 0.0000389489489] = 0.000001
        # probs[probs >= 0.0000389489489] = 1
        probs[probs < self.kdf_threshold] = 0.000001
        probs[probs >= self.kdf_threshold] = 1

        integral = np.sum(probs)

        probs = probs/integral

        indices_chosen = np.random.choice(actions.shape[0], size=shape, p=probs, replace=False)
        actions_chosen = actions[indices_chosen, :]
        actions_chosen = np.expand_dims(actions_chosen, 1)
        # return actions_chosen, probs, positive_impulses
        return actions_chosen

    def impose_zeroed(self, chosen_samples, mu_vals, threshold=0.01):
        zero_chosen = ((mu_vals < threshold) * 1) * ((mu_vals > -threshold) * 1)
        zero_chosen = (zero_chosen == 0) * 1.0
        chosen_samples *= zero_chosen
        return chosen_samples

    def sample_masked(self, shape):
        preliminary_samples = self.sample(shape * 100)
        # self.preliminary_samples = preliminary_samples
        # chosen_samples, self.probs, self.positive_imp = tf.numpy_function(self.get_sample_masked_weights, [self.preliminary_samples, shape], [tf.float32, tf.float64, tf.int64])
        chosen_samples = tf.numpy_function(self.get_sample_masked_weights, [preliminary_samples, shape], tf.float32)
        chosen_samples_thresholded = tf.numpy_function(self.impose_zeroed, [chosen_samples, self.mu_vals], tf.float32)
        return chosen_samples_thresholded


class MaskedMultivariateNormalOLD(tfp.distributions.MultivariateNormalDiag):

    def __init__(self, loc, scale_diag):
        super().__init__(loc=loc, scale_diag=scale_diag)

        # Compute KDF here.
        mat = scipy.io.loadmat("../../Environment/Action_Space/Bout_classification/bouts.mat")
        bout_kinematic_parameters_final_array = mat["BoutKinematicParametersFinalArray"]
        dist_angles = bout_kinematic_parameters_final_array[:, 10]  # Angles including glide
        distance_x_inc_glide = bout_kinematic_parameters_final_array[:, 18]  # In mm
        distance_y_inc_glide = bout_kinematic_parameters_final_array[:, 19]  # In mm

        distance = (distance_x_inc_glide ** 2 + distance_y_inc_glide ** 2) ** 0.5

        impulse = (distance * 10 - (0.004644 * 140.0 + 0.081417)) / 1.771548
        dist_angles_radians = (np.absolute(dist_angles) / 180) * np.pi

        impulse = np.expand_dims(impulse, 1)
        dist_angles_radians = np.expand_dims(dist_angles_radians, 1)
        actions = np.concatenate((impulse, dist_angles_radians), axis=1)

        model = DBSCAN(eps=0.8125, min_samples=5).fit(actions)
        sorted_actions = actions[model.labels_ != -1]

        # Extra step - cut off negative impulse values
        sorted_actions = sorted_actions[sorted_actions[:, 0] >= 0]
        sorted_actions = sorted_actions[sorted_actions[:, 1] >= 0]

        # self.kde_impulse = sm.nonparametric.KDEMultivariate(data=sorted_actions[:, 0], var_type='c', bw='cv_ml')
        # self.kde_angle = sm.nonparametric.KDEMultivariate(data=sorted_actions[:, 1], var_type='c', bw='cv_ml')
        self.kde_impulse = sm.nonparametric.KDEMultivariate(data=sorted_actions[:, 0], var_type='c', bw='cv_ml')
        self.kde_angle = sm.nonparametric.KDEMultivariate(data=sorted_actions[:, 1], var_type='c', bw='cv_ml')

    def index_invalid_actions(self, actions):
        probs = self.kde_impulse.pdf(actions[:, 0]) * self.kde_angle.pdf(np.absolute(actions[:, 1]))
        invalid_indices = np.where(probs < 0.0000237337)[0]
        return invalid_indices.astype(np.int32)

    def get_num_invalid_actions(self, actions):
        probs = self.kde_impulse.pdf(actions[:, 0]) * self.kde_angle.pdf(np.absolute(actions[:, 1]))
        invalid_indices = np.where(probs < 0.0000237337)[0]
        shape = np.array(invalid_indices.shape[0]).astype(np.int32)
        return shape

    def repopulate_array(self, actions, new_samples, re_sample_indices, num_to_replace):
        actions[re_sample_indices, :] = new_samples[:num_to_replace, :]
        return actions

    def re_sample(self, actions, shape):
        re_sample_indices = tf.numpy_function(self.index_invalid_actions, [actions], tf.float32)
        new_samples = self.sample(shape)
        num_to_replace = tf.numpy_function(self.get_num_invalid_actions, [actions], tf.int32)
        re_populated_actions = tf.numpy_function(self.repopulate_array, [actions, new_samples, re_sample_indices, num_to_replace], tf.float32)
        # actions[re_sample_indices] = self.sample(re_sample_indices.shape[0])
        return re_populated_actions

    def new_sample_outer(self, shape):
        actions = tf.Variable(np.tile([-1.0, -100.0], (shape, 1)), shape=(shape, 2))
        shape = tf.Variable(shape)
        conditional = lambda a, s: tf.less(0, tf.numpy_function(self.get_num_invalid_actions, [a], tf.int32))
        resampled_actions = tf.while_loop(conditional, self.re_sample, [actions, shape], shape_invariants=[actions.get_shape(), shape.get_shape()])

        return resampled_actions

    def sample_outer(self, shape):
        while True:
            actions = self.sample(shape)
            invalid_action_indices = tf.numpy_function(self.index_invalid_actions, [actions], tf.float32)
            if invalid_action_indices.shape[0] > 0:
                new_actions = self.sample(len(invalid_action_indices))
                actions[invalid_action_indices.astype(int), :] = new_actions
            else:
                return actions

    def get_sample_masked_weights(self, actions, shape):
        probs = self.kde_impulse.pdf(actions[:, 0]) * self.kde_angle.pdf(np.absolute(actions[:, 1]))
        integral = np.sum(probs)
        indices_chosen = np.random.choice(actions.shape[0], size=shape, p=probs/integral, replace=False)
        actions_chosen = actions[indices_chosen]
        return actions_chosen

    def sample_repeated(self, shape):
        preliminary_samples = self.sample(shape * 1)
        chosen_samples = tf.numpy_function(self.get_sample_masked_weights, [preliminary_samples, shape], tf.float32)
        return chosen_samples

    # def log_prob(self, action):
    #     # Normalisation - for 1000x1000 sampling FOR 12x350, total sum was 70343. Oversampling =5000/21.
    #     probs = self.kde_impulse.pdf(action[:, 0]) * self.kde_angle.pdf(action[:, 1])
    #     probs[probs < 0.0000237337] = 0
    #     probs[probs > 0.0000237337] = 1/295.4406


# sess = tf.Session()
# with sess as sess:
#
#     trainables = tf.trainable_variables()
#     test = MaskedMultivariateNormalOLD([5, 0], [1, 1])
#     # actions = test.new_sample_outer(10)
#     actions = test.sample_repeated(10)
#     init = tf.global_variables_initializer()
#     sess.run(init)
#     for i in range(100):
#         actions2 = sess.run(actions)
#         print(actions2)

import copy
import os
import math
import numpy as np
import json

# noinspection PyUnresolvedReferences
import tensorflow.compat.v1 as tf

from Networks.PPO.proximal_policy_optimizer_continuous import PPONetworkActorMultivariate

tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)


class ContinuousPPO:

    def __init__(self, **kwargs):

        # Placeholders present in service base classes (overwritten by MRO)
        self.learning_params = None
        self.environment_params = None
        self.total_steps = None
        self.simulation = None
        self.buffer = None
        self.sess = None
        self.batch_size = None
        self.trace_length = None
        self.output_dimensions = None
        self.continuous = None

        self.frame_buffer = None
        self.save_frames = None

        # Networks
        self.network = None

        # To check if is assay or training
        self.assay = None

        # Allows use of same episode method
        self.current_episode_max_duration = None
        self.total_episode_reward = 0  # Total reward over episode

        self.init_rnn_state = None  # Reset RNN hidden state
        self.init_rnn_state_ref = None

        # Add attributes only if don't exist yet (prevents errors thrown).
        if not hasattr(self, "get_internal_state_order"):
            self.get_internal_state_order = None
        if not hasattr(self, "step_loop"):
            self.step_loop = None
        if not hasattr(self, "rnn_in_network"):
            self.rnn_in_network = None
        if not hasattr(self, "get_feature_positions"):
            self.get_feature_positions = None
        if not hasattr(self, "save_environmental_data"):
            self.save_environmental_data = None
        if not hasattr(self, "episode_buffer"):
            self.episode_buffer = None
        if not hasattr(self, "use_mu"):
            self.use_mu = None
        if not hasattr(self, "target_rdn"):
            self.target_rdn = None
        if not hasattr(self, "predictor_rdn"):
            self.predictor_rdn = None
        if not hasattr(self, "visual_interruptions"):
            self.visual_interruptions = None
        if not hasattr(self, "efference_copy_interruptions"):
            self.efference_copy_interruptions = None
        if not hasattr(self, "preset_energy_state"):
            self.preset_energy_state = None
        if not hasattr(self, "epsilon"):
            self.epsilon = None
        if not hasattr(self, "model_location"):
            self.model_location = None
        if not hasattr(self, "episode_number"):
            self.episode_number = None
        if not hasattr(self, "in_light_interruptions"):
            self.in_light_interruptions = None
        if not hasattr(self, "salt_interruptions"):
            self.salt_interruptions = None

        self.continuous = True

        self.impulse_sigma = None
        self.angle_sigma = None
        self.sigma_total_steps = 0

        # Placeholders
        self.epsilon_greedy = None
        self.step_drop = None

        self.output_dimensions = 2
        self.just_trained = False

    def create_network(self):
        """
        Create the PPO network, according to the configuration parameters.
        """
        print("Creating networks")
        internal_states = sum(
            [1 for x in [self.environment_params['stress'],
                         self.environment_params['energy_state'], self.environment_params['in_light'],
                         self.environment_params['salt']] if x is True])
        internal_states = max(internal_states, 1)
        cell = tf.nn.rnn_cell.LSTMCell(num_units=self.learning_params['rnn_dim_shared'], state_is_tuple=True)

        self.network = PPONetworkActorMultivariate(simulation=self.simulation,
                                                   rnn_dim=self.learning_params['rnn_dim_shared'],
                                                   rnn_cell=cell,
                                                   my_scope='PPO',
                                                   internal_states=internal_states,
                                                   max_impulse=self.environment_params['max_impulse'],
                                                   max_angle_change=self.environment_params[
                                                       'max_angle_change'],
                                                   clip_param=self.learning_params[
                                                       'clip_param'],
                                                   input_sigmas=True,
                                                   impose_action_mask=True,
                                                   )

        print("Created network")

    def update_sigmas(self):
        """Updates the current sigma exploration parameters according to specified decay. Not really used now, as
        epsilon greedy used for PPO exploration."""

        self.sigma_total_steps += self.simulation.num_steps

        if self.learning_params["sigma_mode"] != "Decreasing":
            self.impulse_sigma = np.array([self.learning_params["max_sigma_impulse"]])
            self.angle_sigma = np.array([self.learning_params["max_sigma_angle"]])
        else:
            # Linear scale
            self.impulse_sigma = np.array([self.learning_params["max_sigma_impulse"] - (
                    self.learning_params["max_sigma_impulse"] - self.learning_params["min_sigma_impulse"]) * (
                                                   self.sigma_total_steps / self.learning_params[
                                               "sigma_reduction_time"])])
            self.angle_sigma = np.array([self.learning_params["max_sigma_angle"] - (
                    self.learning_params["max_sigma_angle"] - self.learning_params["min_sigma_angle"]) * (
                                                 self.sigma_total_steps / self.learning_params["sigma_reduction_time"])])
            # To prevent ever returning NaN
            if math.isnan(self.impulse_sigma[0]):
                self.impulse_sigma = np.array([self.learning_params["min_sigma_impulse"]])
            if math.isnan(self.angle_sigma[0]):
                self.angle_sigma = np.array([self.learning_params["min_sigma_angle"]])

    def _assay_step_loop(self, o, internal_state, a, rnn_state, rnn_state_ref):
        """Run network and simulation step in assay mode."""

        a = [a[0],
             a[1],
             self.simulation.fish.prev_action_impulse,
             self.simulation.fish.prev_action_angle,
             ]

        impulse, angle, V, V_ref, updated_rnn_state, updated_rnn_state_ref, \
        mu_i, mu_a, mu1, mu1_ref, mu_a1, mu_a_ref, si_i, si_a = self.sess.run(
            [self.network.impulse_output, self.network.angle_output, self.network.value_output, self.network.value_fn_2,
             self.network.rnn_state_shared, self.network.rnn_state_ref,
             self.network.mu_impulse_combined,
             self.network.mu_angle_combined,
             self.network.mu_impulse,
             self.network.mu_impulse_ref,
             self.network.mu_angle, self.network.mu_angle_ref,
             self.network.sigma_impulse_combined, self.network.sigma_angle_combined,
             ],
            feed_dict={self.network.observation: o,
                       self.network.internal_state: internal_state,
                       self.network.prev_actions: np.reshape(a, (1, 4)),
                       self.network.sigma_impulse_combined_proto: self.impulse_sigma,
                       self.network.sigma_angle_combined_proto: self.angle_sigma,
                       self.network.rnn_state_in: rnn_state,
                       self.network.rnn_state_in_ref: rnn_state_ref,
                       self.network.batch_size: 1,
                       self.network.train_length: 1,
                       self.network.entropy_coefficient: self.learning_params["lambda_entropy"],
                       }
        )

        if self.use_mu:
            action = [mu_i[0][0] * self.environment_params['max_impulse'],
                      mu_a[0][0] * self.environment_params['max_angle_change']]
        else:
            if self.epsilon_greedy:
                if np.random.rand(1) < self.epsilon:
                    action = [impulse[0][0], angle[0][0]]
                else:
                    action = [mu_i[0][0] * self.environment_params["max_impulse"],
                              mu_a[0][0] * self.environment_params["max_angle_change"]]
            else:
                action = [impulse[0][0], angle[0][0]]

        o1, given_reward, new_internal_state, d, full_masked_image = self.simulation.simulation_step(action=action)

        efference_copy = action + [self.simulation.fish.prev_action_impulse, self.simulation.fish.prev_action_angle]

        # Update buffer
        self.buffer.add_training(observation=o1,
                                 internal_state=new_internal_state,
                                 action=efference_copy,
                                 reward=given_reward,
                                 value=V,
                                 value_ref=V_ref,
                                 l_p_action=0,
                                 rnn_state=updated_rnn_state,
                                 rnn_state_ref=updated_rnn_state_ref,
                                 advantage=0,
                                 advantage_ref=0
                                 )
        self.buffer.add_logging(mu_i, si_i, mu_a, si_a, mu1, mu1_ref, mu_a1, mu_a_ref)
        if "environmental positions" in self.buffer.recordings:
            self.log_data(efference_copy, action)

        return given_reward, new_internal_state, o1, d, updated_rnn_state, updated_rnn_state_ref, action

    def step_loop(self, o, internal_state, a, rnn_state, rnn_state_ref):
        """Run network and simulation step in training mode."""

        a = [a[0],
             a[1],
             self.simulation.fish.prev_action_impulse,
             self.simulation.fish.prev_action_angle,
             ]

        impulse, angle, V, V_ref, updated_rnn_state, updated_rnn_state_ref, neg_log_action_probability, mu_i, mu_a, \
        si = self.sess.run(
            [self.network.impulse_output, self.network.angle_output, self.network.value_fn_1, self.network.value_fn_2,
             self.network.rnn_state_shared, self.network.rnn_state_ref,
             self.network.neg_log_prob,
             self.network.mu_impulse_combined,
             self.network.mu_angle_combined,
             self.network.sigma_action, ],

            feed_dict={self.network.observation: o,
                       self.network.internal_state: internal_state,
                       self.network.prev_actions: np.reshape(a, (1, 4)),
                       self.network.sigma_impulse_combined_proto: self.impulse_sigma,
                       self.network.sigma_angle_combined_proto: self.angle_sigma,
                       self.network.rnn_state_in: rnn_state,
                       # self.network.rnn_state_in_ref: rnn_state_ref,   TODO: Currently incorrect, though learning doesnt occur when correct.
                       self.network.batch_size: 1,
                       self.network.train_length: 1,
                       self.network.entropy_coefficient: self.learning_params["lambda_entropy"],
                       }
        )

        if self.epsilon_greedy:
            if np.random.rand(1) < self.epsilon:
                action = [impulse[0][0], angle[0][0]]
            else:
                action = [mu_i[0][0] * self.environment_params["max_impulse"],
                          mu_a[0][0] * self.environment_params["max_angle_change"]]

                # And get updated neg_log_prob
                neg_log_action_probability = self.sess.run(
                    [self.network.new_neg_log_prob],

                    feed_dict={self.network.observation: o,
                               self.network.internal_state: internal_state,
                               self.network.prev_actions: np.reshape(a, (1, 4)),
                               self.network.sigma_impulse_combined_proto: self.impulse_sigma,
                               self.network.sigma_angle_combined_proto: self.angle_sigma,
                               self.network.rnn_state_in: rnn_state,
                               self.network.rnn_state_in_ref: rnn_state_ref,
                               self.network.batch_size: 1,
                               self.network.train_length: 1,
                               self.network.action_placeholder: np.reshape(action, (1, 2)),
                               }
                )

            if self.epsilon > self.learning_params['endE']:
                self.epsilon -= self.step_drop
        else:
            action = [impulse[0][0], angle[0][0]]

        # Simulation step
        o1, r, new_internal_state, d, full_masked_image = self.simulation.simulation_step(action=action)

        # Changing action to include final action consequences.
        action_consequences = [self.simulation.fish.prev_action_impulse / self.environment_params["max_impulse"],
                               self.simulation.fish.prev_action_angle / self.environment_params["max_angle_change"]]

        efference_copy = action + action_consequences
        self.buffer.add_training(observation=o,
                                 internal_state=internal_state,
                                 action=efference_copy,
                                 reward=r,
                                 l_p_action=neg_log_action_probability,
                                 rnn_state=rnn_state,
                                 rnn_state_ref=rnn_state_ref,
                                 value=V,
                                 value_ref=V_ref,
                                 advantage=0,
                                 advantage_ref=0,
                                 )

        if self.save_environmental_data:
            self.log_data(action, efference_copy)

        si_i = si[0][0]
        si_a = si[0][1]
        self.buffer.add_logging(mu_i, si_i, mu_a, si_a, 0, 0, 0, 0)

        self.total_steps += 1
        return r, new_internal_state, o1, d, updated_rnn_state, updated_rnn_state_ref, action

    def get_batch(self, batch, observation_buffer, internal_state_buffer, action_buffer,
                  previous_action_buffer,
                  log_action_probability_buffer, advantage_buffer, return_buffer, value_buffer):
        """Returns correctly sized arrays for network training values for a given batch."""

        observation_batch = observation_buffer[
                            batch * self.batch_size: (batch + 1) * self.batch_size]
        internal_state_batch = internal_state_buffer[
                               batch * self.batch_size: (batch + 1) * self.batch_size]
        action_batch = action_buffer[
                       batch * self.batch_size: (batch + 1) * self.batch_size]
        previous_action_batch = previous_action_buffer[
                                batch * self.batch_size: (batch + 1) * self.batch_size]
        log_action_probability_batch = log_action_probability_buffer[
                                       batch * self.learning_params["batch_size"]: (batch + 1) * self.learning_params[
                                           "batch_size"]]
        advantage_batch = advantage_buffer[
                          batch * self.learning_params["batch_size"]: (batch + 1) * self.learning_params["batch_size"]]
        return_batch = return_buffer[
                       batch * self.learning_params["batch_size"]: (batch + 1) * self.learning_params["batch_size"]]

        value_batch = value_buffer[
                      batch * self.learning_params["batch_size"]: (batch + 1) * self.learning_params["batch_size"]]

        current_batch_size = observation_batch.shape[0]

        # Stacking for correct network dimensions
        observation_batch = np.vstack(np.vstack(observation_batch))
        internal_state_batch = np.vstack(np.vstack(internal_state_batch))
        action_batch = np.concatenate(np.array(action_batch))
        action_batch = np.reshape(np.array(action_batch),
                                  (self.learning_params["trace_length"] * current_batch_size, 2))

        previous_action_batch = np.vstack(np.vstack(previous_action_batch))
        log_action_probability_batch = log_action_probability_batch.flatten()
        advantage_batch = np.vstack(advantage_batch).flatten()
        return_batch = np.vstack(np.vstack(return_batch)).flatten()
        value_batch = value_batch.flatten()

        return observation_batch, internal_state_batch, action_batch, previous_action_batch, \
               log_action_probability_batch, advantage_batch, return_batch, value_batch, \
               current_batch_size

    def train_network(self):
        """Perform training of PPO network for an episode."""

        if (self.learning_params["batch_size"] * self.learning_params["trace_length"]) * 2 > len(
                self.buffer.reward_buffer):
            return
        else:
            self.buffer.tidy()

        observation_buffer, internal_state_buffer, action_buffer, previous_action_buffer, log_action_probability_buffer, \
        advantage_buffer, return_buffer, value_buffer, key_rnn_points = self.buffer.get_episode_buffer()

        number_of_batches = int(math.ceil(observation_buffer.shape[0] / self.learning_params["batch_size"]))

        for batch in range(number_of_batches):
            # Find steps at start of each trace to compute RNN states
            batch_key_points = [i for i in key_rnn_points if
                                batch * self.learning_params["batch_size"] * self.learning_params[
                                    "trace_length"] <= i < (batch + 1) *
                                self.learning_params["batch_size"] * self.learning_params["trace_length"]]

            # Get the current batch
            observation_batch, internal_state_batch, action_batch, previous_action_batch, log_action_probability_batch, \
            advantage_batch, return_batch, previous_value_batch, current_batch_size = self.get_batch(
                batch, observation_buffer, internal_state_buffer, action_buffer, previous_action_buffer,
                log_action_probability_buffer, advantage_buffer, return_buffer, value_buffer)

            # Loss value logging
            average_loss_value = 0
            average_loss_impulse = 0
            average_loss_angle = 0
            average_loss_entropy = 0

            for i in range(self.learning_params["n_updates_per_iteration"]):
                # Get RNN states for start of each trace.
                rnn_state_slice, rnn_state_ref_slice = self.get_rnn_states(batch_key_points)

                # Optimise
                loss_actor_val, loss_critic_val, loss_entropy, total_loss, _, log_action_probability_batch_new, scaled_actions = self.sess.run(
                    [self.network.policy_loss, self.network.value_loss, self.network.entropy,
                     self.network.total_loss, self.network.train, self.network.new_neg_log_prob,
                     self.network.normalised_action],

                    feed_dict={self.network.observation: observation_batch,
                               self.network.prev_actions: previous_action_batch,
                               self.network.internal_state: internal_state_batch,
                               self.network.rnn_state_in: rnn_state_slice,
                               self.network.rnn_state_in_ref: rnn_state_ref_slice,

                               self.network.sigma_impulse_combined_proto: self.impulse_sigma,
                               self.network.sigma_angle_combined_proto: self.angle_sigma,

                               self.network.action_placeholder: action_batch,
                               self.network.old_neg_log_prob: log_action_probability_batch,
                               self.network.scaled_advantage_placeholder: advantage_batch,
                               self.network.returns_placeholder: return_batch,

                               self.network.old_value_placeholder: previous_value_batch,

                               self.network.train_length: self.learning_params["trace_length"],
                               self.network.batch_size: current_batch_size,
                               self.network.learning_rate: self.learning_params[
                                                               "learning_rate"] * current_batch_size,
                               self.network.entropy_coefficient: self.learning_params["lambda_entropy"]
                               })

                average_loss_impulse += np.mean(loss_actor_val)
                average_loss_angle += np.mean(loss_actor_val)
                average_loss_value += np.abs(loss_critic_val)
                average_loss_entropy += np.mean(loss_entropy)

            self.buffer.add_loss(average_loss_impulse / self.learning_params["n_updates_per_iteration"],
                                 average_loss_angle / self.learning_params["n_updates_per_iteration"],
                                 average_loss_value / self.learning_params["n_updates_per_iteration"],
                                 average_loss_entropy / self.learning_params["n_updates_per_iteration"])
        self.just_trained = True

    def init_states(self):
        """Init states for RNN"""

        if os.path.isfile(f"{self.model_location}/rnn_state-{self.episode_number}.json"):
            with open(f"{self.model_location}/rnn_state-{self.episode_number}.json", 'r') as f:
                print("Successfully loaded previous state.")
                data = json.load(f)
                num_rnns = len(data.keys()) / 4
                self.init_rnn_state = tuple(
                    (np.array(data[f"rnn_state_{shape}_1"]), np.array(data[f"rnn_state_{shape}_2"])) for shape in
                    range(int(num_rnns)))
                self.init_rnn_state_ref = tuple(
                    (np.array(data[f"rnn_state_{shape}_ref_1"]), np.array(data[f"rnn_state_{shape}_ref_2"])) for shape
                    in range(int(num_rnns)))
        else:
            # Init states for RNN - For steps, not training.
            rnn_state_shapes = [self.learning_params['rnn_dim_shared']]
            self.init_rnn_state = tuple(
                (np.zeros([1, shape]), np.zeros([1, shape])) for shape in rnn_state_shapes)
            self.init_rnn_state_ref = tuple(
                (np.zeros([1, shape]), np.zeros([1, shape])) for shape in rnn_state_shapes)

    def _episode_loop(self, a=None):
        """Run episode for Training mode only."""
        self.update_sigmas()
        if a is None:
            a = [4.0, 0.0]

        rnn_state = copy.copy(self.init_rnn_state)
        rnn_state_ref = copy.copy(self.init_rnn_state_ref)

        if self.assay or self.just_trained:
            self.buffer.reset()
            self.just_trained = False
        self.simulation.reset()

        o, r, internal_state, d, full_masked_image = self.simulation.simulation_step(action=a)

        self.total_episode_reward = 0  # Total reward over episode

        action = a + [self.simulation.fish.prev_action_impulse,
                      self.simulation.fish.prev_action_angle]
        self.buffer.action_buffer.append(action)  # Add to buffer for loading of previous actions

        self.step_number = 0
        while self.step_number < self.current_episode_max_duration:
            if self.assay is not None:
                self.previous_action = a

            self.step_number += 1

            r, internal_state, o, d, rnn_state, rnn_state_ref, a = self.step_loop(
                o=o,
                internal_state=internal_state,
                a=a,
                rnn_state=rnn_state,
                rnn_state_ref=rnn_state_ref,
            )

            self.total_episode_reward += r
            if d:
                self.init_rnn_state = rnn_state
                self.init_rnn_state_ref = rnn_state_ref
                break

    def get_rnn_states(self, rnn_key_points):
        """Gets RNN states from buffer - used in Training."""
        batch_size = len(rnn_key_points)
        if self.rnn_in_network:
            rnn_state_buffer, rnn_state_ref_buffer = self.buffer.get_rnn_batch(rnn_key_points, batch_size)
        else:
            rnn_state_buffer = ()
            rnn_state_ref_buffer = ()

        return rnn_state_buffer, rnn_state_ref_buffer

import copy
import os
import math
import numpy as np
import json

import tensorflow.compat.v1 as tf

from Networks.PPO.proximal_policy_optimizer_continuous_sb_emulator_dynamic import PPONetworkActorMultivariate2Dynamic

tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)


class ContinuousPPO:

    def __init__(self, **kwargs):
        print("PPO Constructor called")

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
        self.actor_network = None
        self.critic_network = None

        # To check if is assay or training
        self.assay = None

        # Allows use of same episode method
        self.current_episode_max_duration = None
        self.total_episode_reward = 0  # Total reward over episode

        # self.init_rnn_state_actor = None  # Reset RNN hidden state
        # self.init_rnn_state_actor_ref = None
        self.init_rnn_state_critic = None
        self.init_rnn_state_critic_ref = None

        self.init_rnn_state = None # self.init_rnn_state_actor  # Reset RNN hidden state
        self.init_rnn_state_ref = None  # self.init_rnn_state_actor_ref

        # Add attributes only if don't exist yet (prevents errors thrown).
        if not hasattr(self, "get_internal_state_order"):
            self.get_internal_state_order = None
        if not hasattr(self, "sb_emulator"):
            self.sb_emulator = None
        if not hasattr(self, "step_loop"):
            self.step_loop = None
        if not hasattr(self, "rnn_in_network"):
            self.rnn_in_network = None
        if not hasattr(self, "get_positions"):
            self.get_positions = None
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
        if not hasattr(self, "separate_networks"):
            self.separate_networks = None
        if not hasattr(self, "visual_interruptions"):
            self.visual_interruptions = None
        if not hasattr(self, "reafference_interruptions"):
            self.reafference_interruptions = None
        if not hasattr(self, "preset_energy_state"):
            self.preset_energy_state = None


        self.continuous = True

        self.impulse_sigma = None
        self.angle_sigma = None
        self.multivariate = None
        self.sigma_total_steps = 0

        # Placeholders
        self.epsilon_greedy = None
        # self.e = None
        self.step_drop = None

        self.output_dimensions = 2
        self.just_trained = False

    def create_network(self):
        """
        Create the main and target Q networks, according to the configuration parameters.
        :return: The main network and the target network graphs.
        """
        print("Creating networks...")
        internal_states = sum(
            [1 for x in [self.environment_params['hunger'], self.environment_params['stress'],
                         self.environment_params['energy_state'], self.environment_params['in_light'],
                         self.environment_params['salt']] if x is True])
        internal_states = max(internal_states, 1)
        internal_state_names = self.get_internal_state_order()

        if "reuse_eyes" in self.learning_params:
            reuse_eyes = self.learning_params['reuse_eyes']
        else:
            reuse_eyes = False
        self.actor_network = PPONetworkActorMultivariate2Dynamic(simulation=self.simulation,
                                                                 # rnn_dim=self.learning_params['rnn_dim_shared'],
                                                                 # rnn_cell=actor_cell,
                                                                 my_scope='actor',
                                                                 internal_states=internal_states,
                                                                 internal_state_names=internal_state_names,
                                                                 max_impulse=self.environment_params[
                                                                     'max_impulse'],
                                                                 max_angle_change=self.environment_params[
                                                                     'max_angle_change'],
                                                                 clip_param=self.environment_params[
                                                                     'clip_param'],
                                                                 input_sigmas=self.learning_params[
                                                                     'input_sigmas'],
                                                                 impose_action_mask=self.environment_params[
                                                                     'impose_action_mask'],
                                                                 base_network_layers=self.learning_params[
                                                                     'base_network_layers'],
                                                                 modular_network_layers=self.learning_params[
                                                                     'modular_network_layers'],
                                                                 ops=self.learning_params['ops'],
                                                                 connectivity=self.learning_params[
                                                                     'connectivity'],
                                                                 reflected=self.learning_params['reflected'],
                                                                 reuse_eyes=reuse_eyes,
                                                                 )



        print("Created network")

    def update_sigmas(self):
        self.sigma_total_steps += self.simulation.num_steps

        if self.environment_params["sigma_mode"] != "Decreasing":
            self.impulse_sigma = np.array([self.environment_params["max_sigma_impulse"]])
            self.angle_sigma = np.array([self.environment_params["max_sigma_angle"]])
        else:
            # Linear scale
            self.impulse_sigma = np.array([self.environment_params["max_sigma_impulse"] - (
                    self.environment_params["max_sigma_impulse"] - self.environment_params["min_sigma_impulse"]) * (
                                                   self.sigma_total_steps / self.environment_params[
                                               "sigma_reduction_time"])])
            self.angle_sigma = np.array([self.environment_params["max_sigma_angle"] - (
                    self.environment_params["max_sigma_angle"] - self.environment_params["min_sigma_angle"]) * (
                                                 self.sigma_total_steps / self.environment_params[
                                             "sigma_reduction_time"])])
            # To prevent ever returning NaN
            if math.isnan(self.impulse_sigma[0]):
                self.impulse_sigma = np.array([self.environment_params["min_sigma_impulse"]])
            if math.isnan(self.angle_sigma[0]):
                self.angle_sigma = np.array([self.environment_params["min_sigma_angle"]])

    def _assay_step_loop(self, o, internal_state, a, rnn_state_actor, rnn_state_actor_ref,
                         rnn_state_critic, rnn_state_critic_ref):
        sa = np.zeros((1, 128))  # Placeholder for the state advantage stream.
        a = [a[0],
             a[1],
             self.simulation.fish.prev_action_impulse,
             self.simulation.fish.prev_action_angle,
             ]# Set impulse to scale to be inputted to network

        impulse, angle, V, updated_rnn_state_actor, updated_rnn_state_actor_ref, network_layers, \
        mu_i, mu_a, mu1, mu1_ref, mu_a1, mu_a_ref, si_i, si_a = self.sess.run(
            [self.actor_network.impulse_output, self.actor_network.angle_output, self.actor_network.value_output,
             self.actor_network.rnn_state_shared, self.actor_network.rnn_state_ref,
             self.actor_network.network_graph,
             self.actor_network.mu_impulse_combined,
             self.actor_network.mu_angle_combined,
             self.actor_network.mu_impulse,
             self.actor_network.mu_impulse_ref,
             self.actor_network.mu_angle, self.actor_network.mu_angle_ref,
             self.actor_network.sigma_impulse_combined, self.actor_network.sigma_angle_combined,
             ],
            feed_dict={self.actor_network.observation: o,
                       self.actor_network.internal_state: internal_state,
                       self.actor_network.prev_actions: np.reshape(a, (1, 4)),
                       self.actor_network.sigma_impulse_combined_proto: self.impulse_sigma,
                       self.actor_network.sigma_angle_combined_proto: self.angle_sigma,
                       self.actor_network.rnn_state_in: rnn_state_actor,
                       self.actor_network.rnn_state_in_ref: rnn_state_actor_ref,
                       self.actor_network.batch_size: 1,
                       self.actor_network.train_length: 1,
                       self.actor_network.entropy_coefficient: self.learning_params["lambda_entropy"],
                       }
        )

        if self.use_mu:
            action = [mu_i[0][0] * self.environment_params['max_impulse'],
                      mu_a[0][0] * self.environment_params['max_angle_change']]
        else:
            if self.epsilon_greedy:
                if np.random.rand(1) < self.e:
                    action = [impulse[0][0], angle[0][0]]
                else:
                    action = [mu_i[0][0] * self.environment_params["max_impulse"],
                              mu_a[0][0] * self.environment_params["max_angle_change"]]
            else:
                action = [impulse[0][0], angle[0][0]]

        o1, given_reward, new_internal_state, d, FOV = self.simulation.simulation_step(action=action,
                                                                                       activations=(sa,))

        sand_grain_positions, prey_positions, predator_position, vegetation_positions = self.get_positions()

        action_reafference = action + [self.simulation.fish.prev_action_impulse, self.simulation.fish.prev_action_angle]

        # Update buffer
        self.buffer.add_training(observation=o,
                                 internal_state=internal_state,
                                 action=action_reafference,
                                 reward=given_reward,
                                 value=V,
                                 l_p_action=0,
                                 actor_rnn_state=rnn_state_actor,
                                 actor_rnn_state_ref=rnn_state_actor_ref,
                                 critic_rnn_state=rnn_state_critic,
                                 critic_rnn_state_ref=rnn_state_critic_ref,
                                 )
        self.buffer.add_logging(mu_i, si_i, mu_a, si_a, mu1, mu1_ref, mu_a1, mu_a_ref)

        if "environmental positions" in self.buffer.recordings:
            prey_orientations = np.array([p.angle for p in self.simulation.prey_bodies]).astype(np.float32)
            try:
                predator_orientation = self.simulation.predator_body.angle
            except:
                predator_orientation = 0
            prey_ages = self.simulation.prey_ages
            prey_gait = self.simulation.paramecia_gaits

            self.buffer.save_environmental_positions(self.simulation.fish.body.position,
                                                     self.simulation.prey_consumed_this_step,
                                                     self.simulation.predator_body,
                                                     prey_positions,
                                                     predator_position,
                                                     sand_grain_positions,
                                                     vegetation_positions,
                                                     self.simulation.fish.body.angle,
                                                     self.simulation.fish.salt_health,
                                                     efference_copy=a,
                                                     prey_orientation=prey_orientations,
                                                     predator_orientation=predator_orientation,
                                                     prey_age=prey_ages,
                                                     prey_gait=prey_gait
                                                     )

        self.buffer.make_desired_recordings(network_layers)

        return given_reward, new_internal_state, o1, d, updated_rnn_state_actor, updated_rnn_state_actor_ref, \
               0, 0, action

    def _step_loop_full_logs(self, o, internal_state, a, rnn_state_actor, rnn_state_actor_ref,
                             rnn_state_critic,
                             rnn_state_critic_ref):
        sa = np.zeros((1, 128))  # Placeholder for the state advantage stream.
        a = [a[0],
             a[1],
             self.simulation.fish.prev_action_impulse,
             self.simulation.fish.prev_action_angle,
             ]# Set impulse to scale to be inputted to network

        impulse, angle, V, updated_rnn_state_actor, updated_rnn_state_actor_ref, neg_log_action_probability, mu_i, mu_a, \
        si = self.sess.run(
            [self.actor_network.impulse_output, self.actor_network.angle_output, self.actor_network.value_output,
             self.actor_network.rnn_state_shared, self.actor_network.rnn_state_ref,
             self.actor_network.neg_log_prob,
             self.actor_network.mu_impulse_combined,
             self.actor_network.mu_angle_combined,
             self.actor_network.sigma_action, ],

            feed_dict={self.actor_network.observation: o,
                       self.actor_network.internal_state: internal_state,
                       self.actor_network.prev_actions: np.reshape(a, (1, 4)),
                       self.actor_network.sigma_impulse_combined_proto: self.impulse_sigma,
                       self.actor_network.sigma_angle_combined_proto: self.angle_sigma,
                       self.actor_network.rnn_state_in: rnn_state_actor,
                       # self.actor_network.rnn_state_in_ref: rnn_state_actor_ref,
                       self.actor_network.batch_size: 1,
                       self.actor_network.train_length: 1,
                       self.actor_network.entropy_coefficient: self.learning_params["lambda_entropy"],
                       }
        )

        if self.epsilon_greedy:
            if np.random.rand(1) < self.e:
                action = [impulse[0][0], angle[0][0]]
            else:
                action = [mu_i[0][0] * self.environment_params["max_impulse"],
                          mu_a[0][0] * self.environment_params["max_angle_change"]]

                # And get updated neg_log_prob
                neg_log_action_probability = self.sess.run(
                    [self.actor_network.new_neg_log_prob],

                    feed_dict={self.actor_network.observation: o,
                               self.actor_network.internal_state: internal_state,
                               self.actor_network.prev_actions: np.reshape(a, (1, 4)),
                               self.actor_network.sigma_impulse_combined_proto: self.impulse_sigma,
                               self.actor_network.sigma_angle_combined_proto: self.angle_sigma,
                               self.actor_network.rnn_state_in: rnn_state_actor,
                               # self.actor_network.rnn_state_in_ref: rnn_state_actor_ref,
                               self.actor_network.batch_size: 1,
                               self.actor_network.train_length: 1,

                               self.actor_network.action_placeholder: np.reshape(action, (1, 2)),
                               }
                )

            if self.e > self.learning_params['endE']:
                self.e -= self.step_drop
        else:
            action = [impulse[0][0], angle[0][0]]

        # Simulation step
        o1, r, new_internal_state, d, FOV = self.simulation.simulation_step(
            action=action,
            activations=sa)

        # Changing action to include final action consequences.
        action_consequences = [self.simulation.fish.prev_action_impulse / self.environment_params["max_impulse"],
                               self.simulation.fish.prev_action_angle / self.environment_params["max_angle_change"]]

        if self.use_rnd:
            target_output = self.sess.run(self.target_rdn.rdn_output,
                                          feed_dict={self.target_rdn.observation: o,
                                                     self.target_rdn.internal_state: internal_state,
                                                     self.target_rdn.prev_actions: np.reshape(a, (1, 2)),
                                                     self.target_rdn.batch_size: 1,
                                                     self.target_rdn.train_length: 1,
                                                     }
                                          )
            predictor_output = self.sess.run(self.predictor_rdn.rdn_output,
                                             feed_dict={self.predictor_rdn.observation: o,
                                                        self.predictor_rdn.internal_state: internal_state,
                                                        self.predictor_rdn.prev_actions: np.reshape(a, (1, 2)),
                                                        self.predictor_rdn.batch_size: 1,
                                                        self.predictor_rdn.train_length: 1,
                                                        }
                                             )
            prediction_error = (predictor_output - target_output) ** 2
            # Update buffer
            self.buffer.add_training(observation=o,
                                     internal_state=internal_state,
                                     action=action,
                                     reward=r,
                                     value=V,
                                     l_p_action=neg_log_action_probability,
                                     actor_rnn_state=rnn_state_actor,
                                     actor_rnn_state_ref=rnn_state_actor_ref,
                                     prediction_error=prediction_error,
                                     target_output=target_output,
                                     )
        else:
            action = action + action_consequences
            self.buffer.add_training(observation=o,
                                     internal_state=internal_state,
                                     action=action,
                                     reward=r,
                                     value=V,
                                     l_p_action=neg_log_action_probability,
                                     actor_rnn_state=rnn_state_actor,
                                     actor_rnn_state_ref=rnn_state_actor_ref,
                                     )

        if self.save_environmental_data:
            sand_grain_positions, prey_positions, predator_position, vegetation_positions = self.get_positions()
            prey_orientations = [p.angle for p in self.simulation.prey_bodies]
            try:
                predator_orientation = self.simulation.predator_body.angle
            except:
                predator_orientation = 0
            prey_ages = self.simulation.prey_ages
            prey_gait = self.simulation.paramecia_gaits

            self.buffer.save_environmental_positions(self.simulation.fish.body.position,
                                                     self.simulation.prey_consumed_this_step,
                                                     self.simulation.predator_body,
                                                     prey_positions,
                                                     predator_position,
                                                     sand_grain_positions,
                                                     vegetation_positions,
                                                     self.simulation.fish.body.angle,
                                                     self.simulation.fish.salt_health,
                                                     efference_copy=a,
                                                     prey_orientation=prey_orientations,
                                                     predator_orientation=predator_orientation,
                                                     prey_age=prey_ages,
                                                     prey_gait=prey_gait
                                                     )



        si_i = si[0][0]
        si_a = si[0][1]
        self.buffer.add_logging(mu_i, si_i, mu_a, si_a, 0, 0, 0, 0)

        self.total_steps += 1
        return r, new_internal_state, o1, d, updated_rnn_state_actor, updated_rnn_state_actor_ref, 0, 0, action

    def _step_loop_reduced_logs(self, o, internal_state, a, rnn_state_actor, rnn_state_actor_ref,
                                rnn_state_critic,
                                rnn_state_critic_ref):

        sa = np.zeros((1, 128))  # Placeholder for the state advantage stream.
        # a = [a[0] / self.environment_params['max_impulse'],
        #      a[1] / self.environment_params['max_angle_change']]  # Set impulse to scale to be inputted to network

        a = [a[0],
             a[1],
             self.simulation.fish.prev_action_impulse,
             self.simulation.fish.prev_action_angle,
             ]# Set impulse to scale to be inputted to network


        impulse, angle, V, updated_rnn_state_actor, updated_rnn_state_actor_ref, mu_i, mu_a, neg_log_action_probability = self.sess.run(
            [self.actor_network.impulse_output, self.actor_network.angle_output,
             self.actor_network.value_output,
             self.actor_network.rnn_state_shared, self.actor_network.rnn_state_ref,
             self.actor_network.mu_impulse_combined,
             self.actor_network.mu_angle_combined,
             self.actor_network.neg_log_prob],

            feed_dict={self.actor_network.observation: o,
                       self.actor_network.internal_state: internal_state,
                       self.actor_network.prev_actions: np.reshape(a, (1, 2)),
                       self.actor_network.sigma_impulse_combined_proto: self.impulse_sigma,
                       self.actor_network.sigma_angle_combined_proto: self.angle_sigma,
                       self.actor_network.rnn_state_in: rnn_state_actor,
                       # self.actor_network.rnn_state_in_ref: rnn_state_actor_ref,
                       self.actor_network.batch_size: 1,
                       self.actor_network.train_length: 1,
                       self.actor_network.entropy_coefficient: self.learning_params["lambda_entropy"],
                       }
        )
        if self.epsilon_greedy:
            if np.random.rand(1) < self.e:
                action = [impulse[0][0], angle[0][0]]
            else:
                action = [mu_i[0][0] * self.environment_params["max_impulse"],
                          mu_a[0][0] * self.environment_params["max_angle_change"]]

                # And get updated neg_log_prob
                neg_log_action_probability = self.sess.run(
                    [self.actor_network.new_neg_log_prob],

                    feed_dict={self.actor_network.observation: o,
                               self.actor_network.internal_state: internal_state,
                               self.actor_network.prev_actions: np.reshape(a, (1, 2)),
                               self.actor_network.sigma_impulse_combined_proto: self.impulse_sigma,
                               self.actor_network.sigma_angle_combined_proto: self.angle_sigma,
                               self.actor_network.rnn_state_in: rnn_state_actor,
                               # self.actor_network.rnn_state_in_ref: rnn_state_actor_ref,
                               self.actor_network.batch_size: 1,
                               self.actor_network.train_length: 1,

                               self.actor_network.action_placeholder: action,
                               }
                )

            if self.e > self.learning_params['endE']:
                self.e -= self.step_drop
        else:
            action = [impulse[0][0], angle[0][0]]

        # Simulation step
        o1, r, new_internal_state, d, FOV = self.simulation.simulation_step(
            action=action,
            activations=sa)

        action_consequences = [self.simulation.fish.prev_action_impulse / self.environment_params["max_impulse"],
                               self.simulation.fish.prev_action_angle / self.environment_params["max_angle_change"]]

        if self.use_rnd:
            target_output = self.sess.run(self.target_rdn.rdn_output,
                                          feed_dict={self.actor_network.observation: o,
                                                     self.actor_network.internal_state: internal_state,
                                                     self.actor_network.prev_actions: np.reshape(a, (1, 2)),
                                                     self.actor_network.batch_size: 1,
                                                     self.actor_network.train_length: 1,
                                                     }
                                          )
            predictor_output = self.sess.run(self.predictor_rdn.rdn_output,
                                             feed_dict={self.actor_network.observation: o,
                                                        self.actor_network.internal_state: internal_state,
                                                        self.actor_network.prev_actions: np.reshape(a, (1, 2)),
                                                        self.actor_network.batch_size: 1,
                                                        self.actor_network.train_length: 1,
                                                        }
                                             )
            prediction_error = (predictor_output - target_output) ** 2

            # Update buffer
            self.buffer.add_training(observation=o,
                                     internal_state=internal_state,
                                     action=action,
                                     reward=r,
                                     value=V,
                                     l_p_action=neg_log_action_probability,
                                     actor_rnn_state=rnn_state_actor,
                                     actor_rnn_state_ref=rnn_state_actor_ref,
                                     prediction_error=prediction_error,
                                     target_output=target_output,
                                     )
        else:
            action = action + action_consequences
            self.buffer.add_training(observation=o,
                                     internal_state=internal_state,
                                     action=action,
                                     reward=r,
                                     value=V,
                                     l_p_action=neg_log_action_probability,
                                     actor_rnn_state=rnn_state_actor,
                                     actor_rnn_state_ref=rnn_state_actor_ref,
                                     )

        if self.save_environmental_data:
            sand_grain_positions, prey_positions, predator_position, vegetation_positions = self.get_positions()
            self.episode_buffer.save_environmental_positions(self.simulation.fish.body.position,
                                                             self.simulation.prey_consumed_this_step,
                                                             self.simulation.predator_body,
                                                             prey_positions,
                                                             predator_position,
                                                             sand_grain_positions,
                                                             vegetation_positions,
                                                             self.simulation.fish.body.angle,
                                                             )

        self.total_steps += 1
        return r, new_internal_state, o1, d, updated_rnn_state_actor, updated_rnn_state_actor_ref, 0, 0, action

    def get_batch(self, batch, observation_buffer, internal_state_buffer, action_buffer,
                  previous_action_buffer,
                  log_action_probability_buffer, advantage_buffer, return_buffer, value_buffer,
                  target_outputs_buffer=None):

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

        if target_outputs_buffer is not None:
            target_outputs_batch = target_outputs_buffer[
                                   batch * self.learning_params["batch_size"]: (batch + 1) * self.learning_params[
                                       "batch_size"]]

        current_batch_size = observation_batch.shape[0]

        # Stacking for correct network dimensions
        observation_batch = np.vstack(np.vstack(observation_batch))
        internal_state_batch = np.vstack(np.vstack(internal_state_batch))
        try:
            action_batch = np.concatenate(np.array(action_batch))
            # print(action_batch.shape)
            action_batch = np.reshape(np.array(action_batch), (self.learning_params["trace_length"] * current_batch_size, 2))
        except:
            action_batch = np.concatenate(np.array(action_batch))

            print("Error... ")
            print(current_batch_size)
            print(action_batch.shape)

            action_batch = np.array(action_batch)

        previous_action_batch = np.vstack(np.vstack(previous_action_batch))
        log_action_probability_batch = log_action_probability_batch.flatten()
        advantage_batch = np.vstack(advantage_batch).flatten()
        return_batch = np.vstack(np.vstack(return_batch)).flatten()
        value_batch = value_batch.flatten()
        if target_outputs_buffer is not None:
            target_outputs_batch = np.vstack(target_outputs_batch)

        if self.use_rnd:
            return observation_batch, internal_state_batch, action_batch, previous_action_batch, \
                   log_action_probability_batch, advantage_batch, return_batch, value_batch, target_outputs_batch, \
                   current_batch_size
        else:
            return observation_batch, internal_state_batch, action_batch, previous_action_batch, \
                   log_action_probability_batch, advantage_batch, return_batch, value_batch, \
                   current_batch_size

    def train_network(self):
        if (self.learning_params["batch_size"] * self.learning_params["trace_length"]) * 2 > len(self.buffer.reward_buffer):
            # print(f"Buffer too small: {len(self.buffer.reward_buffer)}")
            return
        else:
            self.buffer.tidy()
            # print(f"Buffer large enough: {len(self.buffer.reward_buffer)}")

        if self.use_rnd:
            observation_buffer, internal_state_buffer, action_buffer, previous_action_buffer, \
            log_action_probability_buffer, advantage_buffer, return_buffer, value_buffer, target_outputs_buffer, \
            key_rnn_points = self.buffer.get_episode_buffer()
        else:
            observation_buffer, internal_state_buffer, action_buffer, previous_action_buffer, \
            log_action_probability_buffer, advantage_buffer, return_buffer, value_buffer, \
            key_rnn_points = self.buffer.get_episode_buffer()

        # if self.learning_params["batch_size"] == 1:
        #     number_of_batches = int(math.ceil(observation_buffer.shape[0] / self.learning_params["batch_size"]))
        # else:
        #     number_of_batches = int(math.ceil(observation_buffer.shape[0] / (self.learning_params["batch_size"] * self.learning_params["trace_length"])))

        number_of_batches = int(math.ceil(observation_buffer.shape[0] / self.learning_params["batch_size"]))

        # print(f"Number of batches: {number_of_batches}")

        for batch in range(number_of_batches):
            # Find steps at start of each trace to compute RNN states
            batch_key_points = [i for i in key_rnn_points if
                                batch * self.learning_params["batch_size"] * self.learning_params[
                                    "trace_length"] <= i < (batch + 1) *
                                self.learning_params["batch_size"] * self.learning_params["trace_length"]]
            # print(f"Batch key points: {batch_key_points}")

            # Get the current batch
            if self.use_rnd:
                observation_batch, internal_state_batch, action_batch, previous_action_batch, \
                log_action_probability_batch, advantage_batch, \
                return_batch, previous_value_batch, target_outputs_batch, current_batch_size = self.get_batch(
                    batch, observation_buffer,
                    internal_state_buffer,
                    action_buffer, previous_action_buffer,
                    log_action_probability_buffer,
                    advantage_buffer,
                    return_buffer, value_buffer, target_outputs_buffer)
            else:
                observation_batch, internal_state_batch, action_batch, previous_action_batch, \
                log_action_probability_batch, advantage_batch, \
                return_batch, previous_value_batch, current_batch_size = self.get_batch(
                    batch, observation_buffer,
                    internal_state_buffer,
                    action_buffer, previous_action_buffer,
                    log_action_probability_buffer,
                    advantage_buffer,
                    return_buffer, value_buffer)

            # Loss value logging
            average_loss_value = 0
            average_loss_impulse = 0
            average_loss_angle = 0
            average_loss_entropy = 0

            # print(f"""
            # o: {np.array(observation_batch).shape}
            # is: {np.array(internal_state_batch).shape}
            # a: {np.array(action_batch).shape}
            # pa: {np.array(previous_action_batch).shape}
            # ad: {np.array(advantage_batch).shape}
            # re: {np.array(return_batch).shape}
            # pv: {np.array(previous_value_batch).shape}
            # cbs: {current_batch_size}
            # """)

            for i in range(self.learning_params["n_updates_per_iteration"]):
                # Compute RNN states for start of each trace.
                actor_rnn_state_slice, actor_rnn_state_ref_slice = self.compute_rnn_states2(batch_key_points,
                                                                                            observation_buffer[
                                                                                            :(batch + 1) *
                                                                                             self.learning_params[
                                                                                                 "batch_size"]],
                                                                                            internal_state_buffer[
                                                                                            :(batch + 1) *
                                                                                             self.learning_params[
                                                                                                 "batch_size"]],
                                                                                            previous_action_buffer[
                                                                                            :(batch + 1) *
                                                                                             self.learning_params[
                                                                                                 "batch_size"]])
                # print(f"RNN state: {actor_rnn_state_ref_slice}")
                # print(f"RNN state shape: {actor_rnn_state_ref_slice[0].shape}")
                # Optimise actor
                loss_actor_val, loss_critic_val, loss_entropy, total_loss, _, log_action_probability_batch_new, scaled_actions = self.sess.run(
                    [self.actor_network.policy_loss, self.actor_network.value_loss, self.actor_network.entropy,
                     self.actor_network.total_loss, self.actor_network.train, self.actor_network.new_neg_log_prob,
                     self.actor_network.normalised_action],

                    feed_dict={self.actor_network.observation: observation_batch,
                               self.actor_network.prev_actions: previous_action_batch,
                               self.actor_network.internal_state: internal_state_batch,
                               self.actor_network.rnn_state_in: actor_rnn_state_slice,
                               # self.actor_network.rnn_state_in_ref: actor_rnn_state_ref_slice,

                               self.actor_network.sigma_impulse_combined_proto: self.impulse_sigma,
                               self.actor_network.sigma_angle_combined_proto: self.angle_sigma,

                               self.actor_network.action_placeholder: action_batch,
                               self.actor_network.old_neg_log_prob: log_action_probability_batch,
                               self.actor_network.scaled_advantage_placeholder: advantage_batch,
                               self.actor_network.returns_placeholder: return_batch,

                               self.actor_network.old_value_placeholder: previous_value_batch,

                               self.actor_network.train_length: self.learning_params["trace_length"],
                               self.actor_network.batch_size: current_batch_size,
                               self.actor_network.learning_rate: self.learning_params[
                                                                     "learning_rate_actor"] * current_batch_size,
                               self.actor_network.entropy_coefficient: self.learning_params["lambda_entropy"]
                               })

                average_loss_impulse += np.mean(loss_actor_val)
                average_loss_angle += np.mean(loss_actor_val)
                average_loss_value += np.abs(loss_critic_val)
                average_loss_entropy += np.mean(loss_entropy)

                if self.use_rnd:
                    train = self.sess.run(self.predictor_rdn.train,
                                          feed_dict={
                                              self.predictor_rdn.observation: observation_batch,
                                              self.predictor_rdn.prev_actions: previous_action_batch,
                                              self.predictor_rdn.internal_state: internal_state_batch,

                                              self.predictor_rdn.target_outputs: target_outputs_batch,

                                              self.predictor_rdn.train_length: self.learning_params["trace_length"],
                                              self.predictor_rdn.batch_size: current_batch_size,
                                              self.predictor_rdn.learning_rate: self.learning_params[
                                                                                    "learning_rate_actor"] * current_batch_size
                                          })

            self.buffer.add_loss(average_loss_impulse / self.learning_params["n_updates_per_iteration"],
                                 average_loss_angle / self.learning_params["n_updates_per_iteration"],
                                 average_loss_value / self.learning_params["n_updates_per_iteration"],
                                 average_loss_entropy / self.learning_params["n_updates_per_iteration"])
        self.just_trained = True

    def init_states(self):
        # Init states for RNN
        if "maintain_state" in self.learning_params:
            if self.learning_params["maintain_state"]:
                # IF SAVE PRESENT
                if os.path.isfile(f"{self.model_location}/rnn_state-{self.episode_number}.json"):
                    if self.environment_params["use_dynamic_network"]:
                        with open(f"{self.model_location}/rnn_state-{self.episode_number}.json", 'r') as f:
                            print("Successfully loaded previous state.")
                            data = json.load(f)
                            num_rnns = len(data.keys())/4
                            self.init_rnn_state = tuple((np.array(data[f"rnn_state_{shape}_1"]), np.array(data[f"rnn_state_{shape}_2"])) for shape in range(int(num_rnns)))
                            self.init_rnn_state_ref = tuple((np.array(data[f"rnn_state_{shape}_ref_1"]), np.array(data[f"rnn_state_{shape}_ref_2"])) for shape in range(int(num_rnns)))

                        return
                    else:
                        with open(f"{self.model_location}/rnn_state-{self.episode_number}.json", 'r') as f:
                            print("Successfully loaded previous state.")
                            data = json.load(f)
                            self.init_rnn_state = (np.array(data["rnn_state_1"]), np.array(data["rnn_state_2"]))
                            self.init_rnn_state_ref = (np.array(data["rnn_state_ref_1"]), np.array(data["rnn_state_ref_2"]))

                        return

        # Init states for RNN - For steps, not training.
        if self.environment_params["use_dynamic_network"]:
            rnn_state_shapes = self.actor_network.get_rnn_state_shapes()
            self.init_rnn_state = tuple(
                (np.zeros([1, shape]), np.zeros([1, shape])) for shape in rnn_state_shapes)
            self.init_rnn_state_ref = tuple(
                (np.zeros([1, shape]), np.zeros([1, shape])) for shape in rnn_state_shapes)
        else:
            self.init_rnn_state = (
                np.zeros([1, self.actor_network.rnn_dim]),
                np.zeros([1, self.actor_network.rnn_dim]))
            self.init_rnn_state_ref = (
                np.zeros([1, self.actor_network.rnn_dim]),
                np.zeros([1, self.actor_network.rnn_dim]))
            if not self.sb_emulator or self.separate_networks:
                self.init_rnn_state_critic = (
                    np.zeros([1, self.critic_network.rnn_dim]),
                    np.zeros([1, self.critic_network.rnn_dim]))
                self.init_rnn_state_critic_ref = (
                    np.zeros([1, self.critic_network.rnn_dim]),
                    np.zeros([1, self.critic_network.rnn_dim]))

    def _episode_loop(self, a=None):
        self.update_sigmas()
        if a is None:
            a = [4.0, 0.0]

        rnn_state_actor = copy.copy(self.init_rnn_state)
        rnn_state_actor_ref = copy.copy(self.init_rnn_state_ref)
        rnn_state_critic = copy.copy(self.init_rnn_state_critic)
        rnn_state_critic_ref = copy.copy(self.init_rnn_state_critic_ref)

        if self.assay or self.just_trained:
            self.buffer.reset()
            self.just_trained = False
        self.simulation.reset()
        sa = np.zeros((1, 128))  # Kept for GIFs.

        o, r, internal_state, d, FOV = self.simulation.simulation_step(action=a, activations=(sa,))

        self.total_episode_reward = 0  # Total reward over episode

        action = a + [self.simulation.fish.prev_action_impulse,
                      self.simulation.fish.prev_action_angle]
        self.buffer.action_buffer.append(action)  # Add to buffer for loading of previous actions

        self.step_number = 0
        while self.step_number < self.current_episode_max_duration:
            # print(self.step_number)
            if self.assay is not None:
                # Deal with interventions
                if self.visual_interruptions is not None:
                    if self.visual_interruptions[self.step_number] == 1:
                        # mean values over all data
                        o[:, 0, :] = 4
                        o[:, 1, :] = 11
                        o[:, 2, :] = 16
                if self.reafference_interruptions is not None:
                    if self.reafference_interruptions[self.step_number] is not False:
                        a = [self.reafference_interruptions[self.step_number]]
                if self.preset_energy_state is not None:
                    if self.preset_energy_state[self.step_number] is not False:
                        self.simulation.fish.energy_level = self.preset_energy_state[self.step_number]
                        internal_state_order = self.get_internal_state_order()
                        index = internal_state_order.index("energy_state")
                        internal_state[0, index] = self.preset_energy_state[self.step_number]
                if self.in_light_interruptions is not False:
                    if self.in_light_interruptions[self.step_number] == 1:
                        internal_state_order = self.get_internal_state_order()
                        index = internal_state_order.index("in_light")
                        internal_state[0, index] = self.in_light_interruptions[self.step_number]
                if self.salt_interruptions is not False:
                    if self.salt_interruptions[self.step_number] == 1:
                        internal_state_order = self.get_internal_state_order()
                        index = internal_state_order.index("salt")
                        internal_state[0, index] = self.salt_interruptions[self.step_number]

                self.previous_action = a

            self.step_number += 1

            r, internal_state, o, d, rnn_state_actor, rnn_state_actor_ref, rnn_state_critic, rnn_state_critic_ref, a = self.step_loop(
                o=o,
                internal_state=internal_state,
                a=a,
                rnn_state_actor=rnn_state_actor,
                rnn_state_actor_ref=rnn_state_actor_ref,
                rnn_state_critic=rnn_state_critic,
                rnn_state_critic_ref=rnn_state_critic_ref
            )

            self.total_episode_reward += r
            if d:
                if "maintain_state" in self.learning_params:
                    if self.learning_params["maintain_state"]:
                        self.init_rnn_state = rnn_state_actor
                        self.init_rnn_state_ref = rnn_state_actor_ref
                break

    def compute_rnn_states(self, rnn_key_points, observation_buffer, internal_state_buffer, previous_action_buffer):
        num_actions = self.output_dimensions
        batch_size = len(rnn_key_points)

        if self.learning_params["rnn_state_computation"]:
            observation_buffer = np.vstack(observation_buffer)
            internal_state_buffer = np.vstack(internal_state_buffer)
            previous_action_buffer = np.reshape(previous_action_buffer, (observation_buffer.shape[0], num_actions))

            rnn_state_actor, rnn_state_actor_ref, rnn_state_critic, rnn_state_critic_ref = \
                self.buffer.get_rnn_batch([min(rnn_key_points)], 1)

            # rnn_state_actor = copy.copy(self.init_rnn_state_actor)
            # rnn_state_actor_ref = copy.copy(self.init_rnn_state_actor_ref)
            # rnn_state_critic = copy.copy(self.init_rnn_state_critic)
            # rnn_state_critic_ref #= copy.copy(self.init_rnn_state_critic_ref)

            observation_buffer_slice = observation_buffer#[min(rnn_key_points): max(rnn_key_points)+1]
            internal_state_buffer_slice = internal_state_buffer#[min(rnn_key_points): max(rnn_key_points)+1]
            previous_action_buffer_slice = previous_action_buffer#[min(rnn_key_points): max(rnn_key_points)+1]

            actor_rnn_state_buffer = ([rnn_state_actor[0][0]], [rnn_state_actor[1][0]])
            actor_rnn_state_ref_buffer = ([rnn_state_actor_ref[0][0]], [rnn_state_actor_ref[1][0]])
            critic_rnn_state_buffer = ([rnn_state_critic[0][0]], [rnn_state_critic[1][0]])
            critic_rnn_state_ref_buffer = ([rnn_state_critic_ref[0][0]], [rnn_state_critic_ref[1][0]])

            for step in range(min(rnn_key_points), max(rnn_key_points)+1):
                rnn_state_critic, rnn_state_critic_ref = self.sess.run(
                    [self.critic_network.rnn_state_shared, self.critic_network.rnn_state_ref],
                    feed_dict={self.critic_network.observation: observation_buffer_slice[step],
                               self.critic_network.prev_actions: previous_action_buffer_slice[step].reshape(1,
                                                                                                      num_actions),
                               self.critic_network.internal_state: internal_state_buffer_slice[step].reshape(1, 2),

                               self.critic_network.rnn_state_in: rnn_state_critic,
                               self.critic_network.rnn_state_in_ref: rnn_state_critic_ref,

                               self.critic_network.trainLength: 1,
                               self.critic_network.batch_size: 1,
                               })
                rnn_state_actor, rnn_state_actor_ref = self.sess.run(
                    [self.actor_network.rnn_state_shared, self.actor_network.rnn_state_ref],
                    feed_dict={self.actor_network.observation: observation_buffer_slice[step],
                               self.actor_network.prev_actions: previous_action_buffer_slice[step].reshape(1,
                                                                                                     num_actions),
                               self.actor_network.internal_state: internal_state_buffer_slice[step].reshape(1, 2),

                               self.actor_network.rnn_state_in: rnn_state_actor,
                               self.actor_network.rnn_state_in_ref: rnn_state_actor_ref,

                               self.actor_network.trainLength: 1,
                               self.actor_network.batch_size: 1,
                               })

                if step - 1 in rnn_key_points:
                    actor_rnn_state_buffer[0].append(rnn_state_actor[0][0])
                    actor_rnn_state_buffer[1].append(rnn_state_actor[1][0])

                    actor_rnn_state_ref_buffer[0].append(rnn_state_actor_ref[0][0])
                    actor_rnn_state_ref_buffer[1].append(rnn_state_actor_ref[1][0])

                    critic_rnn_state_buffer[0].append(rnn_state_critic[0][0])
                    critic_rnn_state_buffer[1].append(rnn_state_critic[1][0])

                    critic_rnn_state_ref_buffer[0].append(rnn_state_critic_ref[0][0])
                    critic_rnn_state_ref_buffer[1].append(rnn_state_critic_ref[1][0])

            actor_rnn_state_buffer = (np.array(actor_rnn_state_buffer[0]), np.array(actor_rnn_state_buffer[1]))
            actor_rnn_state_ref_buffer = (
            np.array(actor_rnn_state_ref_buffer[0]), np.array(actor_rnn_state_ref_buffer[1]))
            critic_rnn_state_buffer = (np.array(critic_rnn_state_buffer[0]), np.array(critic_rnn_state_buffer[1]))
            critic_rnn_state_ref_buffer = (
                np.array(critic_rnn_state_ref_buffer[0]), np.array(critic_rnn_state_ref_buffer[1]))
            # for step in range(max(rnn_key_points)):
            #     rnn_state_critic, rnn_state_critic_ref = self.sess.run(
            #         [self.critic_network.rnn_state_shared, self.critic_network.rnn_state_ref],
            #         feed_dict={self.critic_network.observation: observation_buffer[step],
            #                    self.critic_network.prev_actions: previous_action_buffer[step].reshape(1, num_actions),
            #                    self.critic_network.internal_state: internal_state_buffer[step].reshape(1, 2),
            #
            #                    self.critic_network.rnn_state_in: rnn_state_critic,
            #                    self.critic_network.rnn_state_in_ref: rnn_state_critic_ref,
            #
            #                    self.critic_network.trainLength: 1,
            #                    self.critic_network.batch_size: 1,
            #                    })
            #     rnn_state_actor, rnn_state_actor_ref = self.sess.run(
            #         [self.actor_network.rnn_state_shared, self.actor_network.rnn_state_ref],
            #         feed_dict={self.actor_network.observation: observation_buffer[step],
            #                    self.actor_network.prev_actions: previous_action_buffer[step].reshape(1, num_actions),
            #                    self.actor_network.internal_state: internal_state_buffer[step].reshape(1, 2),
            #
            #                    self.actor_network.rnn_state_in: rnn_state_actor,
            #                    self.actor_network.rnn_state_in_ref: rnn_state_actor_ref,
            #
            #                    self.actor_network.trainLength: 1,
            #                    self.actor_network.batch_size: 1,
            #                    })
            #
            #
            #     if step - 1 in rnn_key_points:
            #         actor_rnn_state_buffer[0].append(rnn_state_actor[0][0])
            #         actor_rnn_state_buffer[1].append(rnn_state_actor[1][0])
            #
            #         actor_rnn_state_ref_buffer[0].append(rnn_state_actor_ref[0][0])
            #         actor_rnn_state_ref_buffer[1].append(rnn_state_actor_ref[1][0])
            #
            #         critic_rnn_state_buffer[0].append(rnn_state_critic[0][0])
            #         critic_rnn_state_buffer[1].append(rnn_state_critic[1][0])
            #
            #         critic_rnn_state_ref_buffer[0].append(rnn_state_critic_ref[0][0])
            #         critic_rnn_state_ref_buffer[1].append(rnn_state_critic_ref[1][0])
            #
            # actor_rnn_state_buffer = (np.array(actor_rnn_state_buffer[0]), np.array(actor_rnn_state_buffer[1]))
            # actor_rnn_state_ref_buffer = (np.array(actor_rnn_state_ref_buffer[0]), np.array(actor_rnn_state_ref_buffer[1]))
            # critic_rnn_state_buffer = (np.array(critic_rnn_state_buffer[0]), np.array(critic_rnn_state_buffer[1]))
            # critic_rnn_state_ref_buffer = (
            #     np.array(critic_rnn_state_ref_buffer[0]), np.array(critic_rnn_state_ref_buffer[1]))
        else:
            actor_rnn_state_buffer, actor_rnn_state_ref_buffer, critic_rnn_state_buffer, critic_rnn_state_ref_buffer = \
                self.buffer.get_rnn_batch(rnn_key_points, batch_size)

            # actor_rnn_state_buffer = (
            #     np.zeros([batch_size, self.actor_network.rnn_dim]),
            #     np.zeros([batch_size, self.actor_network.rnn_dim]))  # Reset RNN hidden state
            # actor_rnn_state_ref_buffer = (
            #     np.zeros([batch_size, self.actor_network.rnn_dim]),
            #     np.zeros([batch_size, self.actor_network.rnn_dim]))  # Reset RNN hidden state
            # critic_rnn_state_buffer = (
            #     np.zeros([batch_size, self.actor_network.rnn_dim]),
            #     np.zeros([batch_size, self.actor_network.rnn_dim]))  # Reset RNN hidden state
            # critic_rnn_state_ref_buffer = (
            #     np.zeros([batch_size, self.actor_network.rnn_dim]),
            #     np.zeros([batch_size, self.actor_network.rnn_dim]))

        return actor_rnn_state_buffer, actor_rnn_state_ref_buffer, critic_rnn_state_buffer, critic_rnn_state_ref_buffer

    def compute_rnn_states2(self, rnn_key_points, observation_buffer, internal_state_buffer, previous_action_buffer, critic=False):
        num_actions = self.output_dimensions
        batch_size = len(rnn_key_points)

        if self.rnn_in_network:
            if self.learning_params["rnn_state_computation"]:
                observation_buffer = np.vstack(observation_buffer)
                internal_state_buffer = np.vstack(internal_state_buffer)
                previous_action_buffer = np.reshape(previous_action_buffer, (observation_buffer.shape[0], num_actions))

                rnn_state_actor, rnn_state_actor_ref = \
                    self.buffer.get_rnn_batch([min(rnn_key_points)], 1)

                # rnn_state_actor = copy.copy(self.init_rnn_state_actor)
                # rnn_state_actor_ref = copy.copy(self.init_rnn_state_actor_ref)
                # rnn_state_critic = copy.copy(self.init_rnn_state_critic)
                # rnn_state_critic_ref #= copy.copy(self.init_rnn_state_critic_ref)

                observation_buffer_slice = observation_buffer#[min(rnn_key_points): max(rnn_key_points)+1]
                internal_state_buffer_slice = internal_state_buffer#[min(rnn_key_points): max(rnn_key_points)+1]
                previous_action_buffer_slice = previous_action_buffer#[min(rnn_key_points): max(rnn_key_points)+1]

                actor_rnn_state_buffer = ([rnn_state_actor[0][0]], [rnn_state_actor[1][0]])
                actor_rnn_state_ref_buffer = ([rnn_state_actor_ref[0][0]], [rnn_state_actor_ref[1][0]])

                for step in range(min(rnn_key_points), max(rnn_key_points)+1):
                    rnn_state_actor, rnn_state_actor_ref = self.sess.run(
                        [self.actor_network.rnn_state_shared, self.actor_network.rnn_state_ref],
                        feed_dict={self.actor_network.observation: observation_buffer_slice[step],
                                   self.actor_network.prev_actions: previous_action_buffer_slice[step].reshape(1,
                                                                                                         num_actions),
                                   self.actor_network.internal_state: internal_state_buffer_slice[step].reshape(1, 2),

                                   self.actor_network.rnn_state_in: rnn_state_actor,
                                   self.actor_network.rnn_state_in_ref: rnn_state_actor_ref,

                                   self.actor_network.trainLength: 1,
                                   self.actor_network.batch_size: 1,
                                   })

                    if step - 1 in rnn_key_points:
                        actor_rnn_state_buffer[0].append(rnn_state_actor[0][0])
                        actor_rnn_state_buffer[1].append(rnn_state_actor[1][0])

                        actor_rnn_state_ref_buffer[0].append(rnn_state_actor_ref[0][0])
                        actor_rnn_state_ref_buffer[1].append(rnn_state_actor_ref[1][0])

                actor_rnn_state_buffer = (np.array(actor_rnn_state_buffer[0]), np.array(actor_rnn_state_buffer[1]))
                actor_rnn_state_ref_buffer = (
                np.array(actor_rnn_state_ref_buffer[0]), np.array(actor_rnn_state_ref_buffer[1]))

                # for step in range(max(rnn_key_points)):
                #     rnn_state_critic, rnn_state_critic_ref = self.sess.run(
                #         [self.critic_network.rnn_state_shared, self.critic_network.rnn_state_ref],
                #         feed_dict={self.critic_network.observation: observation_buffer[step],
                #                    self.critic_network.prev_actions: previous_action_buffer[step].reshape(1, num_actions),
                #                    self.critic_network.internal_state: internal_state_buffer[step].reshape(1, 2),
                #
                #                    self.critic_network.rnn_state_in: rnn_state_critic,
                #                    self.critic_network.rnn_state_in_ref: rnn_state_critic_ref,
                #
                #                    self.critic_network.trainLength: 1,
                #                    self.critic_network.batch_size: 1,
                #                    })
                #     rnn_state_actor, rnn_state_actor_ref = self.sess.run(
                #         [self.actor_network.rnn_state_shared, self.actor_network.rnn_state_ref],
                #         feed_dict={self.actor_network.observation: observation_buffer[step],
                #                    self.actor_network.prev_actions: previous_action_buffer[step].reshape(1, num_actions),
                #                    self.actor_network.internal_state: internal_state_buffer[step].reshape(1, 2),
                #
                #                    self.actor_network.rnn_state_in: rnn_state_actor,
                #                    self.actor_network.rnn_state_in_ref: rnn_state_actor_ref,
                #
                #                    self.actor_network.trainLength: 1,
                #                    self.actor_network.batch_size: 1,
                #                    })
                #
                #
                #     if step - 1 in rnn_key_points:
                #         actor_rnn_state_buffer[0].append(rnn_state_actor[0][0])
                #         actor_rnn_state_buffer[1].append(rnn_state_actor[1][0])
                #
                #         actor_rnn_state_ref_buffer[0].append(rnn_state_actor_ref[0][0])
                #         actor_rnn_state_ref_buffer[1].append(rnn_state_actor_ref[1][0])
                #
                #         critic_rnn_state_buffer[0].append(rnn_state_critic[0][0])
                #         critic_rnn_state_buffer[1].append(rnn_state_critic[1][0])
                #
                #         critic_rnn_state_ref_buffer[0].append(rnn_state_critic_ref[0][0])
                #         critic_rnn_state_ref_buffer[1].append(rnn_state_critic_ref[1][0])
                #
                # actor_rnn_state_buffer = (np.array(actor_rnn_state_buffer[0]), np.array(actor_rnn_state_buffer[1]))
                # actor_rnn_state_ref_buffer = (np.array(actor_rnn_state_ref_buffer[0]), np.array(actor_rnn_state_ref_buffer[1]))
                # critic_rnn_state_buffer = (np.array(critic_rnn_state_buffer[0]), np.array(critic_rnn_state_buffer[1]))
                # critic_rnn_state_ref_buffer = (
                #     np.array(critic_rnn_state_ref_buffer[0]), np.array(critic_rnn_state_ref_buffer[1]))
            else:
                actor_rnn_state_buffer, actor_rnn_state_ref_buffer = \
                    self.buffer.get_rnn_batch(rnn_key_points, batch_size, critic=critic)

                # actor_rnn_state_buffer = (
                #     np.zeros([batch_size, self.actor_network.rnn_dim]),
                #     np.zeros([batch_size, self.actor_network.rnn_dim]))  # Reset RNN hidden state
                # actor_rnn_state_ref_buffer = (
                #     np.zeros([batch_size, self.actor_network.rnn_dim]),
                #     np.zeros([batch_size, self.actor_network.rnn_dim]))  # Reset RNN hidden state
                # critic_rnn_state_buffer = (
                #     np.zeros([batch_size, self.actor_network.rnn_dim]),
                #     np.zeros([batch_size, self.actor_network.rnn_dim]))  # Reset RNN hidden state
                # critic_rnn_state_ref_buffer = (
                #     np.zeros([batch_size, self.actor_network.rnn_dim]),
                #     np.zeros([batch_size, self.actor_network.rnn_dim]))
        else:
            actor_rnn_state_buffer = ()
            actor_rnn_state_ref_buffer = ()


        return actor_rnn_state_buffer, actor_rnn_state_ref_buffer


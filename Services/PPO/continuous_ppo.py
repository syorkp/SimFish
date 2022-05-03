import math
import numpy as np

import tensorflow.compat.v1 as tf

from Networks.PPO.proximal_policy_optimizer_continuous import PPONetworkActor
from Networks.PPO.proximal_policy_optimizer_continuous_multivariate import PPONetworkActorMultivariate
from Networks.PPO.proximal_policy_optimizer_continuous_sb_emulator import PPONetworkActorMultivariate2
from Networks.PPO.proximal_policy_optimizer_continuous_beta_sb_emulator import PPONetworkActorMultivariateBetaNormal2
from Networks.PPO.proximal_policy_optimizer_continuous_sb_emulator_dynamic import PPONetworkActorMultivariate2Dynamic
from Networks.PPO.proximal_policy_optimizer_continuous_sb_emulator_extended import PPONetworkActorMultivariate2Extended
from Networks.RND.rnd import RandomNetworkDistiller
from Services.PPO.base_ppo import BasePPO

tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)


class ContinuousPPO(BasePPO):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("PPO Constructor called")

        self.continuous = True

        self.impulse_sigma = None
        self.angle_sigma = None
        self.multivariate = None
        self.sigma_total_steps = 0

        # Placeholders
        self.epsilon_greedy = None
        self.e = None
        self.step_drop = None

        self.output_dimensions = 2

    def create_network(self):
        """
        Create the main and target Q networks, according to the configuration parameters.
        :return: The main network and the target network graphs.
        """
        actor_cell, internal_states, internal_state_names = BasePPO.create_network(self)

        if self.multivariate:
            if self.environment_params["use_dynamic_network"]:
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
                                                                         # new_simulation=self.new_simulation,
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
                                                                         )
            else:
                # TODO: Remove:

                # self.actor_network = PPONetworkActorMultivariate2Extended(simulation=self.simulation,
                #                                                   rnn_dim=self.learning_params['rnn_dim_shared'],
                #                                                   rnn_cell=actor_cell,
                #                                                   my_scope='actor',
                #                                                   internal_states=internal_states,
                #                                                   max_impulse=self.environment_params['max_impulse'],
                #                                                   max_angle_change=self.environment_params[
                #                                                       'max_angle_change'],
                #                                                   clip_param=self.environment_params['clip_param'],
                #                                                   input_sigmas=self.learning_params['input_sigmas'],
                #                                                   new_simulation=self.new_simulation,
                #                                                   impose_action_mask=self.environment_params[
                #                                                       'impose_action_mask'],
                #                                                   )
                if self.learning_params["beta_distribution"]:
                    self.actor_network = PPONetworkActorMultivariateBetaNormal2(simulation=self.simulation,
                                                                      rnn_dim=self.learning_params['rnn_dim_shared'],
                                                                      rnn_cell=actor_cell,
                                                                      my_scope='actor',
                                                                      internal_states=internal_states,
                                                                      max_impulse=self.environment_params[
                                                                          'max_impulse'],
                                                                      max_angle_change=self.environment_params[
                                                                          'max_angle_change'],
                                                                      clip_param=self.environment_params['clip_param'],
                                                                      input_sigmas=self.learning_params['input_sigmas'],
                                                                      new_simulation=self.new_simulation,
                                                                      impose_action_mask=self.environment_params[
                                                                          'impose_action_mask'],
                                                                      )

                else:
                    self.actor_network = PPONetworkActorMultivariate2(simulation=self.simulation,
                                                                      rnn_dim=self.learning_params['rnn_dim_shared'],
                                                                      rnn_cell=actor_cell,
                                                                      my_scope='actor',
                                                                      internal_states=internal_states,
                                                                      max_impulse=self.environment_params['max_impulse'],
                                                                      max_angle_change=self.environment_params[
                                                                          'max_angle_change'],
                                                                      clip_param=self.environment_params['clip_param'],
                                                                      input_sigmas=self.learning_params['input_sigmas'],
                                                                      new_simulation=self.new_simulation,
                                                                      impose_action_mask=self.environment_params[
                                                                          'impose_action_mask'],
                                                                      )

            if self.sb_emulator:
                pass

            else:
                self.actor_network = PPONetworkActorMultivariate(simulation=self.simulation,
                                                                 rnn_dim=self.learning_params['rnn_dim_shared'],
                                                                 rnn_cell=actor_cell,
                                                                 my_scope='actor',
                                                                 internal_states=internal_states,
                                                                 max_impulse=self.environment_params['max_impulse'],
                                                                 max_angle_change=self.environment_params[
                                                                     'max_angle_change'],
                                                                 clip_param=self.environment_params['clip_param'],
                                                                 input_sigmas=self.learning_params['input_sigmas'],
                                                                 new_simulation=self.new_simulation,
                                                                 impose_action_mask=self.environment_params['impose_action_mask'],
                                                                 impulse_scaling=self.environment_params[
                                                                     'impulse_scaling'],
                                                                 angle_scaling=self.environment_params['angle_scaling'],
                                                                 )

        else:
            self.actor_network = PPONetworkActor(simulation=self.simulation,
                                                 rnn_dim=self.learning_params['rnn_dim_shared'],
                                                 rnn_cell=actor_cell,
                                                 my_scope='actor',
                                                 internal_states=internal_states,
                                                 max_impulse=self.environment_params['max_impulse'],
                                                 max_angle_change=self.environment_params['max_angle_change'],
                                                 clip_param=self.environment_params['clip_param'],
                                                 beta_impulse=self.learning_params['beta_distribution'],
                                                 new_simulation=self.new_simulation,
                                                 impose_action_mask=self.environment_params['impose_action_mask'],

                                                 )
        print("Created network")

        if self.use_rnd:
            print("Creating Random network distillation networks...")
            self.target_rdn = RandomNetworkDistiller(
                simulation=self.simulation,
                my_scope='target_rdn',
                internal_states=internal_states,
                predictor=False,
                new_simulation=self.new_simulation,
            )
            self.predictor_rdn = RandomNetworkDistiller(
                simulation=self.simulation,
                my_scope='predictor_rdn',
                internal_states=internal_states,
                predictor=True,
                new_simulation=self.new_simulation,
            )

    def update_sigmas(self):
        self.sigma_total_steps += self.simulation.num_steps

        if self.environment_params["sigma_mode"] != "Decreasing":
            self.impulse_sigma = np.array([self.environment_params["max_sigma_impulse"]])
            self.angle_sigma = np.array([self.environment_params["max_sigma_angle"]])
        else:
            # Linear scale
            self.impulse_sigma = np.array([self.environment_params["max_sigma_impulse"] - (
                    self.environment_params["max_sigma_impulse"] - self.environment_params["min_sigma_impulse"]) * (
                                                   self.sigma_total_steps / self.environment_params["sigma_reduction_time"])])
            self.angle_sigma = np.array([self.environment_params["max_sigma_angle"] - (
                    self.environment_params["max_sigma_angle"] - self.environment_params["min_sigma_angle"]) * (
                                                 self.sigma_total_steps / self.environment_params["sigma_reduction_time"])])
            # To prevent ever returning NaN
            if math.isnan(self.impulse_sigma[0]):
                self.impulse_sigma = np.array([self.environment_params["min_sigma_impulse"]])
            if math.isnan(self.angle_sigma[0]):
                self.angle_sigma = np.array([self.environment_params["min_sigma_angle"]])

    def _episode_loop(self, a=None):
        self.update_sigmas()
        a = [4.0, 0.0]
        super(ContinuousPPO, self)._episode_loop(a)

    def _assay_step_loop_multivariate2(self, o, internal_state, a, rnn_state_actor, rnn_state_actor_ref,
                                           rnn_state_critic,
                                           rnn_state_critic_ref):
        if self.new_simulation and self.environment_params["use_dynamic_network"]:
            return self._assay_step_loop_multivariate2_new(o, internal_state, a, rnn_state_actor, rnn_state_actor_ref,
                                      rnn_state_critic,
                                      rnn_state_critic_ref)
        else:
            return self._assay_step_loop_multivariate2_old(o, internal_state, a, rnn_state_actor, rnn_state_actor_ref,
                                      rnn_state_critic,
                                      rnn_state_critic_ref)

    def _assay_step_loop_multivariate2_new(self, o, internal_state, a, rnn_state_actor, rnn_state_actor_ref,
                                      rnn_state_critic,
                                      rnn_state_critic_ref):
        sa = np.zeros((1, 128))  # Placeholder for the state advantage stream.
        a = [a[0] / self.environment_params['max_impulse'],
             a[1] / self.environment_params['max_angle_change']]  # Set impulse to scale to be inputted to network

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
                       self.actor_network.prev_actions: np.reshape(a, (1, 2)),
                       self.actor_network.sigma_impulse_combined_proto: self.impulse_sigma,
                       self.actor_network.sigma_angle_combined_proto: self.angle_sigma,
                       self.actor_network.rnn_state_in: rnn_state_actor,
                       self.actor_network.rnn_state_in_ref: rnn_state_actor_ref,
                       self.actor_network.batch_size: 1,
                       self.actor_network.train_length: 1,
                       }
        )

        if self.use_mu:
            action = [mu_i[0][0], mu_a[0][0]]
        else:
            action = [impulse[0][0], angle[0][0]]

        o1, given_reward, new_internal_state, d, self.frame_buffer = self.simulation.simulation_step(action=action,
                                                                                                     frame_buffer=self.frame_buffer,
                                                                                                     save_frames=True,
                                                                                                     activations=(sa,))

        sand_grain_positions, prey_positions, predator_position, vegetation_positions = self.get_positions()

        # Update buffer
        self.buffer.add_training(observation=o,
                                 internal_state=internal_state,
                                 action=action,
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
            self.buffer.save_environmental_positions(self.simulation.fish.body.position,
                                                     self.simulation.prey_consumed_this_step,
                                                     self.simulation.predator_body,
                                                     prey_positions,
                                                     predator_position,
                                                     sand_grain_positions,
                                                     vegetation_positions,
                                                     self.simulation.fish.body.angle,
                                                     )
        self.buffer.make_desired_recordings(network_layers)

        return given_reward, new_internal_state, o1, d, updated_rnn_state_actor, updated_rnn_state_actor_ref, \
               0, 0

    def _assay_step_loop_multivariate2_old(self, o, internal_state, a, rnn_state_actor, rnn_state_actor_ref,
                                      rnn_state_critic,
                                      rnn_state_critic_ref):
        sa = np.zeros((1, 128))  # Placeholder for the state advantage stream.
        a = [a[0] / self.environment_params['max_impulse'],
             a[1] / self.environment_params['max_angle_change']]  # Set impulse to scale to be inputted to network

        impulse, angle, V, updated_rnn_state_actor, updated_rnn_state_actor_ref, conv1l_actor, conv2l_actor, conv3l_actor, \
        conv4l_actor, conv1r_actor, conv2r_actor, conv3r_actor, conv4r_actor, \
        mu_i, mu_a, mu1, mu1_ref, mu_a1, mu_a_ref, si_i, si_a = self.sess.run(
            [self.actor_network.impulse_output, self.actor_network.angle_output, self.actor_network.value_output,
             self.actor_network.rnn_state_shared, self.actor_network.rnn_state_ref,
             self.actor_network.conv1l, self.actor_network.conv2l, self.actor_network.conv3l,
             self.actor_network.conv4l,
             self.actor_network.conv1r, self.actor_network.conv2r, self.actor_network.conv3r,
             self.actor_network.conv4r,
             self.actor_network.mu_impulse_combined,
             self.actor_network.mu_angle_combined,
             self.actor_network.mu_impulse,
             self.actor_network.mu_impulse_ref,
             self.actor_network.mu_angle, self.actor_network.mu_angle_ref,
             self.actor_network.sigma_impulse_combined, self.actor_network.sigma_angle_combined,
             ],
            feed_dict={self.actor_network.observation: o,
                       self.actor_network.internal_state: internal_state,
                       self.actor_network.prev_actions: np.reshape(a, (1, 2)),
                       self.actor_network.sigma_impulse_combined_proto: self.impulse_sigma,
                       self.actor_network.sigma_angle_combined_proto: self.angle_sigma,
                       self.actor_network.rnn_state_in: rnn_state_actor,
                       self.actor_network.rnn_state_in_ref: rnn_state_actor_ref,
                       self.actor_network.batch_size: 1,
                       self.actor_network.train_length: 1,
                       }
        )

        if self.use_mu:
            action = [mu_i[0][0], mu_a[0][0]]
        else:
            action = [impulse[0][0], angle[0][0]]

        o1, given_reward, new_internal_state, d, self.frame_buffer = self.simulation.simulation_step(action=action,
                                                                                                     frame_buffer=self.frame_buffer,
                                                                                                     save_frames=True,
                                                                                                     activations=(sa,))

        sand_grain_positions, prey_positions, predator_position, vegetation_positions = self.get_positions()

        # Update buffer
        self.buffer.add_training(observation=o,
                                 internal_state=internal_state,
                                 action=action,
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
            self.buffer.save_environmental_positions(self.simulation.fish.body.position,
                                                     self.simulation.prey_consumed_this_step,
                                                     self.simulation.predator_body,
                                                     prey_positions,
                                                     predator_position,
                                                     sand_grain_positions,
                                                     vegetation_positions,
                                                     self.simulation.fish.body.angle,
                                                     )
        if "convolutional layers" in self.buffer.recordings:
            self.buffer.save_conv_states(conv1l_actor, conv2l_actor, conv3l_actor, conv4l_actor, conv1r_actor,
                                         conv2r_actor, conv3r_actor, conv4r_actor,
                                         0, 0, 0, 0, 0,
                                         0, 0, 0)

        return given_reward, new_internal_state, o1, d, updated_rnn_state_actor, updated_rnn_state_actor_ref, \
               0, 0

    def _assay_step_loop_multivariate(self, o, internal_state, a, rnn_state_actor, rnn_state_actor_ref,
                                      rnn_state_critic,
                                      rnn_state_critic_ref):
        sa = np.zeros((1, 128))  # Placeholder for the state advantage stream.
        a = [a[0] / self.environment_params['max_impulse'],
             a[1] / self.environment_params['max_angle_change']]  # Set impulse to scale to be inputted to network

        impulse, angle, updated_rnn_state_actor, updated_rnn_state_actor_ref, conv1l_actor, conv2l_actor, conv3l_actor, \
        conv4l_actor, conv1r_actor, conv2r_actor, conv3r_actor, conv4r_actor, action_probability, \
        mu_i, si_i, mu_a, si_a, mu1, mu1_ref, mu_a1, mu_a_ref = self.sess.run(
            [self.actor_network.impulse_output, self.actor_network.angle_output,
             self.actor_network.rnn_state_shared,
             self.actor_network.rnn_state_ref,
             self.actor_network.conv1l, self.actor_network.conv2l, self.actor_network.conv3l,
             self.actor_network.conv4l,
             self.actor_network.conv1r, self.actor_network.conv2r, self.actor_network.conv3r,
             self.actor_network.conv4r,

             self.actor_network.log_prob,
             self.actor_network.mu_impulse_combined, self.actor_network.sigma_impulse_combined,
             self.actor_network.mu_angle_combined,
             self.actor_network.sigma_angle_combined, self.actor_network.mu_impulse,
             self.actor_network.mu_impulse_ref,
             self.actor_network.mu_angle, self.actor_network.mu_angle_ref
             ],
            feed_dict={self.actor_network.observation: o,
                       self.actor_network.internal_state: internal_state,
                       self.actor_network.prev_actions: np.reshape(a, (1, 2)),
                       self.actor_network.rnn_state_in: rnn_state_actor,
                       self.actor_network.rnn_state_in_ref: rnn_state_actor_ref,
                       self.actor_network.batch_size: 1,
                       self.actor_network.train_length: 1,
                       }
        )

        V, updated_rnn_state_critic, updated_rnn_state_critic_ref, conv1l_critic, conv2l_critic, conv3l_critic, \
        conv4l_critic, conv1r_critic, conv2r_critic, conv3r_critic, conv4r_critic = self.sess.run(
            [self.critic_network.Value_output, self.critic_network.rnn_state_shared,
             self.critic_network.rnn_state_ref,
             self.critic_network.conv1l, self.critic_network.conv2l, self.critic_network.conv3l,
             self.critic_network.conv4l,
             self.critic_network.conv1r, self.critic_network.conv2r, self.critic_network.conv3r,
             self.critic_network.conv4r,
             ],
            feed_dict={self.critic_network.observation: o,
                       self.critic_network.internal_state: internal_state,
                       self.critic_network.prev_actions: np.reshape(a, (1, 2)),
                       self.critic_network.rnn_state_in: rnn_state_critic,
                       self.critic_network.rnn_state_in_ref: rnn_state_critic_ref,
                       self.critic_network.batch_size: 1,
                       self.critic_network.train_length: 1,
                       }
        )
        action = [impulse[0][0], angle[0][0]]

        o1, given_reward, new_internal_state, d, self.frame_buffer = self.simulation.simulation_step(action=action,
                                                                                                     frame_buffer=self.frame_buffer,
                                                                                                     save_frames=True,
                                                                                                     activations=(sa,))

        sand_grain_positions, prey_positions, predator_position, vegetation_positions = self.get_positions()

        # Update buffer
        self.buffer.add_training(observation=o,
                                 internal_state=internal_state,
                                 action=action,
                                 reward=given_reward,
                                 value=V,
                                 l_p_action=action_probability,
                                 actor_rnn_state=rnn_state_actor,
                                 actor_rnn_state_ref=rnn_state_actor_ref,
                                 critic_rnn_state=rnn_state_critic,
                                 critic_rnn_state_ref=rnn_state_critic_ref,
                                 )
        self.buffer.add_logging(mu_i, si_i, mu_a, si_a, mu1, mu1_ref, mu_a1, mu_a_ref)

        if "environmental positions" in self.buffer.recordings:
            self.buffer.save_environmental_positions(self.simulation.fish.body.position,
                                                     self.simulation.prey_consumed_this_step,
                                                     self.simulation.predator_body,
                                                     prey_positions,
                                                     predator_position,
                                                     sand_grain_positions,
                                                     vegetation_positions,
                                                     self.simulation.fish.body.angle,
                                                     )
        if "convolutional layers" in self.buffer.recordings:
            self.buffer.save_conv_states(conv1l_actor, conv2l_actor, conv3l_actor, conv4l_actor, conv1r_actor,
                                         conv2r_actor, conv3r_actor, conv4r_actor,
                                         conv1l_critic, conv2l_critic, conv3l_critic, conv4l_critic, conv1r_critic,
                                         conv2r_critic, conv3r_critic, conv4r_critic)

        return given_reward, new_internal_state, o1, d, updated_rnn_state_actor, updated_rnn_state_actor_ref, \
               updated_rnn_state_critic, updated_rnn_state_critic_ref

    def _assay_step_loop(self, o, internal_state, a, rnn_state_actor, rnn_state_actor_ref, rnn_state_critic,
                         rnn_state_critic_ref):
        sa = np.zeros((1, 128))  # Placeholder for the state advantage stream.
        a = [a[0] / self.environment_params['max_impulse'],
             a[1] / self.environment_params['max_angle_change']]  # Set impulse to scale to be inputted to network

        impulse, angle, updated_rnn_state_actor, updated_rnn_state_actor_ref, conv1l_actor, conv2l_actor, conv3l_actor, \
        conv4l_actor, conv1r_actor, conv2r_actor, conv3r_actor, conv4r_actor, impulse_probability, \
        angle_probability, mu_i, si_i, mu_a, si_a, mu1, mu1_ref, mu_a1, mu_a_ref = self.sess.run(
            [self.actor_network.impulse_output, self.actor_network.angle_output,
             self.actor_network.rnn_state_shared,
             self.actor_network.rnn_state_ref,
             self.actor_network.conv1l, self.actor_network.conv2l, self.actor_network.conv3l,
             self.actor_network.conv4l,
             self.actor_network.conv1r, self.actor_network.conv2r, self.actor_network.conv3r,
             self.actor_network.conv4r,

             self.actor_network.log_prob_impulse, self.actor_network.log_prob_angle,
             self.actor_network.mu_impulse_combined, self.actor_network.sigma_impulse_combined,
             self.actor_network.mu_angle_combined,
             self.actor_network.sigma_angle_combined, self.actor_network.mu_impulse,
             self.actor_network.mu_impulse_ref,
             self.actor_network.mu_angle, self.actor_network.mu_angle_ref
             ],
            feed_dict={self.actor_network.observation: o,
                       self.actor_network.internal_state: internal_state,
                       self.actor_network.prev_actions: np.reshape(a, (1, 2)),
                       self.actor_network.rnn_state_in: rnn_state_actor,
                       self.actor_network.rnn_state_in_ref: rnn_state_actor_ref,
                       self.actor_network.batch_size: 1,
                       self.actor_network.train_length: 1,
                       self.actor_network.sigma_impulse_combined: self.impulse_sigma,
                       self.actor_network.sigma_angle_combined: self.angle_sigma,
                       }
        )

        V, updated_rnn_state_critic, updated_rnn_state_critic_ref, conv1l_critic, conv2l_critic, conv3l_critic, \
        conv4l_critic, conv1r_critic, conv2r_critic, conv3r_critic, conv4r_critic = self.sess.run(
            [self.critic_network.Value_output, self.critic_network.rnn_state_shared,
             self.critic_network.rnn_state_ref,
             self.critic_network.conv1l, self.critic_network.conv2l, self.critic_network.conv3l,
             self.critic_network.conv4l,
             self.critic_network.conv1r, self.critic_network.conv2r, self.critic_network.conv3r,
             self.critic_network.conv4r,
             ],
            feed_dict={self.critic_network.observation: o,
                       self.critic_network.internal_state: internal_state,
                       self.critic_network.prev_actions: np.reshape(a, (1, 2)),
                       self.critic_network.rnn_state_in: rnn_state_critic,
                       self.critic_network.rnn_state_in_ref: rnn_state_critic_ref,
                       self.critic_network.batch_size: 1,
                       self.critic_network.train_length: 1,
                       }
        )

        if self.learning_params["beta_distribution"]:
            impulse = [[impulse[0]]]
            if impulse[0][0] == 0.0:
                impulse = [[0.001]]

        action = [impulse[0][0], angle[0][0]]

        o1, given_reward, new_internal_state, d, self.frame_buffer = self.simulation.simulation_step(action=action,
                                                                                                     frame_buffer=self.frame_buffer,
                                                                                                     save_frames=True,
                                                                                                     activations=(sa,))

        sand_grain_positions, prey_positions, predator_position, vegetation_positions = self.get_positions()

        # Update buffer
        self.buffer.add_training(observation=o,
                                 internal_state=internal_state,
                                 action=action,
                                 reward=given_reward,
                                 value=V,
                                 l_p_impulse=impulse_probability,
                                 l_p_angle=angle_probability,
                                 actor_rnn_state=rnn_state_actor,
                                 actor_rnn_state_ref=rnn_state_actor_ref,
                                 critic_rnn_state=rnn_state_critic,
                                 critic_rnn_state_ref=rnn_state_critic_ref,
                                 )
        self.buffer.add_logging(mu_i, si_i, mu_a, si_a, mu1, mu1_ref, mu_a1, mu_a_ref)

        if "environmental positions" in self.buffer.recordings:
            self.buffer.save_environmental_positions(self.simulation.fish.body.position,
                                                     self.simulation.prey_consumed_this_step,
                                                     self.simulation.predator_body,
                                                     prey_positions,
                                                     predator_position,
                                                     sand_grain_positions,
                                                     vegetation_positions,
                                                     self.simulation.fish.body.angle,
                                                     )
        if "convolutional layers" in self.buffer.recordings:
            self.buffer.save_conv_states(conv1l_actor, conv2l_actor, conv3l_actor, conv4l_actor, conv1r_actor,
                                         conv2r_actor, conv3r_actor, conv4r_actor,
                                         conv1l_critic, conv2l_critic, conv3l_critic, conv4l_critic, conv1r_critic,
                                         conv2r_critic, conv3r_critic, conv4r_critic)

        return given_reward, new_internal_state, o1, d, updated_rnn_state_actor, updated_rnn_state_actor_ref, \
               updated_rnn_state_critic, updated_rnn_state_critic_ref

    def _step_loop_multivariate_beta_sbe_full_logs(self, o, internal_state, a, rnn_state_actor, rnn_state_actor_ref,
                                               rnn_state_critic,
                                               rnn_state_critic_ref):

        sa = np.zeros((1, 128))  # Placeholder for the state advantage stream.
        a = [a[0] / self.environment_params['max_impulse'],
             a[1] / self.environment_params['max_angle_change']]  # Set impulse to scale to be inputted to network

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
                       self.actor_network.prev_actions: np.reshape(a, (1, 2)),
                       self.actor_network.sigma_impulse_combined_proto: self.impulse_sigma,
                       self.actor_network.sigma_angle_combined_proto: self.angle_sigma,
                       self.actor_network.rnn_state_in: rnn_state_actor,
                       self.actor_network.rnn_state_in_ref: rnn_state_actor_ref,
                       self.actor_network.batch_size: 1,
                       self.actor_network.train_length: 1,
                       }
        )

        if self.epsilon_greedy:
            if np.random.rand(1) < self.e:
                action = [impulse[0][0], angle[0][0]]
            else:
                action = [impulse[0][0], mu_a[0][0] * self.environment_params["max_angle_change"]]

                # And get updated neg_log_prob
                neg_log_action_probability = self.sess.run(
                    [self.actor_network.new_neg_log_prob],

                    feed_dict={self.actor_network.observation: o,
                               self.actor_network.internal_state: internal_state,
                               self.actor_network.prev_actions: np.reshape(a, (1, 2)),
                               self.actor_network.sigma_impulse_combined_proto: self.impulse_sigma,
                               self.actor_network.sigma_angle_combined_proto: self.angle_sigma,
                               self.actor_network.rnn_state_in: rnn_state_actor,
                               self.actor_network.rnn_state_in_ref: rnn_state_actor_ref,
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
        o1, r, new_internal_state, d, self.frame_buffer = self.simulation.simulation_step(
            action=action,
            frame_buffer=self.frame_buffer,
            save_frames=self.save_frames,
            activations=sa)


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

        si_i = si[0][0]
        si_a = si[0][1]
        self.buffer.add_logging(mu_i, si_i, mu_a, si_a, 0, 0, 0, 0)

        self.total_steps += 1
        return r, new_internal_state, o1, d, updated_rnn_state_actor, updated_rnn_state_actor_ref, 0, 0

    def _step_loop_multivariate_sbe_full_logs(self, o, internal_state, a, rnn_state_actor, rnn_state_actor_ref,
                                               rnn_state_critic,
                                               rnn_state_critic_ref):

        sa = np.zeros((1, 128))  # Placeholder for the state advantage stream.
        a = [a[0] / self.environment_params['max_impulse'],
             a[1] / self.environment_params['max_angle_change']]  # Set impulse to scale to be inputted to network

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
                       self.actor_network.prev_actions: np.reshape(a, (1, 2)),
                       self.actor_network.sigma_impulse_combined_proto: self.impulse_sigma,
                       self.actor_network.sigma_angle_combined_proto: self.angle_sigma,
                       self.actor_network.rnn_state_in: rnn_state_actor,
                       self.actor_network.rnn_state_in_ref: rnn_state_actor_ref,
                       self.actor_network.batch_size: 1,
                       self.actor_network.train_length: 1,
                       }
        )

        if self.epsilon_greedy:
            if np.random.rand(1) < self.e:
                action = [impulse[0][0], angle[0][0]]
            else:
                action = [mu_i[0][0] * self.environment_params["max_impulse"], mu_a[0][0] * self.environment_params["max_angle_change"]]

                # And get updated neg_log_prob
                neg_log_action_probability = self.sess.run(
                    [self.actor_network.new_neg_log_prob],

                    feed_dict={self.actor_network.observation: o,
                               self.actor_network.internal_state: internal_state,
                               self.actor_network.prev_actions: np.reshape(a, (1, 2)),
                               self.actor_network.sigma_impulse_combined_proto: self.impulse_sigma,
                               self.actor_network.sigma_angle_combined_proto: self.angle_sigma,
                               self.actor_network.rnn_state_in: rnn_state_actor,
                               self.actor_network.rnn_state_in_ref: rnn_state_actor_ref,
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
        o1, r, new_internal_state, d, self.frame_buffer = self.simulation.simulation_step(
            action=action,
            frame_buffer=self.frame_buffer,
            save_frames=self.save_frames,
            activations=sa)

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

        si_i = si[0][0]
        si_a = si[0][1]
        self.buffer.add_logging(mu_i, si_i, mu_a, si_a, 0, 0, 0, 0)

        self.total_steps += 1
        return r, new_internal_state, o1, d, updated_rnn_state_actor, updated_rnn_state_actor_ref, 0, 0

    def _step_loop_multivariate_sbe_reduced_logs(self, o, internal_state, a, rnn_state_actor, rnn_state_actor_ref,
                                               rnn_state_critic,
                                               rnn_state_critic_ref):

        sa = np.zeros((1, 128))  # Placeholder for the state advantage stream.
        a = [a[0] / self.environment_params['max_impulse'],
             a[1] / self.environment_params['max_angle_change']]  # Set impulse to scale to be inputted to network

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
                       self.actor_network.rnn_state_in_ref: rnn_state_actor_ref,
                       self.actor_network.batch_size: 1,
                       self.actor_network.train_length: 1,
                       }
        )
        if self.epsilon_greedy:
            if np.random.rand(1) < self.e:
                action = [impulse[0][0], angle[0][0]]
            else:
                action = [mu_i[0][0] * self.environment_params["max_impulse"], mu_a[0][0] * self.environment_params["max_angle_change"]]

                # And get updated neg_log_prob
                neg_log_action_probability = self.sess.run(
                    [self.actor_network.new_neg_log_prob],

                    feed_dict={self.actor_network.observation: o,
                               self.actor_network.internal_state: internal_state,
                               self.actor_network.prev_actions: np.reshape(a, (1, 2)),
                               self.actor_network.sigma_impulse_combined_proto: self.impulse_sigma,
                               self.actor_network.sigma_angle_combined_proto: self.angle_sigma,
                               self.actor_network.rnn_state_in: rnn_state_actor,
                               self.actor_network.rnn_state_in_ref: rnn_state_actor_ref,
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
        o1, r, new_internal_state, d, self.frame_buffer = self.simulation.simulation_step(
            action=action,
            frame_buffer=self.frame_buffer,
            save_frames=self.save_frames,
            activations=sa)

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
        return r, new_internal_state, o1, d, updated_rnn_state_actor, updated_rnn_state_actor_ref, 0, 0

    def _step_loop_multivariate_full_logs(self, o, internal_state, a, rnn_state_actor, rnn_state_actor_ref,
                                              rnn_state_critic,
                                              rnn_state_critic_ref):
        sa = np.zeros((1, 128))  # Placeholder for the state advantage stream.
        a = [a[0] / self.environment_params['max_impulse'],
             a[1] / self.environment_params['max_angle_change']]  # Set impulse to scale to be inputted to network

        # impulse, angle, updated_rnn_state_actor, updated_rnn_state_actor_ref, action_probability, mu_i, si_i, mu_a, \
        # si_a, mu1, mu1_ref, mu_a1, mu_a_ref = self.sess.run(
        #     [self.actor_network.impulse_output, self.actor_network.angle_output, self.actor_network.rnn_state_shared,
        #      self.actor_network.rnn_state_ref,
        #      self.actor_network.log_prob,
        #      self.actor_network.mu_impulse_combined, self.actor_network.sigma_impulse_combined,
        #      self.actor_network.mu_angle_combined,
        #      self.actor_network.sigma_angle_combined, self.actor_network.mu_impulse, self.actor_network.mu_impulse_ref,
        #      self.actor_network.mu_angle, self.actor_network.mu_angle_ref
        #      ],
        #     feed_dict={self.actor_network.observation: o,
        #                self.actor_network.internal_state: internal_state,
        #                self.actor_network.prev_actions: np.reshape(a, (1, 2)),
        #                self.actor_network.rnn_state_in: rnn_state_actor,
        #                self.actor_network.rnn_state_in_ref: rnn_state_actor_ref,
        #                self.actor_network.batch_size: 1,
        #                self.actor_network.train_length: 1,
        #                }
        # )

        impulse, angle, updated_rnn_state_actor, updated_rnn_state_actor_ref, action_probability, mu_i, mu_a, \
        si, mu1, mu1_ref, mu_a1, mu_a_ref = self.sess.run(
            [self.actor_network.impulse_output, self.actor_network.angle_output, self.actor_network.rnn_state_shared,
             self.actor_network.rnn_state_ref,
             self.actor_network.log_prob,
             self.actor_network.mu_impulse_combined,
             self.actor_network.mu_angle_combined,
             self.actor_network.sigma_action, self.actor_network.mu_impulse, self.actor_network.mu_impulse_ref,
             self.actor_network.mu_angle, self.actor_network.mu_angle_ref,
            ],

            feed_dict={self.actor_network.observation: o,
                       self.actor_network.internal_state: internal_state,
                       self.actor_network.prev_actions: np.reshape(a, (1, 2)),
                       self.actor_network.sigma_impulse_combined_proto: self.impulse_sigma,
                       self.actor_network.sigma_angle_combined_proto: self.angle_sigma,
                       self.actor_network.rnn_state_in: rnn_state_actor,
                       self.actor_network.rnn_state_in_ref: rnn_state_actor_ref,
                       self.actor_network.batch_size: 1,
                       self.actor_network.train_length: 1,
                       }
        )

        V, updated_rnn_state_critic, updated_rnn_state_critic_ref = self.sess.run(
            [self.critic_network.Value_output, self.critic_network.rnn_state_shared,
             self.critic_network.rnn_state_ref],
            feed_dict={self.critic_network.observation: o,
                       self.critic_network.internal_state: internal_state,
                       self.critic_network.prev_actions: np.reshape(a, (1, 2)),
                       self.critic_network.rnn_state_in: rnn_state_critic,
                       self.critic_network.rnn_state_in_ref: rnn_state_critic_ref,
                       self.critic_network.batch_size: 1,
                       self.critic_network.train_length: 1,
                       }
        )

        action = [impulse[0][0], angle[0][0]]

        # Simulation step
        o1, r, new_internal_state, d, self.frame_buffer = self.simulation.simulation_step(
            action=action,
            frame_buffer=self.frame_buffer,
            save_frames=self.save_frames,
            activations=sa)

        # Update buffer
        self.buffer.add_training(observation=o,
                                 internal_state=internal_state,
                                 action=action,
                                 reward=r,
                                 value=V,
                                 l_p_action=action_probability,
                                 actor_rnn_state=rnn_state_actor,
                                 actor_rnn_state_ref=rnn_state_actor_ref,
                                 critic_rnn_state=rnn_state_critic,
                                 critic_rnn_state_ref=rnn_state_critic_ref,
                                 )

        si_i = si[0][0]
        si_a = si[0][1]
        self.buffer.add_logging(mu_i, si_i, mu_a, si_a, mu1, mu1_ref, mu_a1, mu_a_ref)

        self.total_steps += 1
        return r, new_internal_state, o1, d, updated_rnn_state_actor, updated_rnn_state_actor_ref, updated_rnn_state_critic, updated_rnn_state_critic_ref

    def _step_loop_multivariate_reduced_logs(self, o, internal_state, a, rnn_state_actor, rnn_state_actor_ref,
                                                 rnn_state_critic,
                                                 rnn_state_critic_ref):
        sa = np.zeros((1, 128))  # Placeholder for the state advantage stream.
        a = [a[0] / self.environment_params['max_impulse'],
             a[1] / self.environment_params['max_angle_change']]  # Set impulse to scale to be inputted to network

        impulse, angle, updated_rnn_state_actor, updated_rnn_state_actor_ref, action_probability = self.sess.run(
            [self.actor_network.impulse_output, self.actor_network.angle_output, self.actor_network.rnn_state_shared,
             self.actor_network.rnn_state_ref,
             self.actor_network.log_prob,
             ],
            feed_dict={self.actor_network.observation: o,
                       self.actor_network.internal_state: internal_state,
                       self.actor_network.prev_actions: np.reshape(a, (1, 2)),
                       self.actor_network.sigma_impulse_combined_proto: self.impulse_sigma,
                       self.actor_network.sigma_angle_combined_proto: self.angle_sigma,
                       self.actor_network.rnn_state_in: rnn_state_actor,
                       self.actor_network.rnn_state_in_ref: rnn_state_actor_ref,
                       self.actor_network.batch_size: 1,
                       self.actor_network.train_length: 1,
                       }
        )

        V, updated_rnn_state_critic, updated_rnn_state_critic_ref = self.sess.run(
            [self.critic_network.Value_output, self.critic_network.rnn_state_shared,
             self.critic_network.rnn_state_ref],
            feed_dict={self.critic_network.observation: o,
                       self.critic_network.internal_state: internal_state,
                       self.critic_network.prev_actions: np.reshape(a, (1, 2)),
                       self.critic_network.rnn_state_in: rnn_state_critic,
                       self.critic_network.rnn_state_in_ref: rnn_state_critic_ref,
                       self.critic_network.batch_size: 1,
                       self.critic_network.train_length: 1,
                       }
        )

        action = [impulse[0][0], angle[0][0]]

        # Simulation step
        o1, r, new_internal_state, d, self.frame_buffer = self.simulation.simulation_step(
            action=action,
            frame_buffer=self.frame_buffer,
            save_frames=self.save_frames,
            activations=sa)

        # Update buffer
        self.buffer.add_training(observation=o,
                                 internal_state=internal_state,
                                 action=action,
                                 reward=r,
                                 value=V,
                                 l_p_action=action_probability,
                                 actor_rnn_state=rnn_state_actor,
                                 actor_rnn_state_ref=rnn_state_actor_ref,
                                 critic_rnn_state=rnn_state_critic,
                                 critic_rnn_state_ref=rnn_state_critic_ref,
                                 )
        self.total_steps += 1
        return r, new_internal_state, o1, d, updated_rnn_state_actor, updated_rnn_state_actor_ref, updated_rnn_state_critic, updated_rnn_state_critic_ref

    def _step_loop_full_logs(self, o, internal_state, a, rnn_state_actor, rnn_state_actor_ref,
                                      rnn_state_critic,
                                      rnn_state_critic_ref):
        # Generate actions and corresponding steps.
        sa = np.zeros((1, 128))  # Placeholder for the state advantage stream.
        a = [a[0] / self.environment_params['max_impulse'],
             a[1] / self.environment_params['max_angle_change']]  # Set impulse to scale to be inputted to network
        impulse, angle, updated_rnn_state_actor, updated_rnn_state_actor_ref, impulse_probability, angle_probability, \
        mu_i, si_i, mu_a, si_a, mu1, mu1_ref, mu_a1, mu_a_ref = self.sess.run(
            [self.actor_network.impulse_output, self.actor_network.angle_output, self.actor_network.rnn_state_shared,
             self.actor_network.rnn_state_ref,
             self.actor_network.log_prob_impulse, self.actor_network.log_prob_angle,
             self.actor_network.mu_impulse_combined, self.actor_network.sigma_impulse_combined,
             self.actor_network.mu_angle_combined,
             self.actor_network.sigma_angle_combined, self.actor_network.mu_impulse, self.actor_network.mu_impulse_ref,
             self.actor_network.mu_angle, self.actor_network.mu_angle_ref
             ],
            feed_dict={self.actor_network.observation: o,
                       self.actor_network.internal_state: internal_state,
                       self.actor_network.prev_actions: np.reshape(a, (1, 2)),
                       self.actor_network.rnn_state_in: rnn_state_actor,
                       self.actor_network.rnn_state_in_ref: rnn_state_actor_ref,
                       self.actor_network.sigma_impulse_combined: self.impulse_sigma,
                       self.actor_network.sigma_angle_combined: self.angle_sigma,
                       self.actor_network.batch_size: 1,
                       self.actor_network.train_length: 1,
                       }
        )

        V, updated_rnn_state_critic, updated_rnn_state_critic_ref = self.sess.run(
            [self.critic_network.Value_output, self.critic_network.rnn_state_shared,
             self.critic_network.rnn_state_ref],
            feed_dict={self.critic_network.observation: o,
                       self.critic_network.internal_state: internal_state,
                       self.critic_network.prev_actions: np.reshape(a, (1, 2)),
                       self.critic_network.rnn_state_in: rnn_state_critic,
                       self.critic_network.rnn_state_in_ref: rnn_state_critic_ref,
                       self.critic_network.batch_size: 1,
                       self.critic_network.train_length: 1,
                       }
        )
        if self.learning_params["beta_distribution"]:
            impulse = [[impulse[0][0]]]
            if impulse[0][0] == 0.0:
                impulse = [[0.001]]

        action = [impulse[0][0], angle[0][0]]

        # Simulation step
        o1, r, new_internal_state, d, self.frame_buffer = self.simulation.simulation_step(
            action=action,
            frame_buffer=self.frame_buffer,
            save_frames=self.save_frames,
            activations=sa)

        # Update buffer
        self.buffer.add_training(observation=o,
                                 internal_state=internal_state,
                                 action=action,
                                 reward=r,
                                 value=V,
                                 l_p_impulse=impulse_probability,
                                 l_p_angle=angle_probability,
                                 actor_rnn_state=rnn_state_actor,
                                 actor_rnn_state_ref=rnn_state_actor_ref,
                                 critic_rnn_state=rnn_state_critic,
                                 critic_rnn_state_ref=rnn_state_critic_ref,
                                 )
        self.buffer.add_logging(mu_i, si_i, mu_a, si_a, mu1, mu1_ref, mu_a1, mu_a_ref)

        self.total_steps += 1
        return r, new_internal_state, o1, d, updated_rnn_state_actor, updated_rnn_state_actor_ref, updated_rnn_state_critic, updated_rnn_state_critic_ref

    def _step_loop_reduced_logs(self, o, internal_state, a, rnn_state_actor, rnn_state_actor_ref,
                                         rnn_state_critic,
                                         rnn_state_critic_ref):
        # Generate actions and corresponding steps.
        sa = np.zeros((1, 128))  # Placeholder for the state advantage stream.
        a = [a[0] / self.environment_params['max_impulse'],
             a[1] / self.environment_params['max_angle_change']]  # Set impulse to scale to be inputted to network

        impulse, angle, updated_rnn_state_actor, updated_rnn_state_actor_ref, impulse_probability, angle_probability = self.sess.run(
            [self.actor_network.impulse_output, self.actor_network.angle_output, self.actor_network.rnn_state_shared,
             self.actor_network.rnn_state_ref,
             self.actor_network.log_prob_impulse, self.actor_network.log_prob_angle,
             ],
            feed_dict={self.actor_network.observation: o,
                       self.actor_network.internal_state: internal_state,
                       self.actor_network.prev_actions: np.reshape(a, (1, 2)),
                       self.actor_network.rnn_state_in: rnn_state_actor,
                       self.actor_network.rnn_state_in_ref: rnn_state_actor_ref,
                       self.actor_network.sigma_impulse_combined: self.impulse_sigma,
                       self.actor_network.sigma_angle_combined: self.angle_sigma,
                       self.actor_network.batch_size: 1,
                       self.actor_network.train_length: 1,
                       }
        )

        V, updated_rnn_state_critic, updated_rnn_state_critic_ref = self.sess.run(
            [self.critic_network.Value_output, self.critic_network.rnn_state_shared,
             self.critic_network.rnn_state_ref],
            feed_dict={self.critic_network.observation: o,
                       self.critic_network.internal_state: internal_state,
                       self.critic_network.prev_actions: np.reshape(a, (1, 2)),
                       self.critic_network.rnn_state_in: rnn_state_critic,
                       self.critic_network.rnn_state_in_ref: rnn_state_critic_ref,
                       self.critic_network.batch_size: 1,
                       self.critic_network.train_length: 1,
                       }
        )

        action = [impulse[0][0], angle[0][0]]

        # Simulation step
        o1, r, new_internal_state, d, self.frame_buffer = self.simulation.simulation_step(
            action=action,
            frame_buffer=self.frame_buffer,
            save_frames=self.save_frames,
            activations=sa)

        # Update buffer
        self.buffer.add_training(observation=o,
                                 internal_state=internal_state,
                                 action=action,
                                 reward=r,
                                 value=V,
                                 l_p_impulse=impulse_probability,
                                 l_p_angle=angle_probability,
                                 actor_rnn_state=rnn_state_actor,
                                 actor_rnn_state_ref=rnn_state_actor_ref,
                                 critic_rnn_state=rnn_state_critic,
                                 critic_rnn_state_ref=rnn_state_critic_ref,
                                 )
        self.total_steps += 1
        return r, new_internal_state, o1, d, updated_rnn_state_actor, updated_rnn_state_actor_ref, updated_rnn_state_critic, updated_rnn_state_critic_ref

    def get_batch_multivariate(self, batch, observation_buffer, internal_state_buffer, action_buffer,
                               previous_action_buffer,
                               log_action_probability_buffer, advantage_buffer, return_buffer):

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

        current_batch_size = observation_batch.shape[0]

        # Stacking for correct network dimensions
        observation_batch = np.vstack(np.vstack(observation_batch))
        internal_state_batch = np.vstack(np.vstack(internal_state_batch))
        action_batch = np.reshape(action_batch[:, :, :],
                                  (self.learning_params["trace_length"] * current_batch_size, 2))
        previous_action_batch = np.vstack(np.vstack(previous_action_batch))
        log_action_probability_batch = log_action_probability_batch.flatten()
        advantage_batch = np.vstack(advantage_batch).flatten()
        return_batch = np.vstack(np.vstack(return_batch)).flatten()

        return observation_batch, internal_state_batch, action_batch, previous_action_batch, \
               log_action_probability_batch, advantage_batch, return_batch, \
               current_batch_size

    def get_batch_multivariate2(self, batch, observation_buffer, internal_state_buffer, action_buffer,
                               previous_action_buffer,
                               log_action_probability_buffer, advantage_buffer, return_buffer, value_buffer, target_outputs_buffer=None):

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
                          batch * self.learning_params["batch_size"]: (batch + 1) * self.learning_params["batch_size"]]

        current_batch_size = observation_batch.shape[0]

        # Stacking for correct network dimensions
        observation_batch = np.vstack(np.vstack(observation_batch))
        internal_state_batch = np.vstack(np.vstack(internal_state_batch))
        action_batch = np.reshape(action_batch[:, :, :],
                                  (self.learning_params["trace_length"] * current_batch_size, 2))
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

    def get_batch(self, batch, observation_buffer, internal_state_buffer, action_buffer, previous_action_buffer,
                  log_impulse_probability_buffer, log_angle_probability_buffer, advantage_buffer, return_buffer):

        observation_batch = observation_buffer[
                            batch * self.batch_size: (batch + 1) * self.batch_size]
        internal_state_batch = internal_state_buffer[
                               batch * self.batch_size: (batch + 1) * self.batch_size]
        action_batch = action_buffer[
                       batch * self.batch_size: (batch + 1) * self.batch_size]
        previous_action_batch = previous_action_buffer[
                                batch * self.batch_size: (batch + 1) * self.batch_size]
        log_impulse_probability_batch = log_impulse_probability_buffer[
                                        batch * self.learning_params["batch_size"]: (batch + 1) * self.learning_params[
                                            "batch_size"]]
        log_angle_probability_batch = log_angle_probability_buffer[
                                      batch * self.learning_params["batch_size"]: (batch + 1) * self.learning_params[
                                          "batch_size"]]
        advantage_batch = advantage_buffer[
                          batch * self.learning_params["batch_size"]: (batch + 1) * self.learning_params["batch_size"]]
        return_batch = return_buffer[
                       batch * self.learning_params["batch_size"]: (batch + 1) * self.learning_params["batch_size"]]

        current_batch_size = observation_batch.shape[0]

        # Stacking for correct network dimensions
        observation_batch = np.vstack(np.vstack(observation_batch))
        internal_state_batch = np.vstack(np.vstack(internal_state_batch))
        impulse_batch = np.reshape(action_batch[:, :, 0],
                                   (self.learning_params["trace_length"] * current_batch_size, 1))
        angle_batch = np.reshape(action_batch[:, :, 1], (self.learning_params["trace_length"] * current_batch_size, 1))
        previous_action_batch = np.vstack(np.vstack(previous_action_batch))
        log_impulse_probability_batch = log_impulse_probability_batch.flatten()
        log_angle_probability_batch = log_angle_probability_batch.flatten()
        advantage_batch = np.vstack(advantage_batch).flatten()
        return_batch = np.vstack(np.vstack(return_batch)).flatten()

        return observation_batch, internal_state_batch, impulse_batch, angle_batch, previous_action_batch, \
               log_impulse_probability_batch, log_angle_probability_batch, advantage_batch, return_batch, \
               current_batch_size

    def train_network(self):
        observation_buffer, internal_state_buffer, action_buffer, previous_action_buffer, \
        log_impulse_probability_buffer, log_angle_probability_buffer, advantage_buffer, return_buffer, \
        key_rnn_points = self.buffer.get_episode_buffer()

        number_of_batches = int(math.ceil(observation_buffer.shape[0] / self.learning_params["batch_size"]))

        for batch in range(number_of_batches):
            # Find steps at start of each trace to compute RNN states
            batch_key_points = [i for i in key_rnn_points if
                                batch * self.learning_params["batch_size"] * self.learning_params[
                                    "trace_length"] <= i < (batch + 1) *
                                self.learning_params["batch_size"] * self.learning_params["trace_length"]]

            # Get the current batch
            observation_batch, internal_state_batch, impulse_batch, angle_batch, previous_action_batch, \
            log_impulse_probability_batch, log_angle_probability_batch, advantage_batch, \
            return_batch, current_batch_size = self.get_batch(batch, observation_buffer, internal_state_buffer,
                                                              action_buffer, previous_action_buffer,
                                                              log_impulse_probability_buffer,
                                                              log_angle_probability_buffer, advantage_buffer,
                                                              return_buffer)

            if self.learning_params['beta_distribution']:
                log_impulse_probability_batch = np.exp(log_impulse_probability_batch)
                maxli = max(log_impulse_probability_batch) + 1
                log_impulse_probability_batch = log_impulse_probability_batch/maxli
                log_impulse_probability_batch = np.log(log_impulse_probability_batch)

            # Loss value logging
            average_loss_value = 0
            average_loss_impulse = 0
            average_loss_angle = 0

            for i in range(self.learning_params["n_updates_per_iteration"]):
                # Compute RNN states for start of each trace.
                actor_rnn_state_slice, actor_rnn_state_ref_slice, critic_rnn_state_slice, \
                critic_rnn_state_ref_slice = self.compute_rnn_states(batch_key_points,
                                                                     observation_buffer[
                                                                     :(batch + 1) * self.learning_params["batch_size"]],
                                                                     internal_state_buffer[
                                                                     :(batch + 1) * self.learning_params["batch_size"]],
                                                                     previous_action_buffer[
                                                                     :(batch + 1) * self.learning_params[
                                                                         "batch_size"]])



                # Optimise critic
                loss_critic_val, _ = self.sess.run(
                    [self.critic_network.critic_loss, self.critic_network.optimizer],
                    feed_dict={self.critic_network.observation: observation_batch,
                               self.critic_network.prev_actions: previous_action_batch,
                               self.critic_network.internal_state: internal_state_batch,
                               self.critic_network.rnn_state_in: critic_rnn_state_slice,
                               self.critic_network.rnn_state_in_ref: critic_rnn_state_ref_slice,

                               self.critic_network.returns_placeholder: return_batch,

                               self.critic_network.train_length: self.learning_params["trace_length"],
                               self.critic_network.batch_size: current_batch_size,
                               self.critic_network.learning_rate: self.learning_params[
                                                                      "learning_rate_critic"] * current_batch_size

                               })

                # Optimise actor
                loss_actor_val_impulse, loss_actor_val_angle, _ = self.sess.run(
                    [self.actor_network.impulse_loss, self.actor_network.angle_loss,
                     self.actor_network.optimizer],
                    feed_dict={self.actor_network.observation: observation_batch,
                               self.actor_network.prev_actions: previous_action_batch,
                               self.actor_network.internal_state: internal_state_batch,
                               self.actor_network.rnn_state_in: actor_rnn_state_slice,
                               self.actor_network.rnn_state_in_ref: actor_rnn_state_ref_slice,
                               self.actor_network.sigma_impulse_combined: self.impulse_sigma,
                               self.actor_network.sigma_angle_combined: self.angle_sigma,

                               self.actor_network.impulse_placeholder: impulse_batch,
                               self.actor_network.angle_placeholder: angle_batch,
                               self.actor_network.old_log_prob_impulse_placeholder: log_impulse_probability_batch,
                               self.actor_network.old_log_prob_angle_placeholder: log_angle_probability_batch,
                               self.actor_network.scaled_advantage_placeholder: advantage_batch,

                               self.actor_network.train_length: self.learning_params["trace_length"],
                               self.actor_network.batch_size: current_batch_size,
                               self.actor_network.learning_rate: self.learning_params[
                                                                     "learning_rate_actor"] * current_batch_size
                               })
                if np.isnan(loss_actor_val_impulse):
                    x = True
                average_loss_impulse += np.mean(np.abs(loss_actor_val_impulse))
                average_loss_angle += np.mean(np.abs(loss_actor_val_angle))
                average_loss_value += np.abs(loss_critic_val)

            self.buffer.add_loss(average_loss_impulse / self.learning_params["n_updates_per_iteration"],
                                 average_loss_angle / self.learning_params["n_updates_per_iteration"],
                                 average_loss_value / self.learning_params["n_updates_per_iteration"])

    def train_network_multivariate2(self):
        if self.use_rnd:
            observation_buffer, internal_state_buffer, action_buffer, previous_action_buffer, \
                log_action_probability_buffer, advantage_buffer, return_buffer, value_buffer, target_outputs_buffer, \
                key_rnn_points = self.buffer.get_episode_buffer()
        else:
            observation_buffer, internal_state_buffer, action_buffer, previous_action_buffer, \
                log_action_probability_buffer, advantage_buffer, return_buffer, value_buffer, \
                key_rnn_points = self.buffer.get_episode_buffer()

        number_of_batches = int(math.ceil(observation_buffer.shape[0] / self.learning_params["batch_size"]))

        for batch in range(number_of_batches):
            # Find steps at start of each trace to compute RNN states
            batch_key_points = [i for i in key_rnn_points if
                                batch * self.learning_params["batch_size"] * self.learning_params[
                                    "trace_length"] <= i < (batch + 1) *
                                self.learning_params["batch_size"] * self.learning_params["trace_length"]]

            # Get the current batch
            if self.use_rnd:
                observation_batch, internal_state_batch, action_batch, previous_action_batch, \
            log_action_probability_batch, advantage_batch, \
            return_batch, previous_value_batch, target_outputs_batch, current_batch_size = self.get_batch_multivariate2(batch, observation_buffer,
                                                                           internal_state_buffer,
                                                                           action_buffer, previous_action_buffer,
                                                                           log_action_probability_buffer,
                                                                           advantage_buffer,
                                                                           return_buffer, value_buffer, target_outputs_buffer)
            else:
                observation_batch, internal_state_batch, action_batch, previous_action_batch, \
                log_action_probability_batch, advantage_batch, \
                return_batch, previous_value_batch, current_batch_size = self.get_batch_multivariate2(
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

                # Optimise actor
                loss_actor_val, loss_critic_val, total_loss, _, log_action_probability_batch_new, scaled_actions = self.sess.run(
                    [self.actor_network.policy_loss, self.actor_network.value_loss, self.actor_network.total_loss,
                     self.actor_network.train, self.actor_network.new_neg_log_prob, self.actor_network.normalised_action],

                    feed_dict={self.actor_network.observation: observation_batch,
                               self.actor_network.prev_actions: previous_action_batch,
                               self.actor_network.internal_state: internal_state_batch,
                               self.actor_network.rnn_state_in: actor_rnn_state_slice,
                               self.actor_network.rnn_state_in_ref: actor_rnn_state_ref_slice,

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
                                                                     "learning_rate_actor"] * current_batch_size
                               })

                average_loss_impulse += np.mean(np.abs(loss_actor_val))
                average_loss_angle += np.mean(np.abs(loss_actor_val))
                average_loss_value += np.abs(loss_critic_val)

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

            # print("RATIO " + str(np.mean(ratio)))
            # print("ADVANTAGE: " + str(np.mean(advantage_batch)))

            self.buffer.add_loss(average_loss_impulse / self.learning_params["n_updates_per_iteration"],
                                 average_loss_angle / self.learning_params["n_updates_per_iteration"],
                                 average_loss_value / self.learning_params["n_updates_per_iteration"])

    def train_network_multivariate(self):
        observation_buffer, internal_state_buffer, action_buffer, previous_action_buffer, \
        log_action_probability_buffer, advantage_buffer, return_buffer, \
        key_rnn_points = self.buffer.get_episode_buffer()

        number_of_batches = int(math.ceil(observation_buffer.shape[0] / self.learning_params["batch_size"]))

        for batch in range(number_of_batches):
            # Find steps at start of each trace to compute RNN states
            batch_key_points = [i for i in key_rnn_points if
                                batch * self.learning_params["batch_size"] * self.learning_params[
                                    "trace_length"] <= i < (batch + 1) *
                                self.learning_params["batch_size"] * self.learning_params["trace_length"]]

            # Get the current batch
            observation_batch, internal_state_batch, action_batch, previous_action_batch, \
            log_action_probability_batch, advantage_batch, \
            return_batch, current_batch_size = self.get_batch_multivariate(batch, observation_buffer,
                                                                           internal_state_buffer,
                                                                           action_buffer, previous_action_buffer,
                                                                           log_action_probability_buffer,
                                                                           advantage_buffer,
                                                                           return_buffer)

            # Loss value logging
            average_loss_value = 0
            average_loss_impulse = 0
            average_loss_angle = 0


            for i in range(self.learning_params["n_updates_per_iteration"]):
                # Compute RNN states for start of each trace.
                actor_rnn_state_slice, actor_rnn_state_ref_slice, critic_rnn_state_slice, \
                critic_rnn_state_ref_slice = self.compute_rnn_states(batch_key_points,
                                                                     observation_buffer[
                                                                     :(batch + 1) * self.learning_params["batch_size"]],
                                                                     internal_state_buffer[
                                                                     :(batch + 1) * self.learning_params["batch_size"]],
                                                                     previous_action_buffer[
                                                                     :(batch + 1) * self.learning_params[
                                                                         "batch_size"]])

                # Optimise critic
                loss_critic_val, _ = self.sess.run(
                    [self.critic_network.critic_loss, self.critic_network.optimizer],
                    feed_dict={self.critic_network.observation: observation_batch,
                               self.critic_network.prev_actions: previous_action_batch,
                               self.critic_network.internal_state: internal_state_batch,
                               self.critic_network.rnn_state_in: critic_rnn_state_slice,
                               self.critic_network.rnn_state_in_ref: critic_rnn_state_ref_slice,

                               self.critic_network.returns_placeholder: return_batch,

                               self.critic_network.train_length: self.learning_params["trace_length"],
                               self.critic_network.batch_size: current_batch_size,
                               self.critic_network.learning_rate: self.learning_params[
                                                                      "learning_rate_critic"] * current_batch_size

                               })

                # Optimise actor
                loss_actor_val, _ = self.sess.run(
                    [self.actor_network.total_loss,
                     self.actor_network.optimizer],
                    feed_dict={self.actor_network.observation: observation_batch,
                               self.actor_network.prev_actions: previous_action_batch,
                               self.actor_network.internal_state: internal_state_batch,
                               self.actor_network.rnn_state_in: actor_rnn_state_slice,
                               self.actor_network.rnn_state_in_ref: actor_rnn_state_ref_slice,

                               self.actor_network.sigma_impulse_combined_proto: self.impulse_sigma,
                               self.actor_network.sigma_angle_combined_proto: self.angle_sigma,

                               self.actor_network.action_placeholder: action_batch,
                               self.actor_network.old_log_prob: log_action_probability_batch,
                               self.actor_network.scaled_advantage_placeholder: advantage_batch,

                               self.actor_network.train_length: self.learning_params["trace_length"],
                               self.actor_network.batch_size: current_batch_size,
                               self.actor_network.learning_rate: self.learning_params[
                                                                     "learning_rate_actor"] * current_batch_size
                               })

                average_loss_impulse += np.mean(np.abs(loss_actor_val))
                average_loss_angle += np.mean(np.abs(loss_actor_val))
                average_loss_value += np.abs(loss_critic_val)

            self.buffer.add_loss(average_loss_impulse / self.learning_params["n_updates_per_iteration"],
                                 average_loss_angle / self.learning_params["n_updates_per_iteration"],
                                 average_loss_value / self.learning_params["n_updates_per_iteration"])

    def old_training(self):
        number_of_batches = 5
        for batch in range(number_of_batches):
            if batch == number_of_batches - 1:
                final_batch = True
                current_batch_size = len(self.buffer.return_buffer) - self.buffer.pointer

            else:
                final_batch = False
                current_batch_size = self.learning_params["batch_size"]
            observation_slice, internal_state_slice, action_slice, previous_action_slice, reward_slice, value_slice, \
            log_impulse_probability_slice, log_angle_probability_slice, advantage_slice, return_slice, \
            actor_rnn_state_slice, actor_rnn_state_ref_slice, critic_rnn_state_slice, critic_rnn_state_ref_slice = self.buffer.get_batch(
                final_batch)

            # Moveed to buffer
            # actor_rnn_state_slice = np.moveaxis(actor_rnn_state_slice, 1, 0).squeeze()[:, 0:1, :]
            # actor_rnn_state_ref_slice = np.moveaxis(actor_rnn_state_ref_slice, 1, 0).squeeze()[:, 0:1, :]
            # critic_rnn_state_slice = np.moveaxis(critic_rnn_state_slice, 1, 0).squeeze()[:, 0:1, :]
            # critic_rnn_state_ref_slice = np.moveaxis(critic_rnn_state_ref_slice, 1, 0).squeeze()[:, 0:1, :]
            #
            # actor_rnn_state_slice = (actor_rnn_state_slice[0], actor_rnn_state_slice[1])
            # actor_rnn_state_ref_slice = (actor_rnn_state_ref_slice[0], actor_rnn_state_ref_slice[1])
            # critic_rnn_state_slice = (critic_rnn_state_slice[0], critic_rnn_state_slice[1])
            # critic_rnn_state_ref_slice = (critic_rnn_state_ref_slice[0], critic_rnn_state_ref_slice[1])

            average_loss_value = 0
            average_loss_impulse = 0
            average_loss_angle = 0
            for i in range(self.learning_params["n_updates_per_iteration"]):
                # Dont Recompute initial state
                loss_critic_val, _ = self.sess.run(
                    [self.critic_network.critic_loss, self.critic_network.optimizer],
                    feed_dict={self.critic_network.observation: np.vstack(observation_slice),
                               # self.critic_network.scaler: np.full(np.vstack(observation_slice).shape, 255),
                               self.critic_network.prev_actions: np.vstack(previous_action_slice),
                               self.critic_network.internal_state: np.vstack(internal_state_slice),
                               self.critic_network.rnn_state_in: critic_rnn_state_slice,
                               self.critic_network.rnn_state_in_ref: critic_rnn_state_ref_slice,

                               self.critic_network.returns_placeholder: np.vstack(return_slice).flatten(),

                               self.critic_network.train_length: current_batch_size,
                               self.critic_network.batch_size: 1,
                               })

                loss_actor_val_impulse, loss_actor_val_angle, _ = self.sess.run(
                    [self.actor_network.impulse_loss, self.actor_network.angle_loss,
                     self.actor_network.optimizer],
                    feed_dict={self.actor_network.observation: np.vstack(observation_slice),
                               # self.actor_network.scaler: np.full(np.vstack(observation_slice).shape, 255),
                               self.actor_network.prev_actions: np.vstack(previous_action_slice),
                               self.actor_network.internal_state: np.vstack(internal_state_slice),
                               self.actor_network.rnn_state_in: actor_rnn_state_slice,
                               self.actor_network.rnn_state_in_ref: actor_rnn_state_ref_slice,
                               self.actor_network.sigma_impulse_combined: self.impulse_sigma,
                               self.actor_network.sigma_angle_combined: self.angle_sigma,

                               self.actor_network.impulse_placeholder: np.vstack(action_slice[:, 0]),
                               self.actor_network.angle_placeholder: np.vstack(action_slice[:, 1]),
                               self.actor_network.old_log_prob_impulse_placeholder: log_impulse_probability_slice.flatten(),
                               self.actor_network.old_log_prob_angle_placeholder: log_angle_probability_slice.flatten(),
                               self.actor_network.scaled_advantage_placeholder: np.vstack(advantage_slice).flatten(),

                               self.actor_network.train_length: current_batch_size,
                               self.actor_network.batch_size: 1,
                               })

                average_loss_impulse += np.mean(np.abs(loss_actor_val_impulse))
                average_loss_angle += np.mean(np.abs(loss_actor_val_angle))
                average_loss_value += np.abs(loss_critic_val)

            self.buffer.add_loss(average_loss_impulse / self.learning_params["n_updates_per_iteration"],
                                 average_loss_angle / self.learning_params["n_updates_per_iteration"],
                                 average_loss_value / self.learning_params["n_updates_per_iteration"])

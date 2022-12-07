import copy
import math
import numpy as np

import tensorflow.compat.v1 as tf

from Networks.PPO.proximal_policy_optimizer_discrete import PPONetworkActor
from Networks.PPO.proximal_policy_optimizer_discrete_sb_emulator import PPONetworkActorDiscreteEmulator
from Services.PPO.base_ppo import BasePPO

tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)


class DiscretePPO(BasePPO):

    def __init__(self):
        super().__init__()
        self.continuous = False

        # Placeholders
        self.epsilon_greedy = None
        self.e = None
        self.step_drop = None

        self.output_dimensions = 1

    def create_network(self):
        """
        Create the main and target Q networks, according to the configuration parameters.
        :return: The main network and the target network graphs.
        """
        actor_cell, internal_states, internal_state_names = BasePPO.create_network(self)

        if self.sb_emulator:
            self.actor_network = PPONetworkActorDiscreteEmulator(simulation=self.simulation,
                                                                 rnn_dim=self.learning_params['rnn_dim_shared'],
                                                                 rnn_cell=actor_cell,
                                                                 my_scope='actor',
                                                                 internal_states=internal_states,
                                                                 clip_param=self.environment_params['clip_param'],
                                                                 num_actions=self.learning_params['num_actions'],
                                                                 epsilon_greedy=self.epsilon_greedy,
                                                                 new_simulation=self.new_simulation,

                                                                 )
        else:
            self.actor_network = PPONetworkActor(simulation=self.simulation,
                                                 rnn_dim=self.learning_params['rnn_dim_shared'],
                                                 rnn_cell=actor_cell,
                                                 my_scope='actor',
                                                 internal_states=internal_states,
                                                 clip_param=self.environment_params['clip_param'],
                                                 num_actions=self.learning_params['num_actions'],
                                                 epsilon_greedy=self.epsilon_greedy,
                                                 new_simulation=self.new_simulation,
                                                 )

    def _episode_loop(self, a=None):
        a = 0
        super(DiscretePPO, self)._episode_loop(a)

    def _assay_step_loop(self, o, internal_state, a, rnn_state_actor, rnn_state_actor_ref, rnn_state_critic,
                         rnn_state_critic_ref):
        sa = np.zeros((1, 128))  # Placeholder for the state advantage stream.

        action, updated_rnn_state_actor, updated_rnn_state_actor_ref, conv1l_actor, conv2l_actor, conv3l_actor, \
        conv4l_actor, conv1r_actor, conv2r_actor, conv3r_actor, conv4r_actor = self.sess.run(
            [self.actor_network.action_output,
             self.actor_network.rnn_state_shared,
             self.actor_network.rnn_state_ref,
             self.actor_network.conv1l, self.actor_network.conv2l, self.actor_network.conv3l,
             self.actor_network.conv4l,
             self.actor_network.conv1r, self.actor_network.conv2r, self.actor_network.conv3r,
             self.actor_network.conv4r,
             ],
            feed_dict={self.actor_network.observation: o,
                       self.actor_network.internal_state: internal_state,
                       self.actor_network.prev_actions: np.reshape(a, (1, 1)),
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
                       self.critic_network.prev_actions: np.reshape(a, (1, 1)),
                       self.critic_network.rnn_state_in: rnn_state_critic,
                       self.critic_network.rnn_state_in_ref: rnn_state_critic_ref,
                       self.critic_network.batch_size: 1,
                       self.critic_network.train_length: 1,
                       }
        )
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
                                 actor_rnn_state=rnn_state_actor,
                                 actor_rnn_state_ref=rnn_state_actor_ref,
                                 critic_rnn_state=rnn_state_critic,
                                 critic_rnn_state_ref=rnn_state_critic_ref,
                                 )

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

    def _training_step_loop_reduced_logs(self, o, internal_state, a, rnn_state_actor, rnn_state_actor_ref,
                                         rnn_state_critic,
                                         rnn_state_critic_ref):
        return self._training_step_loop(o, internal_state, a, rnn_state_actor, rnn_state_actor_ref, rnn_state_critic,
                                        rnn_state_critic_ref)

    def _training_step_loop_full_logs(self, o, internal_state, a, rnn_state_actor, rnn_state_actor_ref,
                                      rnn_state_critic,
                                      rnn_state_critic_ref):
        return self._training_step_loop(o, internal_state, a, rnn_state_actor, rnn_state_actor_ref, rnn_state_critic,
                                        rnn_state_critic_ref)

    def _training_step_loop_reduced_logs2(self, o, internal_state, a, rnn_state_actor, rnn_state_actor_ref,
                                          rnn_state_critic,
                                          rnn_state_critic_ref):
        return self._training_step_loop2(o, internal_state, a, rnn_state_actor, rnn_state_actor_ref, rnn_state_critic,
                                         rnn_state_critic_ref)

    def _training_step_loop_full_logs2(self, o, internal_state, a, rnn_state_actor, rnn_state_actor_ref,
                                       rnn_state_critic,
                                       rnn_state_critic_ref):
        return self._training_step_loop2(o, internal_state, a, rnn_state_actor, rnn_state_actor_ref, rnn_state_critic,
                                         rnn_state_critic_ref)

    def _training_step_loop2(self, o, internal_state, a, rnn_state_actor, rnn_state_actor_ref, rnn_state_critic,
                             rnn_state_critic_ref):
        sa = np.zeros((1, 128))  # Placeholder for the state advantage stream.

        if self.epsilon_greedy:

            action, updated_rnn_state_actor, updated_rnn_state_actor_ref, probability, V = self.sess.run(
                [self.actor_network.action_output, self.actor_network.rnn_state_shared,
                 self.actor_network.rnn_state_ref, self.actor_network.neg_log_prob,
                 self.actor_network.value_output
                 ],
                feed_dict={self.actor_network.observation: o,
                           self.actor_network.internal_state: internal_state,
                           self.actor_network.prev_actions: np.reshape(a, (1, 1)),
                           self.actor_network.rnn_state_in: rnn_state_actor,
                           self.actor_network.rnn_state_in_ref: rnn_state_actor_ref,
                           self.actor_network.batch_size: 1,
                           self.actor_network.train_length: 1,
                           self.actor_network.entropy_coefficient: self.learning_params["lambda_entropy"],
                           }
            )
            if np.random.rand(1) < self.e:
                action = np.random.randint(0, self.learning_params['num_actions'])

            if self.e > self.learning_params['endE']:
                self.e -= self.step_drop
        else:
            action, updated_rnn_state_actor, updated_rnn_state_actor_ref, probability, V = self.sess.run(
                [self.actor_network.action_output, self.actor_network.rnn_state_shared,
                 self.actor_network.rnn_state_ref, self.actor_network.neg_log_prob,
                 self.actor_network.value_output
                 ],
                feed_dict={self.actor_network.observation: o,
                           self.actor_network.internal_state: internal_state,
                           self.actor_network.prev_actions: np.reshape(a, (1, 1)),
                           self.actor_network.rnn_state_in: rnn_state_actor,
                           self.actor_network.rnn_state_in_ref: rnn_state_actor_ref,
                           self.actor_network.batch_size: 1,
                           self.actor_network.train_length: 1,
                           self.actor_network.entropy_coefficient: self.learning_params["lambda_entropy"],
                           }
            )

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
                                 l_p_action=probability,
                                 actor_rnn_state=rnn_state_actor,
                                 actor_rnn_state_ref=rnn_state_actor_ref,
                                 critic_rnn_state=rnn_state_critic,
                                 critic_rnn_state_ref=rnn_state_critic_ref,
                                 )
        self.total_steps += 1
        return r, new_internal_state, o1, d, updated_rnn_state_actor, updated_rnn_state_actor_ref, 0, 0, action

    def _training_step_loop(self, o, internal_state, a, rnn_state_actor, rnn_state_actor_ref, rnn_state_critic,
                            rnn_state_critic_ref):
        sa = np.zeros((1, 128))  # Placeholder for the state advantage stream.

        if self.epsilon_greedy:

            action, updated_rnn_state_actor, updated_rnn_state_actor_ref, action_probabilities = self.sess.run(
                [self.actor_network.action_output, self.actor_network.rnn_state_shared,
                 self.actor_network.rnn_state_ref, self.actor_network.action_probabilities
                 ],
                feed_dict={self.actor_network.observation: o,
                           self.actor_network.internal_state: internal_state,
                           self.actor_network.prev_actions: np.reshape(a, (1, 1)),
                           self.actor_network.rnn_state_in: rnn_state_actor,
                           self.actor_network.rnn_state_in_ref: rnn_state_actor_ref,
                           self.actor_network.batch_size: 1,
                           self.actor_network.train_length: 1,
                           }
            )
            if np.random.rand(1) < self.e:
                action = np.random.randint(0, self.learning_params['num_actions'])
            probability = action_probabilities[0][action]

            if self.e > self.learning_params['endE']:
                self.e -= self.step_drop
        else:
            action, updated_rnn_state_actor, updated_rnn_state_actor_ref, probability, V = self.sess.run(
                [self.actor_network.action_output, self.actor_network.rnn_state_shared,
                 self.actor_network.rnn_state_ref, self.actor_network.chosen_action_probability,
                 self.actor_network.value_output
                 ],
                feed_dict={self.actor_network.observation: o,
                           self.actor_network.internal_state: internal_state,
                           self.actor_network.prev_actions: np.reshape(a, (1, 1)),
                           self.actor_network.rnn_state_in: rnn_state_actor,
                           self.actor_network.rnn_state_in_ref: rnn_state_actor_ref,
                           self.actor_network.batch_size: 1,
                           self.actor_network.trainLength: 1,
                           }
            )

        updated_rnn_state_critic, updated_rnn_state_critic_ref, V = self.sess.run(
            [self.critic_network.rnn_state_shared,
             self.critic_network.rnn_state_ref,
             self.actor_network.value_output
             ],
            feed_dict={self.actor_network.observation: o,
                       self.actor_network.internal_state: internal_state,
                       self.actor_network.prev_actions: np.reshape(a, (1, 1)),
                       self.actor_network.rnn_state_in: rnn_state_actor,
                       self.actor_network.rnn_state_in_ref: rnn_state_actor_ref,
                       self.actor_network.batch_size: 1,
                       self.actor_network.trainLength: 1,
                       }
        )

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
                                 l_p_action=probability,
                                 actor_rnn_state=rnn_state_actor,
                                 actor_rnn_state_ref=rnn_state_actor_ref,
                                 critic_rnn_state=rnn_state_critic,
                                 critic_rnn_state_ref=rnn_state_critic_ref,
                                 )
        self.total_steps += 1
        return r, new_internal_state, o1, d, updated_rnn_state_actor, updated_rnn_state_actor_ref, updated_rnn_state_critic, updated_rnn_state_critic_ref

    def compute_rnn_states_old(self, rnn_key_points, observation_buffer, internal_state_buffer, previous_action_buffer):

        observation_buffer = np.vstack(observation_buffer)
        internal_state_buffer = np.vstack(internal_state_buffer)
        previous_action_buffer = np.reshape(previous_action_buffer, (observation_buffer.shape[0], 1))

        rnn_state_actor = copy.copy(self.init_rnn_state_actor)
        rnn_state_actor_ref = copy.copy(self.init_rnn_state_actor_ref)
        rnn_state_critic = copy.copy(self.init_rnn_state_critic)
        rnn_state_critic_ref = copy.copy(self.init_rnn_state_critic_ref)

        actor_rnn_state_buffer = ([rnn_state_actor[0][0]], [rnn_state_actor[1][0]])
        actor_rnn_state_ref_buffer = ([rnn_state_actor_ref[0][0]], [rnn_state_actor_ref[1][0]])
        critic_rnn_state_buffer = ([rnn_state_critic[0][0]], [rnn_state_critic[1][0]])
        critic_rnn_state_ref_buffer = ([rnn_state_critic_ref[0][0]], [rnn_state_critic_ref[1][0]])

        for step in range(max(rnn_key_points)):
            rnn_state_critic, rnn_state_critic_ref = self.sess.run(
                [self.critic_network.rnn_state_shared, self.critic_network.rnn_state_ref],
                feed_dict={self.critic_network.observation: observation_buffer[step],
                           self.critic_network.prev_actions: previous_action_buffer[step].reshape(1, 1),
                           self.critic_network.internal_state: internal_state_buffer[step].reshape(1, 2),

                           self.critic_network.rnn_state_in: rnn_state_critic,
                           self.critic_network.rnn_state_in_ref: rnn_state_critic_ref,

                           self.critic_network.trainLength: 1,
                           self.critic_network.batch_size: 1,
                           })
            rnn_state_actor, rnn_state_actor_ref = self.sess.run(
                [self.actor_network.rnn_state_shared, self.actor_network.rnn_state_ref],
                feed_dict={self.actor_network.observation: observation_buffer[step],
                           self.actor_network.prev_actions: previous_action_buffer[step].reshape(1, 1),
                           self.actor_network.internal_state: internal_state_buffer[step].reshape(1, 2),

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
        actor_rnn_state_ref_buffer = (np.array(actor_rnn_state_ref_buffer[0]), np.array(actor_rnn_state_ref_buffer[1]))
        critic_rnn_state_buffer = (np.array(critic_rnn_state_buffer[0]), np.array(critic_rnn_state_buffer[1]))
        critic_rnn_state_ref_buffer = (
            np.array(critic_rnn_state_ref_buffer[0]), np.array(critic_rnn_state_ref_buffer[1]))

        return actor_rnn_state_buffer, actor_rnn_state_ref_buffer, critic_rnn_state_buffer, critic_rnn_state_ref_buffer

    def get_batch(self, batch, observation_buffer, internal_state_buffer, action_buffer, previous_action_buffer,
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
        action_batch = np.reshape(action_batch, (action_batch.shape[0] * action_batch.shape[1], 1))
        previous_action_batch = np.reshape(previous_action_batch,
                                           (previous_action_batch.shape[0] * previous_action_batch.shape[1], 1))
        log_action_probability_batch = log_action_probability_batch.flatten()
        advantage_batch = np.vstack(advantage_batch).flatten()
        return_batch = np.vstack(np.vstack(return_batch)).flatten()

        return observation_batch, internal_state_batch, action_batch, previous_action_batch, \
               log_action_probability_batch, advantage_batch, return_batch, \
               current_batch_size

    def get_batch2(self, batch, observation_buffer, internal_state_buffer, action_buffer, previous_action_buffer,
                   log_action_probability_buffer, advantage_buffer, value_buffer, return_buffer):
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
        action_batch = np.reshape(action_batch, (action_batch.shape[0] * action_batch.shape[1], 1))
        previous_action_batch = np.reshape(previous_action_batch,
                                           (previous_action_batch.shape[0] * previous_action_batch.shape[1], 1))
        log_action_probability_batch = log_action_probability_batch.flatten()
        advantage_batch = np.vstack(advantage_batch).flatten()
        return_batch = np.vstack(np.vstack(return_batch)).flatten()
        value_batch = value_batch.flatten()

        return observation_batch, internal_state_batch, action_batch, previous_action_batch, \
               log_action_probability_batch, advantage_batch, return_batch, value_batch, \
               current_batch_size

    def train_network(self):
        self.buffer.tidy()

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
            return_batch, current_batch_size = self.get_batch(batch, observation_buffer, internal_state_buffer,
                                                              action_buffer, previous_action_buffer,
                                                              log_action_probability_buffer,
                                                              advantage_buffer,
                                                              return_buffer)

            # Loss value logging
            average_loss_value = 0
            average_loss_actor = 0

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

                               self.critic_network.trainLength: self.learning_params["trace_length"],
                               self.critic_network.batch_size: current_batch_size,
                               self.critic_network.learning_rate: self.learning_params[
                                                                      "learning_rate_critic"] * current_batch_size

                               })

                # Optimise actor
                actor_loss, _ = self.sess.run(
                    [self.actor_network.action_loss, self.actor_network.optimizer],
                    feed_dict={self.actor_network.observation: observation_batch,
                               self.actor_network.prev_actions: previous_action_batch,
                               self.actor_network.internal_state: internal_state_batch,
                               self.actor_network.rnn_state_in: actor_rnn_state_slice,
                               self.actor_network.rnn_state_in_ref: actor_rnn_state_ref_slice,

                               self.actor_network.action_placeholder: action_batch,
                               self.actor_network.old_log_prob_action_placeholder: log_action_probability_batch,
                               self.actor_network.scaled_advantage_placeholder: advantage_batch,

                               self.actor_network.trainLength: self.learning_params["trace_length"],
                               self.actor_network.batch_size: current_batch_size,
                               self.actor_network.learning_rate: self.learning_params[
                                                                     "learning_rate_actor"] * current_batch_size
                               })

                average_loss_actor += np.mean(np.abs(actor_loss))
                average_loss_value += np.abs(loss_critic_val)

            self.buffer.add_loss(average_loss_actor / self.learning_params["n_updates_per_iteration"],
                                 average_loss_value / self.learning_params["n_updates_per_iteration"])

        self.buffer.reset()

    def train_network_2(self):
        self.buffer.tidy()

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
            observation_batch, internal_state_batch, action_batch, previous_action_batch, \
            log_action_probability_batch, advantage_batch, \
            return_batch, previous_value_batch, current_batch_size = self.get_batch2(batch, observation_buffer,
                                                                                     internal_state_buffer,
                                                                                     action_buffer,
                                                                                     previous_action_buffer,
                                                                                     log_action_probability_buffer,
                                                                                     advantage_buffer, value_buffer,
                                                                                     return_buffer)

            # Loss value logging
            average_loss_value = 0
            average_loss_actor = 0

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
                actor_loss, critic_loss, _ = self.sess.run(
                    [self.actor_network.policy_loss, self.actor_network.value_loss, self.actor_network.train],
                    feed_dict={self.actor_network.observation: observation_batch,
                               self.actor_network.prev_actions: previous_action_batch,
                               self.actor_network.internal_state: internal_state_batch,
                               self.actor_network.rnn_state_in: actor_rnn_state_slice,
                               self.actor_network.rnn_state_in_ref: actor_rnn_state_ref_slice,

                               self.actor_network.action_placeholder: action_batch,
                               self.actor_network.old_neg_log_prob: log_action_probability_batch,
                               self.actor_network.scaled_advantage_placeholder: advantage_batch,

                               self.actor_network.returns_placeholder: return_batch,
                               self.actor_network.old_value_placeholder: previous_value_batch,

                               self.actor_network.train_length: self.learning_params["trace_length"],
                               self.actor_network.batch_size: current_batch_size,
                               self.actor_network.learning_rate: self.learning_params[
                                                                     "learning_rate_actor"] * current_batch_size,
                               self.actor_network.entropy_coefficient: self.learning_params["lambda_entropy"],

                               })

                average_loss_actor += np.mean(np.abs(actor_loss))
                average_loss_value += np.abs(critic_loss)

            self.buffer.add_loss(average_loss_actor / self.learning_params["n_updates_per_iteration"],
                                 average_loss_value / self.learning_params["n_updates_per_iteration"])
        self.buffer.reset()
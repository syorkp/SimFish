from time import time
import json
import os

import numpy as np
import tensorflow.compat.v1 as tf

from Environment.continuous_naturalistic_environment import ContinuousNaturalisticEnvironment
from Network.advantage_actor_critic import A2CNetwork
from Tools.make_gif import make_gif

tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)


def a2c_training_target(trial, total_steps, episode_number, memory_fraction):
    services = A2CTrainingService(model_name=trial["Model Name"],
                                  trial_number=trial["Trial Number"],
                                  model_exists=trial["Model Exists"],
                                  tethered=trial["Tethered"],
                                  scaffold_name=trial["Environment Name"],
                                  episode_transitions=trial["Episode Transitions"],
                                  total_configurations=trial["Total Configurations"],
                                  conditional_transitions=trial["Conditional Transitions"],
                                  total_steps=total_steps,
                                  episode_number=episode_number,
                                  monitor_gpu=trial["monitor gpu"],
                                  realistic_bouts=trial["Realistic Bouts"],
                                  memory_fraction=memory_fraction,
                                  using_gpu=trial["Using GPU"]
                                  )
    services.run()


class A2CTrainingService:

    def __init__(self, model_name, trial_number, model_exists, tethered, scaffold_name, episode_transitions,
                 total_configurations, conditional_transitions, total_steps, episode_number, monitor_gpu,
                 realistic_bouts, memory_fraction, using_gpu):
        """
        An instance of TrainingService handles the training of the DQN within a specified environment, according to
        specified parameters.
        :param model_name: The name of the model, usually to match the naming of the env configuration files.
        :param trial_number: The index of the trial, so that agents trained under the same configuration may be
        distinguished in their output files.
        """

        # Names and directories
        self.trial_id = f"{model_name}-{trial_number}"
        self.output_location = f"./Training-Output/{model_name}-{trial_number}"

        # Configurations
        self.scaffold_name = scaffold_name
        self.total_configurations = total_configurations
        self.episode_transitions = episode_transitions
        self.conditional_transitions = conditional_transitions
        self.tethered = tethered
        self.configuration_index = 1
        self.switched_configuration = False
        self.params, self.env = self.load_configuration_files()

        # Basic Parameters
        self.load_model = model_exists
        self.monitor_gpu = monitor_gpu
        self.using_gpu = using_gpu
        self.realistic_bouts = realistic_bouts
        self.memory_fraction = memory_fraction

        # Maintain variables
        if episode_number is not None:
            self.episode_number = episode_number + 1
        else:
            self.episode_number = 0

        if total_steps is not None:
            self.total_steps = total_steps
        else:
            self.total_steps = 0

        # Network and Training Parameters
        self.saver = None
        self.writer = None
        self.a2c_network = None
        self.init = None
        self.trainables = None
        self.target_ops = None
        self.sess = None
        self.step_drop = (self.params['startE'] - self.params['endE']) / self.params['anneling_steps']
        self.pre_train_steps = self.total_steps + self.params["pre_train_steps"]

        # Simulation
        self.simulation = ContinuousNaturalisticEnvironment(self.env, realistic_bouts)
        self.realistic_bouts = realistic_bouts
        self.save_frames = False
        self.switched_configuration = True

        # Data
        self.frame_buffer = []
        self.training_times = []
        self.reward_list = []

        self.last_episodes_prey_caught = []
        self.last_episodes_predators_avoided = []
        self.last_episodes_sand_grains_bumped = []

        self.gamma = 0.99  # Discount factor

        # Training buffers
        # Buffers for batch training
        self.action_buffer = []
        self.observation_buffer = []
        self.reward_buffer = []
        self.internal_state_buffer = []
        self.value_buffer = []


    def load_configuration_files(self):
        """
        Called by create_trials method, should return the learning and environment configurations in JSON format.
        :return:
        """
        print("Loading configuration...")
        configuration_location = f"./Configurations/{self.scaffold_name}/{str(self.configuration_index)}"
        with open(f"{configuration_location}_learning.json", 'r') as f:
            params = json.load(f)
        with open(f"{configuration_location}_env.json", 'r') as f:
            env = json.load(f)
        return params, env

    def run(self):
        """Run the simulation, either loading a checkpoint if there or starting from scratch. If loading, uses the
        previous checkpoint to set the episode number."""

        print("Running simulation")

        if self.using_gpu:
            # options = tf.GPUOptions(per_process_gpu_memory_fraction=self.memory_fraction)
            # config = tf.ConfigProto(gpu_options=options)
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
        else:
            config = None

        if config:
            with tf.Session(config=config) as self.sess:
                self._run()
        else:
            with tf.Session() as self.sess:
                self._run()

    def _run(self):
        self.a2c_network = self.create_network()
        self.saver = tf.train.Saver(max_to_keep=5)
        self.init = tf.global_variables_initializer()
        self.trainables = tf.trainable_variables()
        if self.load_model:
            print(f"Attempting to load model at {self.output_location}")
            checkpoint = tf.train.get_checkpoint_state(self.output_location)
            if hasattr(checkpoint, "model_checkpoint_path"):
                self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
                print("Loading successful")

            else:
                print("No saved checkpoints found, starting from scratch.")
                self.sess.run(self.init)
        else:
            print("First attempt at running model. Starting from scratch.")
            self.sess.run(self.init)
        self.writer = tf.summary.FileWriter(f"{self.output_location}/logs/", tf.get_default_graph())

        for e_number in range(self.episode_number, self.params["num_episodes"]):
            self.episode_number = e_number
            if self.configuration_index < self.total_configurations:
                self.check_update_configuration()
            self.episode_loop()

    def switch_configuration(self, next_point):
        self.configuration_index = int(next_point)
        self.switched_configuration = True
        print(f"{self.trial_id}: Changing configuration to configuration {self.configuration_index}")
        self.params, self.env = self.load_configuration_files()
        self.simulation = ContinuousNaturalisticEnvironment(self.env, self.realistic_bouts)

    def check_update_configuration(self):
        # TODO: Will want to tidy this up later.
        next_point = str(self.configuration_index + 1)
        episode_transition_points = self.episode_transitions.keys()

        if next_point in episode_transition_points:
            if self.episode_number > self.episode_transitions[next_point]:
                self.switch_configuration(next_point)
                return

        if len(self.last_episodes_prey_caught) >= 20:
            prey_conditional_transition_points = self.conditional_transitions["Prey Caught"].keys()
            predators_conditional_transition_points = self.conditional_transitions["Predators Avoided"].keys()
            grains_bumped_conditional_transfer_points = self.conditional_transitions["Sand Grains Bumped"].keys()

            if next_point in predators_conditional_transition_points:
                if np.mean(self.last_episodes_predators_avoided) > self.conditional_transitions["Predators Avoided"][
                    next_point]:
                    self.switch_configuration(next_point)
                    return

            if next_point in prey_conditional_transition_points:
                if np.mean(self.last_episodes_prey_caught) > self.conditional_transitions["Prey Caught"][next_point]:
                    self.switch_configuration(next_point)
                    return

            if next_point in grains_bumped_conditional_transfer_points:
                if np.mean(self.last_episodes_sand_grains_bumped) > self.conditional_transitions["Sand Grains Bumped"][
                    next_point]:
                    self.switch_configuration(next_point)
                    return
        self.switched_configuration = False

    def create_network(self):
        """
        Create the main and target Q networks, according to the configuration parameters.
        :return: The main network and the target network graphs.
        """
        print("Creating network...")
        internal_states = sum([1 for x in [self.env['hunger'], self.env['stress']] if x is True]) + 1
        cell = tf.nn.rnn_cell.LSTMCell(num_units=self.params['rnn_dim'], state_is_tuple=True)
        a2c_network = A2CNetwork(self.simulation, self.params['rnn_dim'], cell, 'main', internal_states=internal_states,
                                 actor_learning_rate_impulse=self.params['learning_rate_impulse'],
                                 actor_learning_rate_angle=self.params['learning_rate_angle'],
                                 critic_learning_rate=self.params['learning_rate_critic'],
                                 max_impulse=self.env['max_impulse'],
                                 max_angle_change=self.env['max_angle_change'])
        return a2c_network

    def episode_loop(self):
        """
        Loops over an episode, which involves initialisation of the environment and RNN state, then iteration over the
        steps in the episode. The relevant values are then saved to the experience buffer.
        """
        t0 = time()

        rnn_state = (
            np.zeros([1, self.a2c_network.rnn_dim]), np.zeros([1, self.a2c_network.rnn_dim]))  # Reset RNN hidden state
        self.simulation.reset()
        sa = np.zeros((1, 128))  # Kept for GIFs.

        # Take the first simulation step, with a capture action. Assigns observation, reward, internal state, done, and
        o, r, internal_state, d, self.frame_buffer = self.simulation.simulation_step(action=[4.0, 0.0],
                                                                                     frame_buffer=self.frame_buffer,
                                                                                     save_frames=self.save_frames,
                                                                                     activations=(sa,))

        # For benchmarking each episode.
        all_actions = []
        total_episode_reward = 0  # Total reward over episode

        step_number = 0  # To allow exit after maximum steps.
        a = [4.0, 0.0]  # Initialise action for episode.
        all_actions.append([a])

        # Reset buffers
        self.action_buffer = []
        self.observation_buffer = []
        self.reward_buffer = []
        self.internal_state_buffer = []
        self.value_buffer = []

        while step_number < self.params["max_epLength"]:
            if step_number != 0 and step_number % self.params["batch_size"] == 0:
                self.train_network_batches()

                # Reset buffers
                self.action_buffer = []
                self.observation_buffer = []
                self.reward_buffer = []
                self.internal_state_buffer = []
                self.value_buffer = []

            step_number += 1
            o, a, r, internal_state, o1, d, rnn_state, V = self.step_loop(o=o, internal_state=internal_state,
                                                                          a=a, rnn_state=rnn_state)

            # Update buffer
            self.action_buffer.append(a)
            self.observation_buffer.append(o)
            self.reward_buffer.append(r)
            self.internal_state_buffer.append(internal_state)
            self.value_buffer.append(V)

            total_episode_reward += r
            # self.train_network(o, o1, r, V, a, all_actions[-1], internal_state)
            all_actions.append([a])
            o = o1

            if d:
                break
        print("\n")
        print(f"Mean Impulse: {np.mean([i[0][0] for i in all_actions])}")
        print(f"Mean Angle {np.mean([i[0][1] for i in all_actions])}")
        print(f"Total episode reward: {total_episode_reward}")
        # Add the episode to the experience buffer
        self.save_episode(episode_start_t=t0,
                          all_actions=all_actions,
                          total_episode_reward=total_episode_reward,
                          prey_caught=self.simulation.prey_caught,
                          predators_avoided=self.simulation.predators_avoided,
                          sand_grains_bumped=self.simulation.sand_grains_bumped,
                          steps_near_vegetation=self.simulation.steps_near_vegetation
                          )
        # Print saved metrics
        # print(f"Total training time: {sum(self.training_times)}")
        # print(f"Total reward: {sum(self.reward_list)}")

    def step_loop(self, o, internal_state, a, rnn_state):
        # Generate actions and corresponding steps.
        sa = np.zeros((1, 128))  # Placeholder for the state advantage stream.
        a = [a[0]/10, a[1]]  # Set impulse to scale. TODO: Change 10 when systematic impulse given
        impulse, angle, updated_rnn_state, V, test = self.sess.run(
            [self.a2c_network.action_tf_var_impulse, self.a2c_network.action_tf_var_angle, self.a2c_network.rnn_state,
             self.a2c_network.Value, self.a2c_network.conv4l_flat],
            feed_dict={self.a2c_network.observation: o,
                       self.a2c_network.internal_state: internal_state,
                       self.a2c_network.prev_actions: np.reshape(a, (1, 2)),
                       self.a2c_network.trainLength: 1,
                       self.a2c_network.state_in: rnn_state,
                       self.a2c_network.batch_size: 1})
        # print(impulse)
        impulse = impulse[0][0]
        angle = angle[0][0]
        action = [impulse, angle]

        # Simulation step
        o1, given_reward, internal_state, d, self.frame_buffer = self.simulation.simulation_step(
            action=action,
            frame_buffer=self.frame_buffer,
            save_frames=self.save_frames,
            activations=sa)

        self.total_steps += 1
        return o, action, given_reward, internal_state, o1, d, updated_rnn_state, V

    def train_network(self, observation_1, observation_2, reward, V, action, previous_actions, internal_state_1):
        """
        For training without batches (not currently used).
        """
        # Reset the recurrent layer's hidden state
        state_train = (np.zeros([1, self.a2c_network.rnn_dim]),
                       np.zeros([1, self.a2c_network.rnn_dim]))

        V1, internal_state_2 = self.sess.run([self.a2c_network.Value, self.a2c_network.internal_state],
                                             feed_dict={self.a2c_network.observation: observation_1,
                                                        self.a2c_network.trainLength: 1,
                                                        self.a2c_network.batch_size: 1,
                                                        self.a2c_network.prev_actions: np.reshape(previous_actions, (1, 2)),
                                                        self.a2c_network.internal_state: internal_state_1,
                                                        })
        V_of_next_state = self.sess.run(self.a2c_network.Value,
                                        feed_dict={self.a2c_network.observation: observation_2,
                                                   self.a2c_network.trainLength: 1,
                                                   self.a2c_network.batch_size: 1,
                                                   self.a2c_network.prev_actions: np.reshape(action, (1, 2)),
                                                   self.a2c_network.internal_state: internal_state_2,
                                                   })

        target = reward + self.gamma * np.squeeze(V_of_next_state)
        td_error = target - np.squeeze(V1)  # The advantage.

        # Loss function actor (impulse)
        _, loss_actor_val_impulse = self.sess.run(
            [self.a2c_network.training_op_actor_impulse, self.a2c_network.loss_actor_impulse],
            feed_dict={self.a2c_network.action_placeholder: np.reshape(action, (1, 2)),
                       self.a2c_network.observation: observation_1,
                       self.a2c_network.delta_placeholder: td_error,
                       self.a2c_network.prev_actions: np.reshape(previous_actions, (1, 2)),
                       self.a2c_network.trainLength: 1,
                       self.a2c_network.internal_state: internal_state_1,
                       self.a2c_network.state_in: state_train,
                       self.a2c_network.batch_size: 1,
                       })

        # Loss function actor (angle)
        _, loss_actor_val_angle = self.sess.run(
            [self.a2c_network.training_op_actor_angle, self.a2c_network.loss_actor_angle],
            feed_dict={self.a2c_network.action_placeholder: np.reshape(action, (1, 2)),
                       self.a2c_network.observation: observation_1,
                       self.a2c_network.delta_placeholder: td_error,
                       self.a2c_network.prev_actions: np.reshape(previous_actions, (1, 2)),
                       self.a2c_network.trainLength: 1,
                       self.a2c_network.internal_state: internal_state_1,
                       self.a2c_network.state_in: state_train,
                       self.a2c_network.batch_size: 1,
                       })

        # Update critic by minimizing loss  (Critic training)
        _, loss_critic_val = self.sess.run(
            [self.a2c_network.training_op_critic, self.a2c_network.loss_critic],
            feed_dict={self.a2c_network.observation: observation_1,
                       self.a2c_network.target_placeholder: target,
                       self.a2c_network.prev_actions: np.reshape(previous_actions, (1, 2)),
                       self.a2c_network.trainLength: 1,
                       self.a2c_network.internal_state: internal_state_1,
                       self.a2c_network.state_in: state_train,
                       self.a2c_network.batch_size: 1,
                       })

    def train_network_batches(self):
        self.observation_buffer = np.array(self.observation_buffer)
        self.action_buffer = np.array(self.action_buffer)
        self.reward_buffer = np.array(self.reward_buffer)
        self.value_buffer = np.array(self.value_buffer)
        self.internal_state_buffer = np.array(self.internal_state_buffer)

        advantages = self.reward_buffer - self.value_buffer  # TODO: Implement
        state_train = (np.zeros([self.params["batch_size"]-1, self.a2c_network.rnn_dim]),
                       np.zeros([self.params["batch_size"]-1, self.a2c_network.rnn_dim]))

        V1, internal_state_2 = self.sess.run([self.a2c_network.Value, self.a2c_network.internal_state],
                                             feed_dict={self.a2c_network.observation: np.vstack(self.observation_buffer[:-1, :]),
                                                        self.a2c_network.trainLength: 1,
                                                        self.a2c_network.batch_size: self.params["batch_size"] -1,
                                                        self.a2c_network.prev_actions: np.vstack(self.action_buffer[:-1, :]),
                                                        self.a2c_network.internal_state: np.vstack(self.internal_state_buffer[:-1, :]),
                                                        })
        V_of_next_state = self.sess.run(self.a2c_network.Value,
                                        feed_dict={self.a2c_network.observation: np.vstack(self.observation_buffer[1:, :]),
                                                   self.a2c_network.trainLength: 1,
                                                   self.a2c_network.batch_size: self.params["batch_size"] -1,
                                                   self.a2c_network.prev_actions: np.vstack(self.action_buffer[1:, :]),
                                                   self.a2c_network.internal_state: np.vstack(self.internal_state_buffer[1:, :]),
                                                   })

        target = self.reward_buffer[1:] + self.gamma * np.squeeze(V_of_next_state)
        td_error = target - np.squeeze(V1)

        # Loss function actor (impulse)
        _, loss_actor_val_impulse = self.sess.run(
            [self.a2c_network.training_op_actor_impulse, self.a2c_network.loss_actor_impulse],
            feed_dict={self.a2c_network.action_placeholder: np.vstack(self.action_buffer[1:, :]),
                       self.a2c_network.observation: np.vstack(self.observation_buffer[:-1, :]),
                       self.a2c_network.delta_placeholder: td_error,
                       self.a2c_network.prev_actions: np.vstack(self.action_buffer[:-1, :]),
                       self.a2c_network.trainLength: 1,
                       self.a2c_network.internal_state: np.vstack(self.internal_state_buffer[:-1, :]),
                       self.a2c_network.state_in: state_train,
                       self.a2c_network.batch_size: self.params["batch_size"] -1,
                       })

        # Loss function actor (angle)
        _, loss_actor_val_angle = self.sess.run(
            [self.a2c_network.training_op_actor_angle, self.a2c_network.loss_actor_angle],
            feed_dict={self.a2c_network.action_placeholder: np.vstack(self.action_buffer[1:, :]),
                       self.a2c_network.observation: np.vstack(self.observation_buffer[:-1, :]),
                       self.a2c_network.delta_placeholder: td_error,
                       self.a2c_network.prev_actions:  np.vstack(self.action_buffer[:-1, :]),
                       self.a2c_network.trainLength: 1,
                       self.a2c_network.internal_state: np.vstack(self.internal_state_buffer[:-1, :]),
                       self.a2c_network.state_in: state_train,
                       self.a2c_network.batch_size: self.params["batch_size"] -1,
                       })

        # Update critic by minimizing loss  (Critic training)
        _, loss_critic_val = self.sess.run(
            [self.a2c_network.training_op_critic, self.a2c_network.loss_critic],
            feed_dict={self.a2c_network.action_placeholder: np.vstack(self.action_buffer[1:, :]),
                       self.a2c_network.observation: np.vstack(self.observation_buffer[:-1, :]),
                       self.a2c_network.delta_placeholder: td_error,
                       self.a2c_network.target_placeholder: target,
                       self.a2c_network.prev_actions:  np.vstack(self.action_buffer[:-1, :]),
                       self.a2c_network.trainLength: 1,
                       self.a2c_network.internal_state: np.vstack(self.internal_state_buffer[:-1, :]),
                       self.a2c_network.state_in: state_train,
                       self.a2c_network.batch_size: self.params["batch_size"] -1,
                       })

    def save_episode(self, episode_start_t, all_actions, total_episode_reward, prey_caught,
                     predators_avoided, sand_grains_bumped, steps_near_vegetation):
        """
        Saves the episode the the experience buffer. Also creates a gif if at interval.
        :param episode_start_t: The time at the start of the episode, used to calculate the time the episode took.
        :param all_actions: The array of all the actions taken during the episode.
        :param total_episode_reward: The total reward of the episode.
        the environment.
        :return:
        """

        print(f"{self.trial_id} - episode {str(self.episode_number)}: num steps = {str(self.simulation.num_steps)}",
              flush=True)

        # # Log the average training time for episodes (when not saved)
        # if not self.save_frames:
        #     self.training_times.append(time() - episode_start_t)
        #     print(np.mean(self.training_times))

        # Keep recent predators caught.
        self.last_episodes_prey_caught.append(prey_caught)
        self.last_episodes_predators_avoided.append(predators_avoided)
        self.last_episodes_sand_grains_bumped.append(sand_grains_bumped)
        if len(self.last_episodes_predators_avoided) > 20:
            self.last_episodes_prey_caught.pop(0)
            self.last_episodes_predators_avoided.pop(0)
            self.last_episodes_sand_grains_bumped.pop(0)

        # Add Summary to Logs
        episode_summary = tf.Summary(value=[tf.Summary.Value(tag="episode reward", simple_value=total_episode_reward)])
        self.writer.add_summary(episode_summary, self.total_steps)

        # Action Summary
        impulses = [action[0][0] for action in all_actions]
        for step in range(0, len(impulses), 5):
            impulse_summary = tf.Summary(value=[tf.Summary.Value(tag="impulse magnitude", simple_value=impulses[step])])
            self.writer.add_summary(impulse_summary, self.total_steps-len(impulses)+step)

        angles = [action[0][1] for action in all_actions]
        for step in range(0, len(angles), 5):
            angles_summary = tf.Summary(value=[tf.Summary.Value(tag="angle magnitude", simple_value=angles[step])])
            self.writer.add_summary(angles_summary, self.total_steps-len(angles)+step)

        # Raw logs
        prey_caught_summary = tf.Summary(value=[tf.Summary.Value(tag="prey caught", simple_value=prey_caught)])
        self.writer.add_summary(prey_caught_summary, self.episode_number)

        predators_avoided_summary = tf.Summary(
            value=[tf.Summary.Value(tag="predators avoided", simple_value=predators_avoided)])
        self.writer.add_summary(predators_avoided_summary, self.episode_number)

        sand_grains_bumped_summary = tf.Summary(
            value=[tf.Summary.Value(tag="attempted sand grain captures", simple_value=sand_grains_bumped)])
        self.writer.add_summary(sand_grains_bumped_summary, self.episode_number)

        steps_near_vegetation_summary = tf.Summary(
            value=[tf.Summary.Value(tag="steps near vegetation", simple_value=steps_near_vegetation)])
        self.writer.add_summary(steps_near_vegetation_summary, self.episode_number)

        # Normalised Logs
        if self.env["prey_num"] != 0:
            fraction_prey_caught = prey_caught / self.env["prey_num"]
            prey_caught_summary = tf.Summary(
                value=[tf.Summary.Value(tag="prey capture index (fraction caught)", simple_value=fraction_prey_caught)])
            self.writer.add_summary(prey_caught_summary, self.episode_number)

        if self.env["probability_of_predator"] != 0:
            predator_avoided_index = predators_avoided / self.env["probability_of_predator"]
            predators_avoided_summary = tf.Summary(
                value=[tf.Summary.Value(tag="predator avoidance index (avoided/p_pred)",
                                        simple_value=predator_avoided_index)])
            self.writer.add_summary(predators_avoided_summary, self.episode_number)

        if self.env["sand_grain_num"] != 0:
            sand_grain_capture_index = sand_grains_bumped / self.env["sand_grain_num"]
            sand_grains_bumped_summary = tf.Summary(
                value=[tf.Summary.Value(tag="sand grain capture index (fraction attempted caught)",
                                        simple_value=sand_grain_capture_index)])
            self.writer.add_summary(sand_grains_bumped_summary, self.episode_number)

        if self.env["vegetation_num"] != 0:
            vegetation_index = (steps_near_vegetation / self.simulation.num_steps) / self.env["vegetation_num"]
            use_of_vegetation_summary = tf.Summary(
                value=[tf.Summary.Value(tag="use of vegetation index (fraction_steps/vegetation_num",
                                        simple_value=vegetation_index)])
            self.writer.add_summary(use_of_vegetation_summary, self.episode_number)

        if self.switched_configuration:
            configuration_summary = tf.Summary(
                value=[tf.Summary.Value(tag="Configuration change", simple_value=self.configuration_index)]
            )
            self.writer.add_summary(configuration_summary, self.episode_number)

        # Save the parameters to be carried over.
        output_data = {"episode_number": self.episode_number, "total_steps": self.total_steps}
        with open(f"{self.output_location}/saved_parameters.json", "w") as file:
            json.dump(output_data, file)

        self.reward_list.append(total_episode_reward)
        # Periodically save the model.
        if self.episode_number % self.params['summaryLength'] == 0 and self.episode_number != 0:
            # print(f"mean time: {np.mean(self.training_times)}")

            # Save the model
            self.saver.save(self.sess, f"{self.output_location}/model-{str(self.episode_number)}.cptk")
            print("Saved Model")

            # Create the GIF
            make_gif(self.frame_buffer, f"{self.output_location}/episodes/episode-{str(self.episode_number)}.gif",
                     duration=len(self.frame_buffer) * self.params['time_per_step'], true_image=True)
            self.frame_buffer = []
            self.save_frames = False

        if (self.episode_number + 1) % self.params['summaryLength'] == 0:
            print('starting to save frames', flush=True)
            self.save_frames = True
        if self.monitor_gpu:
            print(f"GPU usage {os.system('gpustat -cp')}")

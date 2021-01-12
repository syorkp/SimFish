from time import time
import json
import os

import numpy as np
import tensorflow.compat.v1 as tf

from Environment.naturalistic_environment import NaturalisticEnvironment
from Network.q_network import QNetwork
from Network.experience_buffer import ExperienceBuffer
from Tools.graph_functions import update_target_graph, update_target
from Tools.make_gif import make_gif

tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)


def training_target(trial, epsilon, total_steps, episode_number, memory_fraction):
    using_gpu = tf.test.is_gpu_available(cuda_only=True)
    print(f"Using GPU: {using_gpu}")  # TODO: Test in next model ouptut. If true, replace teh using_gpu configuration parameter with this. Add in for assay_service also
    services = TrainingService(model_name=trial["Model Name"],
                               trial_number=trial["Trial Number"],
                               model_exists=trial["Model Exists"],
                               fish_mode=trial["Fish Setup"],
                               scaffold_name=trial["Environment Name"],
                               episode_transitions=trial["Episode Transitions"],
                               total_configurations=trial["Total Configurations"],
                               conditional_transitions=trial["Conditional Transitions"],
                               e=epsilon,
                               total_steps=total_steps,
                               episode_number=episode_number,
                               monitor_gpu=trial["monitor gpu"],
                               realistic_bouts=trial["Realistic Bouts"],
                               memory_fraction=memory_fraction,
                               using_gpu=trial["Using GPU"]
                               )
    services.run()


class TrainingService:

    def __init__(self, model_name, trial_number, model_exists, fish_mode, scaffold_name, episode_transitions,
                 total_configurations, conditional_transitions,
                 e, total_steps, episode_number, monitor_gpu, realistic_bouts, memory_fraction, using_gpu):
        """
        An instance of TraningService handles the training of the DQN within a specified environment, according to
        specified parameters.
        :param model_name: The name of the model, usually to match the naming of the env configuration files.
        :param trial_number: The index of the trial, so that agents trained under the same configuration may be
        distinguished in their output files.
        """

        self.trial_id = f"{model_name}-{trial_number}"
        self.output_location = f"./Training-Output/{model_name}-{trial_number}"

        self.load_model = model_exists
        self.monitor_gpu = monitor_gpu
        self.scaffold_name = scaffold_name
        self.using_gpu = using_gpu

        self.realistic_bouts = realistic_bouts
        self.total_configurations = total_configurations
        self.episode_transitions = episode_transitions
        self.conditional_transitions = conditional_transitions
        self.memory_fraction = memory_fraction

        self.configuration_index = 1

        self.params, self.env = self.load_configuration_files()

        # Create the training environment.
        self.apparatus_mode = fish_mode
        self.simulation = NaturalisticEnvironment(self.env, realistic_bouts)
        self.realistic_bouts = realistic_bouts

        # Experience buffer
        self.training_buffer = ExperienceBuffer(buffer_size=self.params["exp_buffer_size"])

        self.saver = None
        self.frame_buffer = []

        # Mathematical variables
        if e is not None:
            self.e = e
        else:
            self.e = self.params["startE"]
        if episode_number is not None:
            self.episode_number = episode_number + 1
        else:
            self.episode_number = 0

        # While would fix TB output, Not possible to carry steps over without also carrying the training buffer over.
        if total_steps is not None:
            self.total_steps = total_steps
        else:
            self.total_steps = 0

        self.pre_train_steps = self.total_steps + self.params["pre_train_steps"]

        self.step_drop = (self.params['startE'] - self.params['endE']) / self.params['anneling_steps']

        # Whether to save the frames of an episode
        self.save_frames = False

        # To save the graph (placeholder)
        self.writer = None

        # Global tensorflow placeholders
        self.main_QN, self.target_QN = None, None
        self.init = None
        self.trainables = None
        self.target_ops = None
        self.sess = None  # Placeholder for the tf-session.

        # Used for performance monitoring.
        self.training_times = []
        self.reward_list = []

        self.last_episodes_prey_caught = []
        self.last_episodes_predators_avoided = []

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
        self.main_QN, self.target_QN = self.create_networks()
        self.saver = tf.train.Saver(max_to_keep=5)
        self.init = tf.global_variables_initializer()
        self.trainables = tf.trainable_variables()
        self.target_ops = update_target_graph(self.trainables, self.params['tau'])
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

        update_target(self.target_ops, self.sess)  # Set the target network to be equal to the primary network.
        self.writer = tf.summary.FileWriter(f"{self.output_location}/logs/", tf.get_default_graph())

        for e_number in range(self.episode_number, self.params["num_episodes"]):
            self.episode_number = e_number
            if self.configuration_index < self.total_configurations:
                self.check_update_configuration()
            self.episode_loop()

    def check_update_configuration(self):
        # TODO: Will want to tidy this up later.
        next_point = str(self.configuration_index + 1)
        episode_transition_points = self.episode_transitions.keys()

        if next_point in episode_transition_points:
            if self.episode_number > self.episode_transitions[next_point]:
                print("Changing configuration")
                self.configuration_index = int(next_point)
                print(f"Configuration: {self.configuration_index}")
                self.params, self.env = self.load_configuration_files()
                self.simulation = NaturalisticEnvironment(self.env, self.realistic_bouts)
                return

        if len(self.last_episodes_prey_caught) > 20:
            prey_conditional_transition_points = self.conditional_transitions["Prey Caught"].keys()
            predators_conditional_transition_points = self.conditional_transitions["Predators Avoided"].keys()
            print(np.mean(self.last_episodes_prey_caught))
            if next_point in predators_conditional_transition_points:
                if np.mean(self.last_episodes_predators_avoided) > self.conditional_transitions["Predators Avoided"][next_point]:
                    print("Changing configuration")
                    self.configuration_index = int(next_point)
                    print(f"Configuration: {self.configuration_index}")

                    self.params, self.env = self.load_configuration_files()
                    self.simulation = NaturalisticEnvironment(self.env, self.realistic_bouts)
                    return
            if next_point in prey_conditional_transition_points:
                if np.mean(self.last_episodes_prey_caught) > self.conditional_transitions["Prey Caught"][next_point]:
                    print("Changing configuration")
                    self.configuration_index = int(next_point)
                    print(f"Configuration: {self.configuration_index}")

                    self.params, self.env = self.load_configuration_files()
                    self.simulation = NaturalisticEnvironment(self.env, self.realistic_bouts)
                    return

    def create_networks(self):
        """
        Create the main and target Q networks, according to the configuration parameters.
        :return: The main network and the target network graphs.
        """
        print("Creating networks...")
        cell = tf.nn.rnn_cell.LSTMCell(num_units=self.params['rnn_dim'], state_is_tuple=True)
        cell_t = tf.nn.rnn_cell.LSTMCell(num_units=self.params['rnn_dim'], state_is_tuple=True)
        main_QN = QNetwork(self.simulation, self.params['rnn_dim'], cell, 'main', self.params['num_actions'],
                           learning_rate=self.params['learning_rate'])
        target_QN = QNetwork(self.simulation, self.params['rnn_dim'], cell_t, 'target', self.params['num_actions'],
                             learning_rate=self.params['learning_rate'])
        return main_QN, target_QN

    def episode_loop(self):
        """
        Loops over an episode, which involves initialisation of the environment and RNN state, then iteration over the
        steps in the episode. The relevant values are then saved to the experience buffer.
        """
        t0 = time()
        episode_buffer = []

        rnn_state = (np.zeros([1, self.main_QN.rnn_dim]), np.zeros([1, self.main_QN.rnn_dim]))  # Reset RNN hidden state
        self.simulation.reset()
        sa = np.zeros((1, 128))  # Placeholder for the state advantage stream.
        sv = np.zeros((1, 128))  # Placeholder for the state value stream

        # Take the first simulation step, with a capture action. Assigns observation, reward, internal state, done, and
        o, r, internal_state, d, self.frame_buffer = self.simulation.simulation_step(action=3,
                                                                                     frame_buffer=self.frame_buffer,
                                                                                     save_frames=self.save_frames,
                                                                                     activations=(sa,))

        # For benchmarking each episode.
        all_actions = []
        total_episode_reward = 0  # Total reward over episode

        step_number = 0  # To allow exit after maximum steps.
        a = 0  # Initialise action for episode.
        while step_number < self.params["max_epLength"]:
            step_number += 1
            o, a, r, internal_state, o1, d, rnn_state = self.step_loop(o=o, internal_state=internal_state,
                                                                       a=a, rnn_state=rnn_state)
            all_actions.append(a)
            episode_buffer.append(np.reshape(np.array([o, a, r, internal_state, o1, d]), [1, 6]))
            total_episode_reward += r
            o = o1
            if self.total_steps > self.pre_train_steps:
                if self.e > self.params['endE']:
                    self.e -= self.step_drop
                if self.total_steps % (self.params['update_freq']) == 0:
                    self.train_networks()
            if d:
                break
        # Add the episode to the experience buffer
        self.save_episode(episode_start_t=t0,
                          all_actions=all_actions,
                          total_episode_reward=total_episode_reward,
                          episode_buffer=episode_buffer,
                          prey_caught=self.simulation.prey_caught,
                          predators_avoided=self.simulation.predators_avoided,
                          )
        # Print saved metrics
        # print(f"Total training time: {sum(self.training_times)}")
        # print(f"Total reward: {sum(self.reward_list)}")

    def save_episode(self, episode_start_t, all_actions, total_episode_reward, episode_buffer, prey_caught,
                     predators_avoided):
        """
        Saves the episode the the experience buffer. Also creates a gif if at interval.
        :param episode_start_t: The time at the start of the episode, used to calculate the time the episode took.
        :param all_actions: The array of all the actions taken during the episode.
        :param total_episode_reward: The total reward of the episode.
        :param episode_buffer: A buffer containing all the state transitions, actions and associated rewards yielded by
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
        if len(self.last_episodes_predators_avoided) > 20:
            self.last_episodes_prey_caught.pop(0)
            self.last_episodes_predators_avoided.pop(0)

        # Add Summary to Logs
        episode_summary = tf.Summary(value=[tf.Summary.Value(tag="episode reward", simple_value=total_episode_reward)])
        self.writer.add_summary(episode_summary, self.total_steps)

        # Consider changing two below to total steps.
        prey_caught_summary = tf.Summary(value=[tf.Summary.Value(tag="prey caught", simple_value=prey_caught)])
        self.writer.add_summary(prey_caught_summary, self.episode_number)

        predators_avoided_summary = tf.Summary(
            value=[tf.Summary.Value(tag="predators avoided", simple_value=predators_avoided)])
        self.writer.add_summary(predators_avoided_summary, self.episode_number)

        for act in range(self.params['num_actions']):
            action_freq = np.sum(np.array(all_actions) == act) / len(all_actions)
            a_freq = tf.Summary(value=[tf.Summary.Value(tag="action " + str(act), simple_value=action_freq)])
            self.writer.add_summary(a_freq, self.total_steps)

        # Save the parameters to be carried over.
        output_data = {"epsilon": self.e, "episode_number": self.episode_number, "total_steps": self.total_steps}
        with open(f"{self.output_location}/saved_parameters.json", "w") as file:
            json.dump(output_data, file)

        buffer_array = np.array(episode_buffer)
        episode_buffer = list(zip(buffer_array))
        self.training_buffer.add(episode_buffer)
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

    def step_loop(self, o, internal_state, a, rnn_state):
        """
        Runs a step, choosing an action given an initial condition using the network/randomly, and running this in the
        environment.

        :param
        session: The TF session.
        internal_state: The internal state of the agent - whether it is in light, and whether it is hungry.
        a: The previous chosen action.
        rnn_state: The state inside the RNN.

        :return:
        s: The environment state.
        chosen_a: The action chosen randomly/by the network.
        given_reward: The reward returned.
        internal_state: The internal state of the agent - whether it is in light, and whether it is hungry.
        s1: The subsequent environment state
        d: Boolean indicating agent death.
        updated_rnn_state: The updated RNN state
        """

        # Generate actions and corresponding steps.
        if np.random.rand(1) < self.e or self.total_steps < self.pre_train_steps:
            [updated_rnn_state, sa, sv] = self.sess.run(
                [self.main_QN.rnn_state, self.main_QN.streamA, self.main_QN.streamV],
                feed_dict={self.main_QN.observation: o,
                           self.main_QN.internal_state: internal_state,
                           self.main_QN.prev_actions: [a],
                           self.main_QN.trainLength: 1,
                           self.main_QN.state_in: rnn_state,
                           self.main_QN.batch_size: 1,
                           self.main_QN.exp_keep: 1.0})
            chosen_a = np.random.randint(0, self.params['num_actions'])
        else:
            chosen_a, updated_rnn_state, sa, sv = self.sess.run(
                [self.main_QN.predict, self.main_QN.rnn_state, self.main_QN.streamA, self.main_QN.streamV],
                feed_dict={self.main_QN.observation: o,
                           self.main_QN.internal_state: internal_state,
                           self.main_QN.prev_actions: [a],
                           self.main_QN.trainLength: 1,
                           self.main_QN.state_in: rnn_state,
                           self.main_QN.batch_size: 1,
                           self.main_QN.exp_keep: 1.0})
            chosen_a = chosen_a[0]

        # Simulation step
        o1, given_reward, internal_state, d, self.frame_buffer = self.simulation.simulation_step(action=chosen_a,
                                                                                                 frame_buffer=self.frame_buffer,
                                                                                                 save_frames=self.save_frames,
                                                                                                 activations=(sa,))
        self.total_steps += 1
        return o, chosen_a, given_reward, internal_state, o1, d, updated_rnn_state

    def train_networks(self):
        """
        Trains the two networks, copying over the target network
        :return:
        """
        update_target(self.target_ops, self.sess)
        # Reset the recurrent layer's hidden state
        state_train = (np.zeros([self.params['batch_size'], self.main_QN.rnn_dim]),
                       np.zeros([self.params['batch_size'], self.main_QN.rnn_dim]))

        # Get a random batch of experiences: ndarray 1024x6, with the six columns containing o, a, r, i_s, o1, d
        train_batch = self.training_buffer.sample(self.params['batch_size'], self.params['trace_length'])

        # Below we perform the Double-DQN update to the target Q-values
        Q1 = self.sess.run(self.main_QN.predict, feed_dict={
            self.main_QN.observation: np.vstack(train_batch[:, 4]),
            self.main_QN.prev_actions: np.hstack(([0], train_batch[:-1, 1])),
            self.main_QN.trainLength: self.params['trace_length'],
            self.main_QN.internal_state: np.vstack(train_batch[:, 3]),
            self.main_QN.state_in: state_train,
            self.main_QN.batch_size: self.params['batch_size'],
            self.main_QN.exp_keep: 1.0})

        Q2 = self.sess.run(self.target_QN.Q_out, feed_dict={
            self.target_QN.observation: np.vstack(train_batch[:, 4]),
            self.target_QN.prev_actions: np.hstack(([0], train_batch[:-1, 1])),
            self.target_QN.trainLength: self.params['trace_length'],
            self.target_QN.internal_state: np.vstack(train_batch[:, 3]),
            self.target_QN.state_in: state_train,
            self.target_QN.batch_size: self.params['batch_size'],
            self.target_QN.exp_keep: 1.0})

        end_multiplier = -(train_batch[:, 5] - 1)

        double_Q = Q2[range(self.params['batch_size'] * self.params['trace_length']), Q1]
        target_Q = train_batch[:, 2] + (self.params['y'] * double_Q * end_multiplier)
        # Update the network with our target values.
        self.sess.run(self.main_QN.updateModel,
                      feed_dict={self.main_QN.observation: np.vstack(train_batch[:, 0]),
                                 self.main_QN.targetQ: target_Q,
                                 self.main_QN.actions: train_batch[:, 1],
                                 self.main_QN.internal_state: np.vstack(train_batch[:, 3]),
                                 self.main_QN.prev_actions: np.hstack(([3], train_batch[:-1, 1])),
                                 self.main_QN.trainLength: self.params['trace_length'],
                                 self.main_QN.state_in: state_train,
                                 self.main_QN.batch_size: self.params['batch_size'],
                                 self.main_QN.exp_keep: 1.0})

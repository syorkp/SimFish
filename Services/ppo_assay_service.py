import json
import h5py
from datetime import datetime

import numpy as np
import tensorflow.compat.v1 as tf

from Environment.continuous_naturalistic_environment import ContinuousNaturalisticEnvironment
from Environment.controlled_stimulus_environment import ControlledStimulusEnvironment
from Network.proximal_policy_optimizer import PPONetworkActor, PPONetworkCritic
from Buffers.ppo_buffer import PPOBuffer
from Tools.make_gif import make_gif

tf.logging.set_verbosity(tf.logging.ERROR)


def ppo_assay_target(trial, learning_params, environment_params, total_steps, episode_number, memory_fraction):
    service = PPOAssayService(model_name=trial["Model Name"],
                              trial_number=trial["Trial Number"],
                              assay_config_name=trial["Assay Configuration Name"],
                              learning_params=learning_params,
                              environment_params=environment_params,
                              total_steps=total_steps,
                              episode_number=episode_number,
                              assays=trial["Assays"],
                              realistic_bouts=trial["Realistic Bouts"],
                              memory_fraction=memory_fraction,
                              using_gpu=trial["Using GPU"],
                              set_random_seed=trial["set random seed"]
                              )
    service.run()


class PPOAssayService:

    def __init__(self, model_name, trial_number, assay_config_name, learning_params, environment_params, total_steps,
                 episode_number, assays, realistic_bouts, memory_fraction, using_gpu, set_random_seed):
        """
        Runs a set of assays provided by the run configuraiton.
        """

        # Set random seed
        if set_random_seed:
            np.random.seed(404)

        # Names and Directories
        self.model_id = f"{model_name}-{trial_number}"
        self.model_location = f"./Training-Output/{self.model_id}"
        self.data_save_location = f"./Assay-Output/{self.model_id}"

        # Configurations
        self.assay_configuration_id = assay_config_name
        self.learning_params = learning_params
        self.environment_params = environment_params
        self.assays = assays

        # Basic Parameters
        self.using_gpu = using_gpu
        self.realistic_bouts = realistic_bouts
        self.memory_fraction = memory_fraction

        # Network Parameters
        self.saver = None
        self.actor_network = None
        self.critic_network = None
        self.init = None
        self.sess = None

        # Simulation
        self.simulation = ContinuousNaturalisticEnvironment(self.environment_params, self.realistic_bouts)
        self.step_number = 0

        # Data
        self.metadata = {
            "Total Episodes": episode_number,
            "Total Steps": total_steps,
        }
        self.total_steps = total_steps
        self.frame_buffer = []
        self.assay_output_data_format = None
        self.assay_output_data = []
        self.output_data = {}
        self.episode_summary_data = None

        self.save_frames = False

        # Hacky fix for h5py problem:
        self.last_position_dim = self.environment_params["prey_num"]
        self.stimuli_data = []

        # Buffer for saving results of assay
        self.buffer = PPOBuffer(gamma=0.99, lmbda=0.9, batch_size=self.learning_params["batch_size"],
                                train_length=self.learning_params["trace_length"], assay=True, debug=False)

        self.impulse_sigma = None
        self.angle_sigma = None

    def update_sigmas(self):
        self.impulse_sigma = np.array([self.environment_params["min_sigma_impulse"] + (
                    self.environment_params["max_sigma_impulse"] - self.environment_params["min_sigma_impulse"]) * np.e ** (
                                                   -self.total_steps * self.environment_params["sigma_time_constant"])])
        self.angle_sigma = np.array([self.environment_params["min_sigma_angle"] + (
                    self.environment_params["max_sigma_angle"] - self.environment_params["min_sigma_angle"]) * np.e ** (
                                                 -self.total_steps * self.environment_params["sigma_time_constant"])])

    def create_network(self):
        print("Creating networks...")
        internal_states = sum(
            [1 for x in [self.environment_params['hunger'], self.environment_params['stress']] if x is True]) + 1

        actor_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.learning_params['rnn_dim_shared'], state_is_tuple=True)
        critic_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.learning_params['rnn_dim_shared'], state_is_tuple=True)

        ppo_network_actor = PPONetworkActor(simulation=self.simulation,
                                            rnn_dim=self.learning_params['rnn_dim_shared'],
                                            rnn_cell=actor_cell,
                                            my_scope='actor',
                                            internal_states=internal_states,
                                            learning_rate=self.learning_params['learning_rate_actor'],
                                            max_impulse=self.environment_params['max_impulse'],
                                            max_angle_change=self.environment_params['max_angle_change'],
                                            clip_param=self.environment_params['clip_param']
                                            )
        ppo_network_critic = PPONetworkCritic(simulation=self.simulation,
                                              rnn_dim=self.learning_params['rnn_dim_shared'],
                                              rnn_cell=critic_cell,
                                              my_scope='critic',
                                              internal_states=internal_states,
                                              learning_rate=self.learning_params['learning_rate_critic'],
                                              )
        return ppo_network_actor, ppo_network_critic

    def create_testing_environment(self, assay):
        """
        Creates the testing environment as specified  by apparatus mode and given assays.
        :return:
        """
        if assay["stimulus paradigm"] == "Projection":
            self.simulation = ControlledStimulusEnvironment(self.environment_params, assay["stimuli"],
                                                            self.realistic_bouts,
                                                            tethered=assay["Tethered"],
                                                            set_positions=assay["set positions"],
                                                            random=assay["random positions"],
                                                            moving=assay["moving"],
                                                            reset_each_step=assay["reset"],
                                                            reset_interval=assay["reset interval"],
                                                            background=assay["background"]
                                                            )
        elif assay["stimulus paradigm"] == "Naturalistic":
            self.simulation = ContinuousNaturalisticEnvironment(self.environment_params, self.realistic_bouts,
                                                                collisions=assay["collisions"])
        else:
            self.simulation = ContinuousNaturalisticEnvironment(self.environment_params, self.realistic_bouts)

    def run(self):
        if self.using_gpu:
            options = tf.GPUOptions(per_process_gpu_memory_fraction=self.memory_fraction)
        else:
            options = None

        if options:
            with tf.Session(config=tf.ConfigProto(gpu_options=options)) as self.sess:
                self._run()
        else:
            with tf.Session() as self.sess:
                self._run()

    def _run(self):
        self.actor_network, self.critic_network = self.create_network()
        self.saver = tf.train.Saver(max_to_keep=5)
        self.init = tf.global_variables_initializer()
        checkpoint = tf.train.get_checkpoint_state(self.model_location)
        self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
        print("Model loaded")
        for assay in self.assays:
            if assay["ablations"]:
                self.ablate_units(assay["ablations"])
            self.save_frames = assay["save frames"]
            self.create_output_data_storage(assay)
            self.create_testing_environment(assay)
            self.perform_assay(assay)
            if assay["save stimuli"]:
                self.save_stimuli_data(assay)
            # self.save_assay_results(assay)
            # self.save_hdf5_data(assay)
        self.save_metadata()
        self.save_episode_data()

    def perform_assay(self, assay):
        self.assay_output_data_format = {key: None for key in assay["recordings"]}
        self.buffer.recordings = assay["recordings"]
        self.update_sigmas()

        rnn_state_actor = (
            np.zeros([1, self.actor_network.rnn_dim]),
            np.zeros([1, self.actor_network.rnn_dim]))  # Reset RNN hidden state
        rnn_state_actor_ref = (
            np.zeros([1, self.actor_network.rnn_dim]),
            np.zeros([1, self.actor_network.rnn_dim]))  # Reset RNN hidden state

        rnn_state_critic = (
            np.zeros([1, self.actor_network.rnn_dim]),
            np.zeros([1, self.actor_network.rnn_dim]))  # Reset RNN hidden state
        rnn_state_critic_ref = (
            np.zeros([1, self.actor_network.rnn_dim]),
            np.zeros([1, self.actor_network.rnn_dim]))

        self.simulation.reset()
        sa = np.zeros((1, 128))  # Kept for GIFs.

        # Take the first simulation step, with a capture action. Assigns observation, reward, internal state, done, and
        o, r, internal_state, d, self.frame_buffer = self.simulation.simulation_step(action=[4.0, 0.0],
                                                                                     frame_buffer=self.frame_buffer,
                                                                                     save_frames=self.save_frames,
                                                                                     activations=(sa,))

        # For benchmarking each episode.
        total_episode_reward = 0  # Total reward over episode

        a = [4.0, 0.0]  # Initialise action for episode.

        # Reset buffers
        self.buffer.reset()
        self.buffer.action_buffer.append(a)  # Add to buffer for loading of previous actions

        self.step_number = 0
        while self.step_number < assay["duration"]:
            if assay["reset"] and self.step_number % assay["reset interval"] == 0:
                rnn_state_actor = (
                    np.zeros([1, self.actor_network.rnn_dim]),
                    np.zeros([1, self.actor_network.rnn_dim]))  # Reset RNN hidden state
                rnn_state_actor_ref = (
                    np.zeros([1, self.actor_network.rnn_dim]),
                    np.zeros([1, self.actor_network.rnn_dim]))  # Reset RNN hidden state

                rnn_state_critic = (
                    np.zeros([1, self.actor_network.rnn_dim]),
                    np.zeros([1, self.actor_network.rnn_dim]))  # Reset RNN hidden state
                rnn_state_critic_ref = (
                    np.zeros([1, self.actor_network.rnn_dim]),
                    np.zeros([1, self.actor_network.rnn_dim]))

            self.step_number += 1

            r, internal_state, o, d, rnn_state_actor, rnn_state_actor_ref, rnn_state_critic, rnn_state_critic_ref = self.step_loop(
                o=o,
                internal_state=internal_state,
                a=a,
                rnn_state_actor=rnn_state_actor,
                rnn_state_actor_ref=rnn_state_actor_ref,
                rnn_state_critic=rnn_state_critic,
                rnn_state_critic_ref=rnn_state_critic_ref
            )

            total_episode_reward += r
            if d:
                break

        if assay["save frames"]:
            make_gif(self.frame_buffer,
                     f"{self.data_save_location}/{self.assay_configuration_id}-{assay['assay id']}.gif",
                     duration=len(self.frame_buffer) * self.learning_params['time_per_step'], true_image=True)
        self.frame_buffer = []

        self.buffer.tidy()

        if "reward assessments" in self.buffer.recordings:
            self.buffer.calculate_advantages_and_returns()
        self.buffer.save_assay_data(assay['assay id'], self.data_save_location, self.assay_configuration_id)

    def step_loop(self, o, internal_state, a, rnn_state_actor, rnn_state_actor_ref, rnn_state_critic,
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
                           # self.actor_network.scaler: np.full(o.shape, 255),
                           self.actor_network.internal_state: internal_state,
                           self.actor_network.prev_actions: np.reshape(a, (1, 2)),
                           self.actor_network.rnn_state_in: rnn_state_actor,
                           self.actor_network.rnn_state_in_ref: rnn_state_actor_ref,
                           self.actor_network.batch_size: 1,
                           self.actor_network.trainLength: 1,
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
                           # self.critic_network.scaler: np.full(o.shape, 255),
                           self.critic_network.internal_state: internal_state,
                           self.critic_network.prev_actions: np.reshape(a, (1, 2)),
                           self.critic_network.rnn_state_in: rnn_state_critic,
                           self.critic_network.rnn_state_in_ref: rnn_state_critic_ref,
                           self.critic_network.batch_size: 1,
                           self.critic_network.trainLength: 1,
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

    def get_positions(self):
        if not self.simulation.sand_grain_bodies:
            sand_grain_positions = [self.simulation.sand_grain_bodies[i].position for i, b in
                                    enumerate(self.simulation.sand_grain_bodies)]
            sand_grain_positions = [[i[0], i[1]] for i in sand_grain_positions]
        else:
            sand_grain_positions = [[10000, 10000]]

        if self.simulation.prey_bodies:
            # TODO: Note hacky fix which may want to clean up later.
            prey_positions = [prey.position for prey in self.simulation.prey_bodies]
            prey_positions = [[i[0], i[1]] for i in prey_positions]
            while True:
                if len(prey_positions) < self.last_position_dim:
                    prey_positions = np.append(prey_positions, [[10000, 10000]], axis=0)
                else:
                    break

            self.last_position_dim = len(prey_positions)

        else:
            prey_positions = np.array([[10000, 10000]])

        if self.simulation.predator_body is not None:
            predator_position = self.simulation.predator_body.position
            predator_position = np.array([predator_position[0], predator_position[1]])
        else:
            predator_position = np.array([10000, 10000])

        if self.simulation.vegetation_bodies is not None:
            vegetation_positions = [self.simulation.vegetation_bodies[i].position for i, b in
                                    enumerate(self.simulation.vegetation_bodies)]
            vegetation_positions = [[i[0], i[1]] for i in vegetation_positions]
        else:
            vegetation_positions = [[10000, 10000]]

        return sand_grain_positions, prey_positions, predator_position, vegetation_positions

    def create_output_data_storage(self, assay):
        self.output_data = {key: [] for key in assay["recordings"]}
        self.output_data["step"] = []

    def ablate_units(self, unit_indexes):
        for unit in unit_indexes:
            if unit < 256:
                output = self.sess.graph.get_tensor_by_name('mainaw:0')  # TODO: Will need to update for new network architecture.
                new_tensor = output.eval()
                new_tensor[unit] = np.array([0 for i in range(10)])
                self.sess.run(tf.assign(output, new_tensor))
            else:
                output = self.sess.graph.get_tensor_by_name('mainvw:0')
                new_tensor = output.eval()
                new_tensor[unit - 256] = np.array([0])
                self.sess.run(tf.assign(output, new_tensor))

    def save_episode_data(self):
        self.episode_summary_data = {
            "Prey Caught": self.simulation.prey_caught,
            "Predators Avoided": self.simulation.predators_avoided,
            "Sand Grains Bumped": self.simulation.sand_grains_bumped,
            "Steps Near Vegetation": self.simulation.steps_near_vegetation
        }
        with open(f"{self.data_save_location}/{self.assay_configuration_id}-summary_data.json", "w") as output_file:
            json.dump(self.episode_summary_data, output_file)
        self.episode_summary_data = None

    def save_stimuli_data(self, assay):
        with open(f"{self.data_save_location}/{self.assay_configuration_id}-{assay['assay id']}-stimuli_data.json",
                  "w") as output_file:
            json.dump(self.stimuli_data, output_file)
        self.stimuli_data = []

    def save_metadata(self):
        self.metadata["Assay Date"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        with open(f"{self.data_save_location}/{self.assay_configuration_id}.json", "w") as output_file:
            json.dump(self.metadata, output_file)

    def save_assay_results(self, assay):
        """No longer used - saves data in JSON"""
        # Saves all the information from the assays in JSON format.
        if assay["save frames"]:
            make_gif(self.frame_buffer, f"{self.data_save_location}/{assay['assay id']}.gif",
                     duration=len(self.frame_buffer) * self.learning_params['time_per_step'],
                     true_image=True)

        self.frame_buffer = []
        with open(f"{self.data_save_location}/{assay['assay id']}.json", "w") as output_file:
            json.dump(self.assay_output_data, output_file)

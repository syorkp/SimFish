import numpy as np
import tensorflow.compat.v1 as tf

from Buffers.ppo_buffer_discrete import PPOBufferDiscrete
from Tools.make_gif import make_gif

from Services.PPO.discrete_ppo import DiscretePPO
from Services.assay_service import AssayService

tf.logging.set_verbosity(tf.logging.ERROR)


def ppo_assay_target_discrete(trial, total_steps, episode_number, memory_fraction):
    service = PPOAssayServiceDiscrete(model_name=trial["Model Name"],
                                      trial_number=trial["Trial Number"],
                                      total_steps=total_steps,
                                      episode_number=episode_number,
                                      monitor_gpu=trial["monitor gpu"],
                                      using_gpu=trial["Using GPU"],
                                      memory_fraction=memory_fraction,
                                      config_name=trial["Environment Name"],
                                      realistic_bouts=trial["Realistic Bouts"],
                                      continuous_environment=trial["Continuous Actions"],

                                      assays=trial["Assays"],
                                      set_random_seed=trial["set random seed"],
                                      assay_config_name=trial["Assay Configuration Name"],
                                      )
    service.run()


class PPOAssayServiceDiscrete(AssayService, DiscretePPO):

    def __init__(self, model_name, trial_number, total_steps, episode_number, monitor_gpu, using_gpu, memory_fraction,
                 config_name, realistic_bouts, continuous_environment, assays, set_random_seed, assay_config_name):
        """
        Runs a set of assays provided by the run configuraiton.
        """
        # Set random seed
        super().__init__(model_name=model_name, trial_number=trial_number, total_steps=total_steps,
                         episode_number=episode_number, monitor_gpu=monitor_gpu, using_gpu=using_gpu,
                         memory_fraction=memory_fraction, config_name=config_name, realistic_bouts=realistic_bouts,
                         continuous_environment=continuous_environment, assays=assays, set_random_seed=set_random_seed,
                         assay_config_name=assay_config_name)

        # Buffer for saving results of assay
        self.buffer = PPOBufferDiscrete(gamma=0.99, lmbda=0.9, batch_size=self.learning_params["batch_size"],
                                        train_length=self.learning_params["trace_length"], assay=True, debug=False)

    def _run(self):
        self.create_network()  # Could also achieve by
        AssayService._run(self)

    def perform_assay(self, assay):
        self.assay_output_data_format = {key: None for key in assay["recordings"]}
        self.buffer.recordings = assay["recordings"]
        self.current_episode_max_duration = assay["duration"]

        DiscretePPO.episode_loop(self)

        if assay["save frames"]:
            make_gif(self.frame_buffer,
                     f"{self.data_save_location}/{self.assay_configuration_id}-{assay['assay id']}.gif",
                     duration=len(self.frame_buffer) * self.learning_params['time_per_step'], true_image=True)
        self.frame_buffer = []

        if "reward assessments" in self.buffer.recordings:
            self.buffer.calculate_advantages_and_returns()
        self.buffer.save_assay_data(assay['assay id'], self.data_save_location, self.assay_configuration_id)

    def step_loop(self, o, internal_state, a, rnn_state_actor, rnn_state_actor_ref, rnn_state_critic,
                  rnn_state_critic_ref):
        # TODO: Either eliminate or make useful
        return DiscretePPO._assay_step_loop(self, o, internal_state, a, rnn_state_actor, rnn_state_actor_ref,
                                              rnn_state_critic,
                                              rnn_state_critic_ref)

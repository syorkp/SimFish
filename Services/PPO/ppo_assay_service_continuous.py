import tensorflow.compat.v1 as tf

from Buffers.PPO.ppo_buffer_continuous import PPOBufferContinuous
from Buffers.PPO.ppo_buffer_continuous_multivariate import PPOBufferContinuousMultivariate

from Services.PPO.continuous_ppo import ContinuousPPO
from Services.assay_service import AssayService

tf.logging.set_verbosity(tf.logging.ERROR)


def ppo_assay_target_continuous(trial, total_steps, episode_number, memory_fraction):
    if "monitor gpu" in trial:
        monitor_gpu = trial["monitor gpu"]
    else:
        monitor_gpu = False

    if "Using GPU" in trial:
        using_gpu = trial["Using GPU"]
    else:
        using_gpu = True

    if "Continuous Actions" in trial:
        continuous_actions = trial["Continuous Actions"]
    else:
        continuous_actions = True

    if "Realistic Bouts" in trial:
        realistic_bouts = trial["Realistic Bouts"]
    else:
        realistic_bouts = True

    if "SB Emulator" in trial:
        sb_emulator = trial["SB Emulator"]
    else:
        sb_emulator = True

    if "Checkpoint" in trial:
        checkpoint = trial["Checkpoint"]
    else:
        checkpoint = None

    if "interventions" in trial:
        interventions = trial["interventions"]
    else:
        interventions = None

    if "set random seed" in trial:
        set_random_seed = trial["set random seed"]
    else:
        set_random_seed = False

    if "behavioural recordings" in trial:
        behavioural_recordings = trial["behavioural recordings"]
    else:
        behavioural_recordings = None

    if "network recordings" in trial:
        network_recordings = trial["network recordings"]
    else:
        network_recordings = None

    # Handling for when using split assay version.
    modification = None
    split_event = None
    if trial["Run Mode"] == "Split-Assay":
        if "Run Index" not in trial:
            run_version = "Original"
        else:
            modification = trial["Modification"]
            split_event = trial["Split Event"]
            run_version = trial["Run Index"]
    else:
        run_version = None

    service = PPOAssayServiceContinuous(model_name=trial["Model Name"],
                                        trial_number=trial["Trial Number"],
                                        total_steps=total_steps,
                                        episode_number=episode_number,
                                        monitor_gpu=monitor_gpu,
                                        using_gpu=using_gpu,
                                        memory_fraction=memory_fraction,
                                        config_name=trial["Environment Name"],
                                        realistic_bouts=realistic_bouts,
                                        continuous_environment=continuous_actions,
                                        assays=trial["Assays"],
                                        set_random_seed=set_random_seed,
                                        assay_config_name=trial["Assay Configuration Name"],

                                        sb_emulator=sb_emulator,
                                        checkpoint=checkpoint,
                                        behavioural_recordings=behavioural_recordings,
                                        network_recordings=network_recordings,
                                        interventions=interventions,
                                        run_version=run_version,
                                        split_event=split_event,
                                        modification=modification
                                        )
    service.run()


class PPOAssayServiceContinuous(AssayService, ContinuousPPO):

    def __init__(self, model_name, trial_number, total_steps, episode_number, monitor_gpu, using_gpu, memory_fraction,
                 config_name, realistic_bouts, continuous_environment, assays, set_random_seed,
                 assay_config_name, sb_emulator, checkpoint, behavioural_recordings, network_recordings, interventions,
                 run_version, split_event, modification):
        """
        Runs a set of assays provided by the run configuraiton.
        """
        # Set random seed
        super().__init__(model_name=model_name,
                         trial_number=trial_number,
                         total_steps=total_steps,
                         episode_number=episode_number,
                         monitor_gpu=monitor_gpu,
                         using_gpu=using_gpu,
                         memory_fraction=memory_fraction,
                         config_name=config_name,
                         realistic_bouts=realistic_bouts,
                         continuous_environment=continuous_environment,
                         assays=assays,
                         set_random_seed=set_random_seed,
                         assay_config_name=assay_config_name,
                         checkpoint=checkpoint,
                         behavioural_recordings=behavioural_recordings,
                         network_recordings=network_recordings,
                         interventions=interventions,
                         run_version=run_version,
                         split_event=split_event,
                         modification=modification
                         )

        self.multivariate = self.learning_params["multivariate"]

        # Buffer for saving results of assay
        if self.multivariate:
            self.buffer = PPOBufferContinuousMultivariate(gamma=0.99,
                                                          lmbda=0.9,
                                                          batch_size=self.learning_params["batch_size"],
                                                          train_length=self.learning_params["trace_length"],
                                                          assay=True,
                                                          debug=False,
                                                          use_dynamic_network=self.environment_params[
                                                              "use_dynamic_network"],
                                                          )
        else:
            self.buffer = PPOBufferContinuous(gamma=0.99,
                                              lmbda=0.9,
                                              batch_size=self.learning_params["batch_size"],
                                              train_length=self.learning_params["trace_length"],
                                              assay=True,
                                              debug=False,
                                              use_dynamic_network=self.environment_params["use_dynamic_network"],
                                              )

        self.ppo_version = ContinuousPPO
        self.use_rnd = self.learning_params["use_rnd"]
        self.sb_emulator = sb_emulator
        self.e = self.learning_params["startE"] / 2
        self.epsilon_greedy = self.learning_params["epsilon_greedy"]

    def run(self):
        sess = self.create_session()
        with sess as self.sess:
            self.create_network()
            self.init_states()
            AssayService._run(self)

    def step_loop(self, o, internal_state, a, rnn_state_actor, rnn_state_actor_ref, rnn_state_critic,
                  rnn_state_critic_ref):
        if self.multivariate:
            if self.sb_emulator:
                return self._assay_step_loop_multivariate2(o, internal_state, a, rnn_state_actor, rnn_state_actor_ref,
                                                           rnn_state_critic,
                                                           rnn_state_critic_ref)
            else:
                return self._assay_step_loop_multivariate(o, internal_state, a, rnn_state_actor, rnn_state_actor_ref,
                                                          rnn_state_critic,
                                                          rnn_state_critic_ref)
        else:
            return self._assay_step_loop(o, internal_state, a, rnn_state_actor, rnn_state_actor_ref, rnn_state_critic,
                                         rnn_state_critic_ref)



class BasePPO:

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        print("Base PPO Constructor called")

        # Placeholders present in service base classes (overwritten by MRO)
        self.learning_params = None
        self.environment_params = None
        self.total_steps = None
        self.simulation = None
        self.buffer = None
        self.sess = None

        self.frame_buffer = None
        self.save_frames = None

        # Network
        self.actor_network = None
        self.critic_network = None

        # To check if is assay or training
        self.assay = None

        # Allows use of same episode method
        self.current_episode_max_duration = None
        self.total_episode_reward = 0  # Total reward over episode


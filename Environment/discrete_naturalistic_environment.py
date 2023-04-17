from Environment.naturalistic_environment import NaturalisticEnvironment
from Environment.Fish.fish import Fish


class DiscreteNaturalisticEnvironment(NaturalisticEnvironment):

    def __init__(self, env_variables, using_gpu, relocate_fish=None, num_actions=10, run_version=None, split_event=None,
                 modification=None):

        super().__init__(env_variables=env_variables,
                         using_gpu=using_gpu,
                         relocate_fish=relocate_fish,
                         num_actions=num_actions,
                         run_version=run_version,
                         split_event=split_event,
                         modification=modification,
                         )

        # Create the fish class instance and add to the space.
        self.fish = Fish(board=self.board,
                         env_variables=env_variables,
                         dark_col=self.dark_col,
                         using_gpu=using_gpu
                         )

        self.space.add(self.fish.body, self.fish.mouth, self.fish.head, self.fish.tail)

        # Create walls.
        self.create_walls()
        self.reset()

        self.set_collisions()

        self.continuous_actions = False

    def simulation_step(self, action, impulse=None):
        self.fish.making_capture = False
        return super().simulation_step(action, impulse)

    def load_simulation(self, buffer, sediment, energy_state):
        self.fish.prev_action_impulse = buffer.efference_copy_buffer[-1][0][1]
        self.fish.prev_action_angle = buffer.efference_copy_buffer[-1][0][2]
        super().load_simulation(buffer, sediment, energy_state)

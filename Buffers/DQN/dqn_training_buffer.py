import random
import numpy as np


class DQNTrainingBuffer:

    def __init__(self, output_location, buffer_size=4000):
        self.buffer = []
        self.buffer_size = buffer_size
        self.output_location = output_location

        # Buffer has structure (episodes, 6):
        # 0 - Observation
        # 1 - Previous Actions
        # 2 - Reward
        # 3 - Internal state
        # 4 - Observation (t+1)
        # 5 - End multiplier (death)
        # 6 - Internal state (t+1)

    def reset(self):
        self.buffer = []
        
    def add(self, experience):
        if len(self.buffer) + 1 >= self.buffer_size:
            self.buffer[0:(1 + len(self.buffer)) - self.buffer_size] = []
        self.buffer.append(experience)

    def sample(self, batch_size, trace_length):
        sampled_episodes = random.sample(self.buffer, batch_size)
        sampled_traces = []
        for episode in sampled_episodes:
            point = np.random.randint(0, len(episode) + 1 - trace_length)
            sampled_traces.append(episode[point:point + trace_length])
        sampled_traces = np.array(sampled_traces)
        return np.reshape(sampled_traces, [batch_size * trace_length, 7])

import random
import numpy as np
import h5py
from pathlib import Path
import pickle



class ExperienceBuffer:

    def __init__(self, output_location, buffer_size=4000):
        self.buffer = []
        self.buffer_size = buffer_size
        self.output_location = output_location

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
        return np.reshape(sampled_traces, [batch_size * trace_length, 6])

    def save(self):
        # hdf5_file = h5py.File(f"{self.output_location}/training_buffer.h5", "w")
        # group = hdf5_file.create_group("buffer")
        # for i, l in enumerate(self.buffer):
        #     # if l.dtype == "O":
        #     #     l = l.astype(np.float64)
        #     group.create_dataset(str(i), data=l.astype(np.float64))
        # hdf5_file.close()
        with open(f"{self.output_location}/training_buffer.data", "wb") as file:
            pickle.dump(self.buffer, file)

    def load(self):
        # hdf5_file = h5py.File(f"{self.output_location}/training_buffer.h5", "r")
        # group = hdf5_file.get("buffer")
        # self.buffer = [l for l in group]
        # hdf5_file.close()
        with open("f{self.output_location}/training_buffer.data", "rb") as file:
            self.buffer = pickle.load(file)

    def check_saved(self):
        """Returns true if a saved experience buffer exists for the model."""
        file = Path(f"{self.output_location}/training_buffer.data")
#        file = Path(f"{self.output_location}/training_buffer.h5")
        if file.is_file():
            return True
        else:
            return False

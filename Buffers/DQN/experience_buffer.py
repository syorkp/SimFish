import random
import numpy as np
import h5py
from pathlib import Path
import pickle


class ExperienceBuffer:

    def __init__(self, output_location, buffer_size=4000, attention_sampling_bias=1):
        self.buffer = []
        self.attentions = []
        self.buffer_size = buffer_size
        self.output_location = output_location
        self.attention_bias = attention_sampling_bias

        # Buffer has structure (episodes, 6):
        # 0 - Observation
        # 1 - Previous Actions
        # 2 - Reward
        # 3 - Internal state
        # 4 - Observation (t+1)
        # 5 - End multiplier (death)
        # 6 - Internal state (t+1)
        # 7 - Attention

    def reset(self):
        self.buffer = []
        self.attentions = []
        
    def add(self, experience, total_episode_attention):
        if len(self.buffer) + 1 >= self.buffer_size:
            self.attentions[0:(1 + len(self.buffer)) - self.buffer_size] = []
            self.buffer[0:(1 + len(self.buffer)) - self.buffer_size] = []
        self.buffer.append(experience)
        self.attentions.append(total_episode_attention)

    def sample(self, batch_size, trace_length):
        attentions_array = np.array(self.attentions)
        probabilities = np.exp(attentions_array * self.attention_bias) / sum(np.exp(attentions_array * self.attention_bias)) # Calculate the probability of each episode
        sampled_inds = np.random.choice(range(len(self.buffer)), batch_size, replace=False, p=probabilities) # Sample episodes
        sampled_episodes = [self.buffer[i] for i in sampled_inds] # Get the sampled episodes
        #sampled_episodes = random.sample(self.buffer, batch_size)
        sampled_traces = []
        for episode in sampled_episodes:
            step_attention = np.reshape(episode, [len(episode), 8])[:,7].astype(int) # Get the attention values for each step
            step_prob = np.exp(step_attention * self.attention_bias*2)/np.sum(np.exp(step_attention * self.attention_bias*2)) # Calculate the probability of each step
            step_prob[:trace_length+1] = 0 # Remove the start of the episode
            step_prob[-(trace_length+1):] = 0 # Remove the end of the episode
            step_prob = step_prob/sum(step_prob) # Normalise
            point = np.random.choice(range(len(episode)), p=step_prob)
            point -= np.random.randint(0, trace_length)    
            sampled_traces.append(episode[point:point + trace_length])
        sampled_traces = np.array(sampled_traces)
        return np.reshape(sampled_traces, [batch_size * trace_length, 8])

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

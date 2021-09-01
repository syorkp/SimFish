import numpy as np
import scipy.signal as sig


class PPOBuffer:
    """Buffer for full episode for PPO training"""

    def __init__(self, gamma, lmbda, batch_size):
        self.gamma = gamma
        self.lmbda = lmbda
        self.batch_size = batch_size
        self.pointer = 0

        self.action_buffer = []
        self.observation_buffer = []
        self.reward_buffer = []
        self.internal_state_buffer = []
        self.value_buffer = []
        self.log_impulse_probability_buffer = []
        self.log_angle_probability_buffer = []
        self.advantage_buffer = []
        self.return_buffer = []

    def reset(self):
        self.action_buffer = []
        self.observation_buffer = []
        self.reward_buffer = []
        self.internal_state_buffer = []
        self.value_buffer = []
        self.log_impulse_probability_buffer = []
        self.log_angle_probability_buffer = []
        self.pointer = 0

    def add(self, observation, internal_state, action, reward, value, l_p_impulse, l_p_angle):
        self.observation_buffer.append(observation)
        self.internal_state_buffer.append(internal_state)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.value_buffer.append(value)
        self.log_impulse_probability_buffer.append(l_p_impulse)
        self.log_angle_probability_buffer.append(l_p_angle)

    def tidy(self):
        self.observation_buffer = np.array(self.observation_buffer)
        self.action_buffer = np.array(self.action_buffer)
        self.reward_buffer = np.array(self.reward_buffer)
        self.value_buffer = np.array(self.value_buffer).flatten()
        self.internal_state_buffer = np.array(self.internal_state_buffer)
        self.log_impulse_probability_buffer = np.array(self.log_impulse_probability_buffer)
        self.log_angle_probability_buffer = np.array(self.log_angle_probability_buffer)

    def get_batch(self, final_batch):
        if final_batch:
            observation_slice = self.observation_buffer[self.pointer:-1, :]
            internal_state_slice = self.internal_state_buffer[self.pointer:-1, :]
            action_slice = self.action_buffer[self.pointer + 1:-1, :]
            previous_action_slice = self.action_buffer[self.pointer:-2, :]
            reward_slice = self.reward_buffer[self.pointer:-1]
            value_slice = self.value_buffer[self.pointer:-1]
            log_impulse_probability_slice = self.log_impulse_probability_buffer[self.pointer:-1]
            log_angle_probability_slice = self.log_angle_probability_buffer[self.pointer:-1]
            advantage_slice = self.advantage_buffer[self.pointer:]
            return_slice = self.return_buffer[self.pointer:]
        else:
            observation_slice = self.observation_buffer[self.pointer:self.pointer+self.batch_size, :]
            internal_state_slice = self.internal_state_buffer[self.pointer:self.pointer+self.batch_size, :]
            action_slice = self.action_buffer[self.pointer + 1:self.pointer+self.batch_size + 1, :]
            previous_action_slice = self.action_buffer[self.pointer:self.pointer+self.batch_size, :]
            reward_slice = self.reward_buffer[self.pointer:self.pointer+self.batch_size,]
            value_slice = self.value_buffer[self.pointer:self.pointer+self.batch_size,]
            log_impulse_probability_slice = self.log_impulse_probability_buffer[self.pointer:self.pointer+self.batch_size, :]
            log_angle_probability_slice = self.log_angle_probability_buffer[self.pointer:self.pointer+self.batch_size, :]
            advantage_slice = self.advantage_buffer[self.pointer:self.pointer+self.batch_size]
            return_slice = self.return_buffer[self.pointer:self.pointer+self.batch_size]

        self.pointer += self.batch_size

        return observation_slice, internal_state_slice, action_slice, previous_action_slice, reward_slice, value_slice, \
               log_impulse_probability_slice, log_angle_probability_slice, advantage_slice, return_slice

    def discount_cumsum(self, x, discount):
        """
        magic from rllab for computing discounted cumulative sums of vectors.
        input:
            vector x,
            [x0,
             x1,
             x2]
        output:
            [x0 + discount * x1 + discount^2 * x2,
             x1 + discount * x2,
             x2]
        """
        return sig.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def calculate_advantages_and_returns(self, normalise_advantage=True):
        delta = self.reward_buffer[:-1] + self.gamma * self.value_buffer[1:] - self.value_buffer[:-1]
        advantage = self.discount_cumsum(delta, self.gamma * self.lmbda)
        if normalise_advantage:
            advantages = (advantage - advantage.mean()) / (advantage.std() + 1e-10)
        returns = self.discount_cumsum(self.reward_buffer, self.gamma)[:-1]
        self.advantage_buffer = advantage
        self.return_buffer = returns

    def compute_rewards_to_go(self):
        # NOT USED
        rewards_to_go = []
        current_discounted_reward = 0
        for i, reward in enumerate(reversed(self.reward_buffer)):
            current_discounted_reward = reward + current_discounted_reward * self.gamma
            rewards_to_go.insert(0, current_discounted_reward)
        return rewards_to_go

    def compute_advantages(self):
        """
        According to GAE
        """
        # NOT USED
        g = 0
        lmda = 0.95
        returns = []
        for i in reversed(range(1, len(self.reward_buffer))):
            delta = self.reward_buffer[i - 1] + self.gamma * self.value_buffer[i] - self.value_buffer[i - 1]
            g = delta + self.gamma * lmda * g
            returns.append(g + self.value_buffer[i - 1])
        returns.reverse()
        adv = np.array(returns, dtype=np.float32) - self.value_buffer[:-1]
        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-10)
        returns = np.array(returns, dtype=np.float32)
        return returns, adv

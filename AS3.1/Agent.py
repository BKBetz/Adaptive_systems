import tensorflow as tf
import numpy as np


class Agent:

    def __init__(self, policy, memory, discount):
        self.policy = policy
        self.memory = memory
        self.discount = discount

    def train(self):
        states, actions, rewards, n_states, terminated = self.memory.sample(32)

        # predicted action made in first states
        q_s = self.policy.model.predict(states)
        # predicted action in next states
        q_ns = self.policy.model.predict(n_states)

        # calculate q value and overwrite/replace q s
        q_t = np.copy(q_s)
        for row, action in zip(range(len(q_s[0])), actions):
            # for each row, only calculate the new action value that was chosen by the nn
            q_t[0][row][action] = rewards[row] + self.discount * np.max(q_ns[0][row]) * terminated[row]

        self.policy.model.train_on_batch(states, q_t)

        self.policy.decay()

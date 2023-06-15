import numpy as np
import random as rd


class Memory:

    def __init__(self, size):
        self.size = size
        self.deq = []

    def store(self, transition):
        if len(self.deq) > self.size - 1:
            # remove old memory
            del self.deq[0]
            self.deq.append(transition)
        else:
            self.deq.append(transition)

    def sample(self, size):
        batch = rd.sample(self.deq, size)
        states, actions, rewards, n_states, terminated = [], [], [], [], []
        for sample in batch:
            states.append(sample[0])
            actions.append(sample[1])
            rewards.append(sample[2])
            n_states.append(sample[3])
            terminated.append(sample[4])

        return np.array([states]), np.array(actions), np.array(rewards), np.array([n_states]), np.array(terminated)

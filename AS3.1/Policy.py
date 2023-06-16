import random as rd
import tensorflow as tf
from tensorflow.keras import layers
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
import numpy as np


class Policy:
    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.model = ""

    def select_action(self, state):
        rd_num = round(rd.random(), 2)
        if rd_num < self.epsilon:
            # choose random action
            action = rd.choice((0, 1, 2, 3))
            return action

        else:
            # choose action based on state
            state = np.array([state])
            output = self.model.predict(state)
            action = np.argmax(output)
            return action

    def decay(self):
        # decay epsilon over time
        if self.epsilon > 0.01:
            self.epsilon = self.epsilon * 0.99

    def create_model(self, input_dms, actions, d1, d2, lr):
        model = tf.keras.Sequential()
        model.add(layers.Dense(input_dms[0], input_shape=(32, 8), name="inputlayer"))
        model.add(layers.Dense(d1, activation="relu", name="hl1"))
        model.add(layers.Dense(d2, activation="relu", name="hl2"))
        model.add(layers.Dense(actions, name="outputlayer"))
        model.compile(optimizer=Adam(learning_rate=lr), loss=MeanSquaredError())

        self.model = model


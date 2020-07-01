from collections import deque

from utils import read_config

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam

import numpy as np
import random
from collections import deque


class DQN:
    """
    Observation_Space: Tuple of shape of state vector
    Action_Space: List of actions
    """
    def __init__(self, config_file=None, observation_space=None, action_n=None):
        print("DQN Initialised")
        if config_file:
            self.config = read_config(config_file, head="dqn")

        self.observation_space = observation_space
        self.action_space = list(range(action_n))

        self.epsilon = self.config["epsilon"]
        self.epsilon_decay = self.config["epsilon_decay"]
        self.epsilon_min = self.config["epsilon_min"]
        self.batch_size = self.config["batch_size"]
        self.gamma = self.config["gamma"]

        self.memory = deque(maxlen=self.config["deque_len"])
        self.model = self.dqn_network()
        self.target = self.dqn_network()

    def dqn_network(self):
        model = Sequential()

        # Input layer
        model.add(Dense(self.config["layers"][0], input_dim=self.observation_space[0], activation=self.config["activations"][0]))

        # Hidden Layers
        for i in range(1, len(self.config["layers"])):
            model.add(Dense(self.config["layers"][i], activation=self.config["activations"][i]))

        # Output layer
        model.add(Dense(len(self.action_space)))

        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.config["lr"]))

        return model

    def act(self, state, evaluate=False):
        if evaluate:
            return np.argmax(self.model.predict(state)[0])
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return random.choice(self.action_space)
        return np.argmax(self.model.predict(state)[0])

    def experience(self, s, a, r, ns, done):
        self.memory.append([s, a, r, ns, done])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        samples = random.sample(self.memory, self.batch_size)
        for sample in samples:
            s, a, r, ns, done = sample
            target = self.target.predict(s)
            if done:
                target[0][a] = r
            else:
                target[0][a] = r + self.gamma * (max(self.target.predict(ns)[0]))
            self.model.fit(s, target, epochs=1)

    def target_train(self):
        model_weights = self.model.get_weights()
        target_weights = self.target.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = model_weights[i]
        self.target.set_weights(self.model.get_weights())

    def save_model(self, fn):
        self.model.save(fn)

    def load_model(self, path):
        self.model = load_model(path)

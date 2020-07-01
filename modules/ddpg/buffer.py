import numpy as np
import tensorflow as tf


class Buffer:
    def __init__(self, buffer_capacity, batch_size, num_states, num_actions):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size

        self.buffer_counter = 0

        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    def record(self, observation_tuple):

        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = observation_tuple[0]
        self.action_buffer[index] = observation_tuple[1]
        self.reward_buffer[index] = observation_tuple[2]
        self.next_state_buffer[index] = observation_tuple[3]

        self.buffer_counter += 1

    def get_records(self):
        pass

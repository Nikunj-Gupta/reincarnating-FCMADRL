from utils import read_config

from keras import layers, Model, optimizers, losses
import numpy as np
import tensorflow as tf


class DQN:
    def __init__(self, config_file=None, head=None, num_states=None, num_actions=None):
        self.config = read_config(file=config_file, head=head)

        self.gamma = self.config["gamma"]
        self.epsilon = self.config["epsilon"]
        self.epsilon_min = self.config["epsilon_min"]
        self.epsilon_decay = self.config["epsilon_decay"]
        self.batch_size = self.config["batch_size"]
        self.layers = self.config["layers"]
        self.activations = self.config["activations"]

        self.num_states = num_states
        self.num_actions = num_actions

        self.model = self.create_q_model()
        self.model_target = self.create_q_model()
        self.model_target.set_weights(self.model.get_weights())
        self.optimizer = optimizers.Adam(learning_rate=self.config["learning_rate"], clipnorm=self.config["clipnorm"])

        self.state_history = []
        self.action_history = []
        self.reward_history = []
        self.next_state_history = []
        self.done_history = []
        self.max_memory_len = self.config["max_memory_len"]

        # self.update_after_actions = self.config[""]
        self.loss_fn = losses.Huber()

    def create_q_model(self):
        inputs = layers.Input(shape=(self.num_states,))

        layer = layers.Dense(self.layers[0], activation=self.activations[0])(inputs)
        for i in range(1, len(self.layers)):
            layer = layers.Dense(self.layers[i], activation=self.activations[i])(layer)

        outputs = layers.Dense(self.num_actions, activation="linear")(layer)

        model = Model(inputs=inputs, outputs=outputs)
        return model

    def add_experience(self, observation_tuple=None):
        if len(self.done_history) > self.max_memory_len:
            del self.state_history[:1]
            del self.action_history[:1]
            del self.reward_history[:1]
            del self.next_state_history[:1]
            del self.done_history[:1]
        self.state_history.append(observation_tuple[0])
        self.action_history.append(observation_tuple[1])
        self.reward_history.append(observation_tuple[2])
        self.next_state_history.append(observation_tuple[3])
        self.done_history.append(observation_tuple[4])

    def train(self):
        if len(self.done_history) >= self.batch_size:
            indices = np.random.choice(range(len(self.done_history)), size=self.batch_size)

            state_sample = np.array([self.state_history[i] for i in indices])
            action_sample = [self.action_history[i] for i in indices]
            reward_sample = [self.reward_history[i] for i in indices]
            next_state_sample = np.array([self.next_state_history[i] for i in indices])
            done_sample = tf.convert_to_tensor([float(self.done_history[i]) for i in indices])

            future_rewards = self.model_target(next_state_sample)
            updated_q_values = reward_sample + self.gamma * tf.reduce_max(future_rewards, axis=1)

            updated_q_values = updated_q_values * (1 - done_sample) - done_sample
            # masks = tf.one_hot(action_sample, self.num_actions)
            # masks = tf.cast(masks, tf.float32)

            with tf.GradientTape() as tape:
                q_values = self.model(state_sample)
                q_action = tf.reduce_sum(q_values, axis=1)
                loss = self.loss_fn(updated_q_values, q_action)

            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def act(self, state, evaluate=False):
        if (self.epsilon > np.random.random(1)[0]) and (not evaluate):
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            return np.random.choice(self.num_actions)
        state = tf.convert_to_tensor(state)
        state = tf.expand_dims(state, 0)
        action_probabilities = self.model(state, training=False)
        return tf.argmax(action_probabilities[0]).numpy()

    def update_target(self):
        self.model_target.set_weights(self.model.get_weights())


if __name__ == '__main__':
    dqn = DQN(config_file="configs/config.yaml", head="dqn", num_states=3, num_actions=1)
    for i in range(10):
        print("action: ", dqn.act(tf.random.uniform(shape=(3, )), evaluate=False))

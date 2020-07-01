from utils import read_config
from modules.ddpg.ou_action_noise import OUNoise
from modules.ddpg.buffer import Buffer

import numpy as np
import tensorflow as tf
from keras import layers, Model


class DDPG:
    def __init__(self, config_file=None, head=None, num_states=None, num_actions=None, action_lower_bound=None, action_upper_bound=None):
        self.config = {}
        if config_file and head:
            self.config = read_config(config_file, head=head)

        self.num_states = num_states
        self.num_actions = num_actions

        self.upper_bound = action_lower_bound
        self.lower_bound = action_upper_bound

        self.std_dev = self.config["std_dev"]
        self.ou_noise = OUNoise(num_actions=self.num_actions, low=self.lower_bound, high=self.upper_bound)

        self.buffer = Buffer(buffer_capacity=self.config["buffer_capacity"], batch_size=self.config["batch_size"],
                             num_states=self.num_states, num_actions=self.num_actions)

        self.gamma = self.config["gamma"]

        self.actor_model = self.get_actor()
        self.critic_model = self.get_critic()

        self.target_actor = self.get_actor()
        self.target_critic = self.get_critic()

        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())

        self.critic_lr = self.config["critic_lr"]
        self.actor_lr = self.config["actor_lr"]
        self.tau = self.config["tau"]

        self.critic_optimizer = tf.keras.optimizers.Adam(self.critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)

    def get_actor(self):
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        inputs = layers.Input(shape=(self.num_states, ))
        out = layers.Dense(self.config["input_dims"])(inputs)
        out = layers.BatchNormalization()(out)
        for i in range(len(self.config["actor"]["layers"])):
            out = layers.Dense(self.config["actor"]["layers"][i], self.config["actor"]["activations"][i])(out)
            out = layers.BatchNormalization()(out)
        outputs = layers.Dense(self.config["output_dims"], activation=self.config["output_activation"],
                               kernel_initializer=last_init)(out)

        outputs = outputs * self.upper_bound
        model = Model(inputs, outputs)

        return model

    def get_critic(self):
        state_input = layers.Input(shape=(self.num_states))
        state_out = layers.Dense(self.config["critic"]["state_out_layers"][0],
                                 activation=self.config["critic"]["state_out_activations"][0])(state_input)
        state_out = layers.BatchNormalization()(state_out)
        for i in range(1, len(self.config["critic"]["state_out_layers"])):
            state_out = layers.Dense(self.config["critic"]["state_out_layers"][i],
                                     activation=self.config["critic"]["state_out_activations"][i])(state_out)
            state_out = layers.BatchNormalization()(state_out)

        action_input = layers.Input(shape=(self.num_actions))
        action_out = layers.Dense(self.config["critic"]["action_out_layers"][0],
                                  activation=self.config["critic"]["action_out_activations"][0])(action_input)
        action_out = layers.BatchNormalization()(action_out)
        for i in range(1, len(self.config["critic"]["action_out_layers"])):
            state_out = layers.Dense(self.config["critic"]["action_out_layers"][i],
                                     activation=self.config["critic"]["action_out_activations"][i])(state_out)
            state_out = layers.BatchNormalization()(state_out)

        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(self.config["critic"]["out_layers"][0],
                           activation=self.config["critic"]["out_activations"][0])(concat)
        out = layers.BatchNormalization()(out)
        for i in range(1, len(self.config["critic"]["out_layers"])):
            out = layers.Dense(self.config["critic"]["out_layers"][i], activation=self.config["critic"]["out_activations"][i])(out)
            out = layers.BatchNormalization()(out)

        outputs = layers.Dense(1)(out)

        model = Model([state_input, action_input], outputs)

        return model

    def learn(self):
        record_range = min(self.buffer.buffer_counter, self.buffer.buffer_capacity)
        batch_indices = np.random.choice(record_range, self.buffer.batch_size)

        state_batch = tf.convert_to_tensor(self.buffer.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.buffer.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.buffer.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.buffer.next_state_buffer[batch_indices])

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch)
            y = reward_batch + self.gamma * self.target_critic([next_state_batch, target_actions])
            critic_value = self.critic_model([state_batch, action_batch])
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic_model.trainable_variables))

        with tf.GradientTape() as tape:
            actions = self.actor_model(state_batch)
            critic_value = self.critic_model([state_batch, actions])
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor_model.trainable_variables))

    def update_target(self):
        new_weights = []
        target_variables = self.target_critic.weights
        for i, variable in enumerate(self.critic_model.weights):
            new_weights.append(variable * self.tau + target_variables[i] * (1 - self.tau))

        self.target_critic.set_weights(new_weights)

        new_weights = []
        target_variables = self.target_actor.weights
        for i, variable in enumerate(self.actor_model.weights):
            new_weights.append(variable * self.tau + target_variables[i] * (1 - self.tau))

        self.target_actor.set_weights(new_weights)

    def policy(self, state):
        sampled_actions = tf.squeeze(self.actor_model(state))
        legal_action = self.ou_noise.get_action(sampled_actions)

        return [np.squeeze(legal_action)]


if __name__ == '__main__':
    ddpg = DDPG(config_file="configs/config.yaml", head="ddpg")
    print(ddpg.update_target())


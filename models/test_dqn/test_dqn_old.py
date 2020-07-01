from modules.dqn.dqn_old import DQN
from utils import read_config

import tensorflow as tf
import gym
import numpy as np
import matplotlib.pyplot as plt

config = read_config(file="configs/config.yaml", head="main")

problem = "CartPole-v1"
env = gym.make(problem)

dqn = DQN(config_file="configs/config.yaml", observation_space=env.observation_space.shape, action_n=env.action_space.n)

ep_rew_list = []
avg_rew_list = []

for episode in range(100):
    curr_state = env.reset()
    curr_state = tf.expand_dims(tf.convert_to_tensor(curr_state), 0)
    ep_rew = 0
    step = 0
    while True:
        env.render()
        step += 1

        action = dqn.act(curr_state)

        next_state, reward, done, info = env.step(action)
        next_state = tf.expand_dims(tf.convert_to_tensor(next_state), 0)

        dqn.experience(curr_state, action, reward, next_state, done)
        dqn.replay()
        dqn.target_train()

        if done or step>=200:
            break

        curr_state = next_state

    ep_rew_list.append(ep_rew)
    avg_rew = np.mean(ep_rew_list[-40:])
    avg_rew_list.append(avg_rew)
    print("Episode * {} * Avg Reward is ==> {}".format(episode, avg_rew))

plt.plot(avg_rew_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.show()

dqn.save_model("pendulum_dqn.h5")


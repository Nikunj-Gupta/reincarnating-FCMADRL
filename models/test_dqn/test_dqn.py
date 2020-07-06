from modules.dqn.dqn import DQN
from utils import read_config

import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

config = read_config(file="configs/config.yaml", head="main")

problem = "CartPole-v1"
env = gym.make(problem)

dqn = DQN(
    config_file="configs/config.yaml",
    head="dqn",
    num_states=env.observation_space.shape[0],
    num_actions=env.action_space.n
)

ep_rew_list = []
avg_rew_list = []

summary_writer = tf.summary.create_file_writer("logs/test_dqn")
for episode in range(20000):
    curr_state = env.reset()
    ep_rew = 0
    count = 0
    for step in tqdm(range(200)):
        env.render()
        action_mask = np.zeros(env.action_space.n)
        action = dqn.act(curr_state, evaluate=False)
        action_mask[action] = 1.0
        next_state, reward, done, info = env.step(action)
        dqn.add_experience(observation_tuple=(curr_state, action_mask, reward, next_state, done))

        ep_rew += reward
        curr_state = next_state
        dqn.train()

        if done:
            break

        if step % 250 == 0:
            dqn.update_target()
    ep_rew_list.append(ep_rew)
    avg_rew = np.mean(ep_rew_list[-10:])
    avg_rew_list.append(avg_rew)

    with summary_writer.as_default():
        tf.summary.scalar('episode reward', ep_rew, step=episode)
        tf.summary.scalar('running average reward (100)', avg_rew, step=episode)

    print("Episode * {} * Avg Reward is ==> {} exploration is {}\n".format(episode, avg_rew, dqn.epsilon))

    if avg_rew >= 195:
        count += 1
        if count >= 10:
            print("Task Solved ")
            break
    else:
        count = 0

plt.plot(avg_rew_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Episodic Reward")
plt.show()

dqn.model.save_weights("models/test_dqn/model_weights.h5")
dqn.model_target.save_weights("models/test_dqn/model_target_weights.h5")
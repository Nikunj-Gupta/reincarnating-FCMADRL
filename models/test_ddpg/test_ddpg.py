from modules.ddpg.ddpg import DDPG
from utils import read_config

import tensorflow as tf
import gym
import numpy as np
import matplotlib.pyplot as plt

config = read_config(file="configs/config.yaml", head="main")

problem = "Pendulum-v0"
env = gym.make(problem)

ddpg = DDPG(
    config_file="configs/config.yaml",
    head="ddpg",
    num_states=env.observation_space.shape[0],
    num_actions=env.action_space.shape[0],
    action_lower_bound=env.action_space.low[0],
    action_upper_bound=env.action_space.high[0]
    )


ep_rew_list = []
avg_rew_list = []


for episode in range(100):
    curr_state = env.reset()
    ep_rew = 0
    while True:

        env.render()

        action = ddpg.policy(tf.expand_dims(tf.convert_to_tensor(curr_state), 0))
        next_state, reward, done, info = env.step(action)

        ddpg.buffer.record(observation_tuple=(curr_state, tf.convert_to_tensor(action), reward, next_state))
        ep_rew += reward
        ddpg.learn()
        ddpg.update_target()

        if done:
            break

        curr_state = next_state

    ep_rew_list.append(ep_rew)
    avg_rew = np.mean(ep_rew_list[-40:])
    avg_rew_list.append(avg_rew)
    print("Episode * {} * Avg Reward is ==> {}".format(episode, avg_rew))

plt.plot(avg_rew_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Episodic Reward")
plt.show()




ddpg.actor_model.save_weights("pendulum_actor.h5")
ddpg.critic_model.save_weights("pendulum_critic.h5")

ddpg.target_actor.save_weights("pendulum_target_actor.h5")
ddpg.target_critic.save_weights("pendulum_target_critic.h5")

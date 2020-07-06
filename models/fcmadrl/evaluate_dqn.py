import time

from environments.multiagent_particle_envs.make_env import make_env
from utils import read_config
from modules.ddpg.ddpg import DDPG
from modules.dqn.dqn import DQN

import tensorflow as tf
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1200)])
  except RuntimeError as e:
    print(e)


def run(config_file=None, head=None):
    config = read_config(config_file, head)
    env = make_env(scenario_name="simple_spread", benchmark=False)
    dqn = DQN(
        config_file=config_file,
        head="dqn",
        num_states=12,
        num_actions=5
    )

    dqn.model.load_weights(config["weights"]["dqn_model"])
    dqn.model_target.load_weights(config["weights"]["dqn_model_target"])

    ep_rew_list = []
    avg_rew_list = []
    for episode in range(1, 10):
        curr_states = env.reset()
        ep_rew = 0

        for step in range(1, 25):
            env.render()
            time.sleep(0.1)

            dqn_actions = [dqn.act(state, evaluate=True) for state in curr_states]

            actions = [np.zeros(env.action_space[0].n) for _ in range(config["num_agents"])]
            for i in range(config["num_agents"]):
                actions[i][dqn_actions[i]] = 1.0
            dqn_actions = actions

            next_states, rewards, dones, infos = env.step(dqn_actions)
            ep_rew += np.mean(rewards)

            curr_states = next_states

        ep_rew_list.append(ep_rew)
        avg_rew = np.mean(ep_rew_list[-10:])
        avg_rew_list.append(avg_rew)

        print("Episode * {} * Reward is ==> {}".format(episode, ep_rew))
        print("Avg Reward ==> * {} * exploration is ==> {}".format(avg_rew, dqn.epsilon))
        print()


if __name__ == '__main__':
    run(config_file="configs/fcmadrl_12.yaml", head="fcmadrl")
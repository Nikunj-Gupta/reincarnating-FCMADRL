from environments.multiagent_particle_envs.make_env import make_env
from utils import read_config
from modules.ddpg.ddpg import DDPG
from modules.dqn.dqn import DQN

import tensorflow as tf
import numpy as np
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1200)])
  except RuntimeError as e:
    print(e)


def run(config_file=None, head=None):
    # load config
    config = read_config(config_file, head)

    # create cooperative navigation environment
    env = make_env(scenario_name="simple_spread", benchmark=False)

    # create dqn agent for action selection in environment
    dqn = DQN(
        config_file=config_file,
        head="dqn",
        num_states=12,
        num_actions=5
    )

    summary_writer = tf.summary.create_file_writer(config["logs_dir"])

    # initialize reward lists
    ep_rew_list = []
    avg_rew_list = []
    count = 0

    for episode in range(1, config["max_episodes"]):
        curr_states = env.reset()
        ep_rew = 0

        step = 0
        for step in range(1, config["max_steps"]):
            env.render()

            dqn_actions = [dqn.act(state, evaluate=False) for state in curr_states]

            actions = [np.zeros(env.action_space[0].n) for _ in range(config["num_agents"])]
            for i in range(config["num_agents"]):
                actions[i][dqn_actions[i]] = 1.0
            dqn_actions = actions

            next_states, rewards, dones, infos = env.step(dqn_actions)
            ep_rew += np.mean(rewards)

            for i in range(config["num_agents"]):
                dqn.add_experience(observation_tuple=(
                    curr_states[i],
                    dqn_actions[i],
                    rewards[i],
                    next_states[i],
                    dones[i]
                ))

            dqn.train()
            if step % config["dqn_update_target_network"] == 0:
                dqn.update_target()

            if all(dones):
                break

            curr_states = next_states

        ep_rew_list.append(ep_rew)
        avg_rew = np.mean(ep_rew_list[-10:])
        avg_rew_list.append(avg_rew)

        with summary_writer.as_default():
            tf.summary.scalar('episode reward', ep_rew, step=episode)
            tf.summary.scalar('running average reward (10)', avg_rew, step=episode)
            tf.summary.scalar('mean episode reward', ep_rew/step, step=episode)

        if episode % config["checkpoint"] == 0:
            if not os.path.exists(config["save_dir"]):
                os.mkdir(config["save_dir"])
            dqn.model.save_weights(os.path.join(config["save_dir"], "dqn_model_weights.h5"))
            dqn.model_target.save_weights(os.path.join(config["save_dir"], "dqn_model_target_weights.h5"))

        print("Episode * {} * Reward is ==> {}".format(episode, ep_rew))
        print("Avg Reward ==> * {} * exploration is ==> {}".format(avg_rew, dqn.epsilon))
        print()


if __name__ == '__main__':
    run(config_file="configs/fcmadrl_12.yaml", head="fcmadrl")
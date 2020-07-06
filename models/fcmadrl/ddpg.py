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
    ddpg = DDPG(
        config_file=config_file,
        head="ddpg",
        num_states=12,  # ((config["num_agents"]-1)*2 + (config["num_agents"]-1))
        num_actions=1,
        action_lower_bound=0.0,
        action_upper_bound=5.9
    )

    summary_writer = tf.summary.create_file_writer(config["logs_dir"])
    ep_rew_list = []
    avg_rew_list = []
    count = 0

    for episode in range(1, config["max_episodes"]):
        curr_states = env.reset()
        ep_rew = 0
        ep_acts = []
        step = 0

        for step in range(1, config["max_steps"]):
            env.render()

            ddpg_actions = [ddpg.policy(tf.expand_dims(tf.convert_to_tensor(ddpg_state), 0))
                            for ddpg_state in curr_states]

            ep_acts.append(ddpg_actions)

            actions = [np.zeros(env.action_space[0].n) for _ in range(config["num_agents"])]
            for i in range(config["num_agents"]):
                actions[i][int(ddpg_actions[i][0])] = 1.0
            ddpg_actions = actions

            next_states, rewards, dones, infos = env.step(ddpg_actions)
            ep_rew += np.mean(rewards)

            for i in range(config["num_agents"]):
                ddpg.buffer.record(observation_tuple=(
                    curr_states[i],
                    tf.convert_to_tensor(ddpg_actions[i][0]),
                    rewards[i],
                    next_states[i]
                ))

            ddpg.learn()
            ddpg.update_target()

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
            ddpg.actor_model.save_weights(os.path.join(config["save_dir"], "ddpg_actor_model_weights.h5"))
            ddpg.critic_model.save_weights(os.path.join(config["save_dir"], "ddpg_critic_model_weights.h5"))
            ddpg.target_actor.save_weights(os.path.join(config["save_dir"], "ddpg_actor_target_weights.h5"))
            ddpg.target_critic.save_weights(os.path.join(config["save_dir"], "ddpg_critic_target_weights.h5"))

        print("Episode * {} * Reward ==> {} Avg Reward is ==> {}".format(episode, ep_rew, avg_rew))
        print("Episode Actions ==>")
        print(ep_acts)
        print()


if __name__ == '__main__':
    run(config_file="configs/fcmadrl_8.yaml", head="fcmadrl")

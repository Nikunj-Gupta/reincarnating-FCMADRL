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
        num_states=((config["num_agents"] - 1) * 3),  # ((config["num_agents"]-1)*2 + (config["num_agents"]-1))
        num_actions=config["communication"]["message_len"],
        action_lower_bound=config["communication"]["lower_bound"],
        action_upper_bound=config["communication"]["upper_bound"]
    )

    ddpg.actor_model.load_weights(config["weights"]["ddpg_actor"])
    ddpg.critic_model.load_weights(config["weights"]["ddpg_critic"])
    ddpg.target_actor.load_weights(config["weights"]["ddpg_actor_target"])
    ddpg.target_critic.load_weights(config["weights"]["ddpg_critic_target"])

    dqn = DQN(
        config_file=config_file,
        head="dqn",
        num_states=env.observation_space[0].shape[0] + config["communication"]["message_len"],
        num_actions=env.action_space[0].n
    )

    dqn.model.load_weights(config["weights"]["dqn_model"])
    dqn.model_target.load_weights(config["weights"]["dqn_model_target"])

    ep_rew_list = []
    avg_rew_list = []

    for episode in range(1, 11):
        curr_states = env.reset()
        ep_rew = 0

        # random initialisation of dqn actions
        prev_dqn_actions = [np.random.choice(env.action_space[0].n) for _ in range(config["num_agents"])]

        # random initialisation of ddpg states
        ddpg_states = []
        for i in range(config["num_agents"]):
            ddpg_state = []
            for j in range(config["num_agents"]):
                if j != i:
                    ddpg_state.extend(env.world.agents[j].state.p_pos)
                    ddpg_state.append(prev_dqn_actions[j])
            ddpg_states.append(ddpg_state)

        step = 0
        for step in range(1, 26):
            env.render()

            ddpg_actions = [ddpg.policy(tf.expand_dims(tf.convert_to_tensor(ddpg_state), 0))
                            for ddpg_state in ddpg_states]

            dqn_states = []
            for i in range(config["num_agents"]):
                dqn_state = []
                dqn_state.extend(curr_states[i])
                if ddpg_actions[i][0].shape != ():
                    dqn_state.extend(ddpg_actions[i][0])
                else:
                    dqn_state.append(ddpg_actions[i][0])
                dqn_states.append(dqn_state)

            dqn_actions = [dqn.act(state, evaluate=True) for state in dqn_states]
            prev_dqn_actions = dqn_actions

            ddpg_next_states = []
            for i in range(len(ddpg_states)):
                ns = ddpg_states[i][:-1]
                ns.append(prev_dqn_actions[i])
                ddpg_next_states.append(ns)

            actions = [np.zeros(env.action_space[0].n) for _ in range(config["num_agents"])]
            for i in range(config["num_agents"]):
                actions[i][dqn_actions[i]] = 1.0
            dqn_actions = actions

            next_states, rewards, dones, infos = env.step(dqn_actions)
            ep_rew += np.mean(rewards)

            ddpg_next_actions = [ddpg.policy(
                    tf.expand_dims(tf.convert_to_tensor(ddpg_next_state), 0)) for ddpg_next_state in ddpg_next_states]

            dqn_next_states = []
            for i in range(len(dqn_states)):
                dqn_ns = []
                dqn_ns.extend(next_states[i])
                if ddpg_actions[i][0].shape != ():
                    dqn_ns.extend(ddpg_next_actions[i][0])
                else:
                    dqn_ns.append(ddpg_next_actions[i][0])
                dqn_next_states.append(dqn_ns)

            curr_states = next_states
            ddpg_states = ddpg_next_states

            if all(dones):
                break

        ep_rew_list.append(ep_rew)
        avg_rew = np.mean(ep_rew_list[-100:])
        avg_rew_list.append(avg_rew)

        print("Episode * {} * Reward ==> {} Avg Reward is ==> {}".format(episode, ep_rew, avg_rew))
        print("All dones * {} * exploration is ==> {}".format(all(dones), dqn.epsilon))
        print()


if __name__ == '__main__':
    run(config_file="configs/fcmadrl_7.yaml", head="fcmadrl")
from environments.multiagent_particle_envs.make_env import make_env
from utils import read_config
from modules.ddpg.ddpg import DDPG
from modules.dqn.dqn import DQN

import tensorflow as tf
import numpy as np
import os


def run(config_file=None, head=None):
    # load config
    config = read_config(config_file, head)

    # create cooperative navigation environment
    env = make_env(scenario_name="simple_spread", benchmark=False)

    # create ddpg agent for communication
    ddpg = DDPG(
        config_file=config_file,
        head="ddpg",
        num_states=((config["num_agents"] - 1) * 3),  # ((config["num_agents"]-1)*2 + (config["num_agents"]-1))
        num_actions=config["communication"]["message_len"],
        action_lower_bound=config["communication"]["lower_bound"],
        action_upper_bound=config["communication"]["upper_bound"]
    )

    # create dqn agent for action selection in environment
    dqn = DQN(
        config_file=config_file,
        head="dqn",
        num_states=env.observation_space[0].shape[0] + config["communication"]["message_len"],
        num_actions=env.action_space[0].n
    )

    summary_writer = tf.summary.create_file_writer(config["logs_dir"])

    # initialize reward lists
    ep_rew_list = []
    avg_rew_list = []
    count = 0

    for episode in range(1, config["max_episodes"]):
        curr_states = env.reset()
        # curr_states = [state.reshape(1, state.shape[0]) for state in curr_states]
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
        for step in range(1, config["max_steps"]):
            env.render()

            ddpg_actions = [ddpg.policy(tf.expand_dims(tf.convert_to_tensor(ddpg_state), 0))
                            for ddpg_state in ddpg_states]

            dqn_states = []
            for i in range(config["num_agents"]):
                dqn_state = []
                dqn_state.extend(curr_states[i])
                dqn_state.extend(ddpg_actions[i][0])
                dqn_states.append(dqn_state)

            dqn_actions = [dqn.act(state, evaluate=False) for state in dqn_states]
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
                dqn_ns.extend(ddpg_next_actions[i][0])
                dqn_next_states.append(dqn_ns)

            for i in range(config["num_agents"]):
                dqn.add_experience(observation_tuple=(
                    dqn_states[i],
                    dqn_actions[i],
                    rewards[i],
                    dqn_next_states[i],
                    dones[i]
                ))

                ddpg.buffer.record(observation_tuple=(
                    ddpg_states[i],
                    tf.convert_to_tensor(ddpg_actions[i][0]),
                    rewards[i],
                    ddpg_next_states[i]
                ))

            dqn.train()
            if step % config["dqn_update_target_network"] == 0:
                dqn.update_target()

            ddpg.learn()
            ddpg.update_target()

            curr_states = next_states
            ddpg_states = ddpg_next_states

            if all(dones):
                break

        ep_rew_list.append(ep_rew)
        avg_rew = np.mean(ep_rew_list[-100:])
        avg_rew_list.append(avg_rew)

        with summary_writer.as_default():
            tf.summary.scalar('episode reward', ep_rew, step=episode)
            tf.summary.scalar('running average reward (100)', avg_rew, step=episode)
            tf.summary.scalar('mean episode reward', ep_rew/step, step=episode)

        if episode % config["checkpoint"] == 0:
            if not os.path.exists(config["save_dir"]):
                os.mkdir(config["save_dir"])
            dqn.model.save_weights(os.path.join(config["save_dir"], "dqn_model_weights.h5"))
            dqn.model_target.save_weights(os.path.join(config["save_dir"], "dqn_model_target_weights.h5"))

            ddpg.actor_model.save_weights(os.path.join(config["save_dir"], "ddpg_actor_model_weights.h5"))
            ddpg.critic_model.save_weights(os.path.join(config["save_dir"], "ddpg_critic_model_weights.h5"))
            ddpg.target_actor.save_weights(os.path.join(config["save_dir"], "ddpg_actor_target_weights.h5"))
            ddpg.target_critic.save_weights(os.path.join(config["save_dir"], "ddpg_critic_target_weights.h5"))

        print("Episode * {} * Avg Reward is ==> {}".format(episode, avg_rew))
        print("All dones * {} * exploration is ==> {}".format(all(dones), dqn.epsilon))
        print()

        if avg_rew > -0.3:
            count += 1
            if count == 10:
                print("Task Learnt!")
                break
        else:
            count = 0


if __name__ == '__main__':
    run(config_file="configs/fcmadrl.yaml", head="fcmadrl")
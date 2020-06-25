import pickle

from modules.dqn.dqn import DQN
from utils import read_config
from environments.multiagent_particle_envs.make_env import make_env

import argparse
import numpy as np
import os
from tqdm import tqdm, trange


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments")

    # configuration file
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="path of the config file")
    parser.add_argument("--config-head", type=str, default=None, help="desired first level key in config")

    return parser.parse_args()


def train(config_file, head=None):
    config = read_config(config_file, head)
    env = make_env(scenario_name="simple_spread", benchmark=False)

    # new observation space shape for shared parameters model
    obs_shape = tuple((env.observation_space[0].shape[0]+1, ))
    agent = DQN(config_file=config_file, observation_space=obs_shape,
                action_n=env.action_space[0].n)

    current_states = env.reset()
    current_states = [np.concatenate(([i], current_states[i])) for i in range(config["num_agents"])]
    current_states = [state.reshape(1, state.shape[0]) for state in current_states]

    episode_step = 0
    episode_rewards = [0.0]

    print("\n\n")
    while True:
        if config["display"]:
            env.render()
        actions = [np.zeros(env.action_space[0].n) for _ in range(config["num_agents"])]
        for i in range(config["num_agents"]):
            actions[i][agent.act(current_states[i])] = 1.0
        next_states, rewards, dones, infos = env.step(actions)
        next_states = [np.concatenate(([i], next_states[i])) for i in range(config["num_agents"])]
        next_states = [ns.reshape(1, ns.shape[0]) for ns in next_states]

        episode_step += 1
        print("Episode: {}, Step: {}, Current Episode rewards: {:.2f}, Total Episode rewards in last 100 episodes: {:.2f}"
              .format(len(episode_rewards), episode_step, episode_rewards[-1], sum(episode_rewards[-100:])))

        if not config["evaluate"]:
            for i in range(config["num_agents"]):
                agent.experience(current_states[i], np.where(actions[i] == 1)[0][0], rewards[i], next_states[i], dones[i])

        terminal = episode_step >= config["episode_len"]

        current_states = next_states

        for r in rewards:
            episode_rewards[-1] += r

        if all(dones) or terminal:
            print("\nAll dones: ", all(dones), "Terminal: ", episode_step >= config["episode_len"])
            print('\n\n**** Completed {} episodes ****'.format(len(episode_rewards)))
            print("Episode {} reward: {}\n\n".format(len(episode_rewards), episode_rewards[-1]))
            current_states = env.reset()
            current_states = [np.concatenate(([i], current_states[i])) for i in range(config["num_agents"])]
            current_states = [state.reshape(1, state.shape[0]) for state in current_states]
            episode_step = 0
            episode_rewards.append(0)

        if not config["evaluate"]:
            agent.replay()
            agent.target_train()

        if (all(dones) or terminal) and (len(episode_rewards) % config["save_model_after"] == 0):
            print("Saving model!")
            agent.save_model(os.path.join(config["save_dir"], "checkpoint-"+str(len(episode_rewards))+"-episodes.model"))
            rew_file_name = os.path.join(config["plot_dir"], 'rewards-'+str(len(episode_rewards))+'-episode.pkl')
            if not os.path.exists(config["plot_dir"]):
                os.makedirs(config["plot_dir"])
            with open(rew_file_name, 'wb') as fp:
                pickle.dump(episode_rewards, fp)

        if len(episode_rewards) >= config["total_episodes"]:
            break


if __name__ == '__main__':
    args = parse_args()
    print(args.config, args.config_head)

    train(config_file=args.config, head=args.config_head)


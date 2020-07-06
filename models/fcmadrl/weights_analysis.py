from modules.dqn.dqn import DQN
from utils import read_config

from environments.multiagent_particle_envs.make_env import make_env

config_file = "configs/fcmadrl.yaml"
config = read_config(file=config_file)

env = make_env(scenario_name="simple_spread", benchmark=False)

dqn = DQN(
    config_file=config_file,
    head="dqn",
    num_states=env.observation_space[0].shape[0] + config["fcmadrl"]["communication"]["message_len"],
    num_actions=env.action_space[0].n
)

dqn.model.load_weights(config["weights"]["dqn_model"])
dqn.model_target.load_weights(config["weights"]["dqn_model_target"])

print(dqn.model.weights)
import pickle as pkl
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser("Reinforcement Learning experiments")
parser.add_argument("--episodes", type=str, default=200, help="corresponding to the pickle file")

args = parser.parse_args()

file_name = "save-dir/dqn-shared-params/plots/rewards-" + str(args.episodes) + "-episode.pkl"
data = pkl.load(open(file_name,  'rb'))

plt.plot(data)
plt.show()

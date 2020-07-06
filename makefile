dqn-shared-params:
	clear
	PYTHONPATH=$(PWD) python models/dqn_shared_params/dqn_shared_parameters.py --config configs/experiment-1.yaml --config-head main
plot:
	clear
	PYTHONPATH=$(PWD) python models/dqn_shared_params/plot_rewards.py --episodes 1400

ddpg:
	clear
	PYTHONPATH=$(PWD) python modules/ddpg_new/ddpg.py

dqn:
	clear
	PYTHONPATH=$(PWD) python models/test_dqn/test_dqn.py

fcmadrl:
	clear
	PYTHONPATH=$(PWD) python models/fcmadrl/dqn.py

evaluate:
	clear
	PYTHONPATH=$(PWD) python models/fcmadrl/evaluate_dqn.py

ddpg_nav:
	clear
	PYTHONPATH=$(PWD) python models/fcmadrl/ddpg.py

dqn_nav:
	clear
	PYTHONPATH=$(PWD) python models/fcmadrl/dqn.py

tf_agents:
	clear
	PYTHONPATH=$(PWD) python models/tf_agents/dqn_cartpole.py

rllib:
	clear
	PYTHONPATH=$(PWD) python models/rllib/dqn.py

keras-rl:
	clear
	PYTHONPATH=$(PWD) python models/keras-rl/ddpg.py
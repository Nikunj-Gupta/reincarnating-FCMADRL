dqn-shared-params:
	clear
	PYTHONPATH=$(PWD) python models/dqn_shared_params/dqn_shared_parameters.py --config configs/experiment-1.yaml --config-head main
plot:
	clear
	PYTHONPATH=$(PWD) python models/dqn_shared_params/plot_rewards.py --episodes 1400

ddpg:
	clear
	PYTHONPATH=$(PWD) python models/test_ddpg/test_ddpg.py

dqn:
	clear
	PYTHONPATH=$(PWD) python models/test_dqn/test_dqn.py

fcmadrl:
	clear
	PYTHONPATH=$(PWD) python models/fcmadrl/fcmadrl.py
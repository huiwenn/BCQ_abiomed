import argparse
import copy
import importlib
import json
import os

import numpy as np
import torch
import pickle
import discrete_BCQ
import DQN
import utils

# Trains BCQ offline
def train_BCQ(replay_buffer, is_atari, num_actions, state_dim, device, args, parameters):
	# For saving files
	#setting = f"{args.env}_{args.seed}"
	buffer_name = f"{args.buffer_name}"
	setting = 'abiomed_long'
	# Initialize and load policy
	
	policy = discrete_BCQ.discrete_BCQ(
		is_atari,
		num_actions,
		state_dim,
		device,
		args.BCQ_threshold,
		parameters["discount"],
		parameters["optimizer"],
		parameters["optimizer_parameters"],
		parameters["polyak_target_update"],
		parameters["target_update_freq"],
		parameters["tau"],
		parameters["initial_eps"],
		parameters["end_eps"],
		parameters["eps_decay_period"],
		parameters["eval_eps"]
	)

	# Load replay buffer	
	#replay_buffer.load(f"./buffers/{buffer_name}")
	
	evaluations = []
	episode_num = 0
	done = True 
	training_iters = 0
	
	while training_iters < args.max_timesteps: 

		losses = []
		for _ in range(int(parameters["eval_freq"])):
			Q_loss = policy.train(replay_buffer)
			losses.append(Q_loss)
			
		#evaluations.append(eval_policy(policy, args.env, args.seed))
		np.save(f"./results/{buffer_name}/BCQ_{setting}_{training_iters}", losses)

		training_iters += int(parameters["eval_freq"])
		print(f"Training iterations: {training_iters}, loss: {np.mean(losses)}")
	
		with open(f'models/{buffer_name}/bcq_{setting}_{training_iters}_{np.mean(losses):.2f}.pkl', 'wb') as handle:
			pickle.dump(policy, handle, protocol=pickle.HIGHEST_PROTOCOL)
	
# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
	eval_env, _, _, _ = utils.make_env(env_name, atari_preprocessing)
	eval_env.seed(seed + 100)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			action = policy.select_action(np.array(state), eval=True)
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward


if __name__ == "__main__":



	regular_parameters = {
		# Exploration
		"start_timesteps": 1e3,
		"initial_eps": 0.1,
		"end_eps": 0.1,
		"eps_decay_period": 1,
		# Evaluation
		"eval_freq": 5e3, #5e3,
		"eval_eps": 0,
		# Learning
		"discount": 0.99,
		"buffer_size": 1e6,
		"batch_size": 64,
		"optimizer": "Adam",
		"optimizer_parameters": {
			"lr": 3e-4
		},
		"train_freq": 1,
		"polyak_target_update": True,
		"target_update_freq": 1,
		"tau": 0.005
	}

	# Load parameters
	parser = argparse.ArgumentParser()
	parser.add_argument("--env", default="PongNoFrameskip-v0")	 # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)			 # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--buffer_name", default="Default")		# Prepends name to filename
	parser.add_argument("--max_timesteps", default=2e5, type=int)  # Max time steps to run environment or train for 1e6
	parser.add_argument("--BCQ_threshold", default=0.3, type=float)# Threshold hyper-parameter for BCQ
	parser.add_argument("--low_noise_p", default=0.2, type=float)  # Probability of a low noise episode when generating buffer
	parser.add_argument("--rand_action_p", default=0.2, type=float)# Probability of taking a random action when generating buffer, during non-low noise episode
	#parser.add_argument("--train_behavioral", action="store_true") # If true, train behavioral policy
	#parser.add_argument("--generate_buffer", action="store_true")  # If true, generate buffer
	args = parser.parse_args()
	
	print("---------------------------------------")	
	# if args.train_behavioral:
	# 	print(f"Setting: Training behavioral, Env: {args.env}, Seed: {args.seed}")
	# elif args.generate_buffer:
	# 	print(f"Setting: Generating buffer, Env: {args.env}, Seed: {args.seed}")
	# else:
	#	print(f"Setting: Training BCQ, Env: {args.env}, Seed: {args.seed}")
	print(f"Setting: Training BCQ")
	print("---------------------------------------")

	# if args.train_behavioral and args.generate_buffer:
	# 	print("Train_behavioral and generate_buffer cannot both be true.")
	# 	exit()

	if not os.path.exists(f"./results/{args.buffer_name}"):
		os.makedirs(f"./results/{args.buffer_name}")

	if not os.path.exists(f"./models/{args.buffer_name}"):
		os.makedirs(f"./models/{args.buffer_name}")

	# if not os.path.exists("./buffers"):
	# 	os.makedirs("./buffers")

	# Make env and determine properties
	#env, is_atari, state_dim, num_actions = utils.make_env(args.env, atari_preprocessing)
	is_atari = False
	num_actions = 9
	state_dim = 5*int(args.buffer_name) #7*int(args.buffer_name) + 6 # 76 #6
	parameters = atari_parameters if is_atari else regular_parameters

	# Set seeds
	#env.seed(args.seed)
	#env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	device = torch.device("cpu")

	# Initialize buffer
	# replay_buffer = utils.ReplayBuffer(state_dim, is_atari, atari_preprocessing, parameters["batch_size"], parameters["buffer_size"], device)

	# if args.train_behavioral or args.generate_buffer:
	# 	interact_with_environment(env, replay_buffer, is_atari, num_actions, state_dim, device, args, parameters)
	# else:
	# 	train_BCQ(env, replay_buffer, is_atari, num_actions, state_dim, device, args, parameters)

	hrpci_buffer = utils.StandardBuffer(state_dim, 24, 1e5, device) #state_dim, batch_size, buffer_size, device
	hrpci_buffer.load(f"../../Abiomed_Forecasting/src/data/buffers/hrpci_{args.buffer_name}_discrete_train") # need to be run from this directory
	# problem_index = [11206, 11305, 15257, 11207, 11306, 15258, 23040]
	# hrpci_buffer.state = np.delete(  hrpci_buffer.state, problem_index,  axis=0)
	# hrpci_buffer.reward = np.delete(  hrpci_buffer.reward, problem_index,  axis=0)
	# hrpci_buffer.action = np.delete(  hrpci_buffer.action, problem_index,  axis=0)
	# hrpci_buffer.next_state = np.delete(  hrpci_buffer.next_state, problem_index,  axis=0)
	# hrpci_buffer.not_done = np.delete(  hrpci_buffer.not_done, problem_index,  axis=0)
	# hrpci_buffer.crt_size = hrpci_buffer.crt_size - len(problem_index)

	print(np.mean(hrpci_buffer.reward))
	assert not np.any(np.isnan(hrpci_buffer.state))
	assert not np.any(np.isnan(hrpci_buffer.reward))
	assert not np.any(np.isnan(hrpci_buffer.action))
	assert not np.any(np.isnan(hrpci_buffer.next_state))

	train_BCQ(hrpci_buffer, is_atari, num_actions, state_dim, device, args, parameters)


import argparse
import gym
import numpy as np
import os
import torch

import BCQ
import DDPG
import utils
import pickle


# Trains BCQ offline
def train_BCQ(replay_buffer, state_dim, action_dim, max_action, device, args):
	# For saving files
	# setting = f"{args.env}_{args.seed}"
	# buffer_name = f"{args.buffer_name}_{setting}"

	# Initialize policy
	policy = BCQ.BCQ(state_dim, action_dim, max_action, device, args.discount, args.tau, args.lmbda, args.phi)

	# # Load buffer
	# replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)
	# replay_buffer.load(f"./buffers/{buffer_name}")
	
	evaluations = []
	episode_num = 0
	done = True 
	training_iters = 0
	eval_freq = 1000
	buffer_name = f"{args.buffer_name}"

	while training_iters < args.max_timesteps: 
		#iterations=int(args.eval_freq),
		pol_vals = policy.train(replay_buffer, iterations=int(eval_freq), batch_size=args.batch_size)

		#evaluations.append(eval_policy(policy, args.env, args.seed))
		#np.save(f"./results/BCQ_{setting}", evaluations)

		training_iters += eval_freq
		print(f"Training iterations: {training_iters}")
		#_{np.mean(losses):.2f}
		with open(f'models/bcq_{buffer_name}_{training_iters}_apr15.pkl', 'wb') as handle:
			pickle.dump(policy, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + 100)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			action = policy.select_action(np.array(state))
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--env", default="Hopper-v3")               # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--buffer_name", default="full")          # Prepends name to filename
	parser.add_argument("--max_timesteps", default=2e4, type=int)   # Max time steps to run environment or train for (this defines buffer size)
	parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used before training behavioral
	parser.add_argument("--rand_action_p", default=0.0, type=float) # 0.00 Probability of selecting random action during batch generation
	parser.add_argument("--gaussian_std", default=0.3, type=float)  # Std of Gaussian exploration noise (Set to 0.1 if DDPG trains poorly)
	parser.add_argument("--batch_size", default=100, type=int)      # Mini batch size for networks
	parser.add_argument("--discount", default=0.99)                 # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--lmbda", default=0.75)                    # Weighting for clipped double Q-learning in BCQ
	parser.add_argument("--phi", default=0.05)                      # 0.05 Max perturbation hyper-parameter for BCQ
	#parser.add_argument("--train_behavioral", action="store_true")  # If true, train behavioral (DDPG)
	#parser.add_argument("--generate_buffer", action="store_true")   # If true, generate buffer
	args = parser.parse_args()

	print("---------------------------------------")	
	# if args.train_behavioral:
	# 	print(f"Setting: Training behavioral, Env: {args.env}, Seed: {args.seed}")
	# elif args.generate_buffer:
	# 	print(f"Setting: Generating buffer, Env: {args.env}, Seed: {args.seed}")
	# else:
	# 	print(f"Setting: Training BCQ, Env: {args.env}, Seed: {args.seed}")
	print(f"Setting: Training BCQ")

	print("---------------------------------------")

	# if args.train_behavioral and args.generate_buffer:
	# 	print("Train_behavioral and generate_buffer cannot both be true.")
	# 	exit()

	# if not os.path.exists("./results"):
	# 	os.makedirs("./results")

	# if not os.path.exists("./models"):
	# 	os.makedirs("./models")ß

	# if not os.path.exists("./buffers"):
	# 	os.makedirs("./buffers")

	# env = gym.make(args.env)

	# env.seed(args.seed)
	# env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim = 450 #env.observation_space.shape[0]
	action_dim = 90 #env.action_space.shape[0] 
	max_action = 4 #float(env.action_space.high[0])

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	hrpci_buffer = utils.ReplayBuffer(state_dim,action_dim,  device) #state_dim, batch_size, buffer_size, device
	hrpci_buffer.load(f"../../Abiomed_Forecasting/src/data/buffers/hrpci_{args.buffer_name}_train") # need to be run from this directory

	train_BCQ(hrpci_buffer, state_dim, action_dim, max_action, device, args)

import gym
from baselines import deepq

def callback(lcl, _glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved

env = gym.make("RLEnv-v0", nS = 20, nA = 20, gamma = 0.8)
agent = deepq.learn(
	env,
	network='mlp',
	lr=1e-4,
	total_timesteps=1000000,
	buffer_size=50000,
	exploration_fraction=0.1,
	exploration_final_eps=0.02,
	print_freq=10,
	gamma = 1,
	callback=callback,
	load_path = None#'agent-mle-v0-' + str(_ - 1) + '.pkl' if _ > 0 else None
)



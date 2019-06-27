import gym
import numpy as np
import tensorflow as tf
from baselines import deepq

def callback(lcl, _glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved

env = gym.make("RLEnv-v0", nS = 20, nA = 20, gamma = 0.8)
agent = deepq.learn(
	env,
	network='mlp',
	lr=1e-3,
	total_timesteps=2000,
	buffer_size=50000,
	exploration_fraction=0.1,
	exploration_final_eps=0.02,
	print_freq=30,
	gamma = 1,
	callback=callback,
	load_path = None#'agent-mle-v0-' + str(_ - 1) + '.pkl' if _ > 0 else None
)
agent.save("save-test.txt")

print("\n\n== test 0 ==\n")
policy = [0] * 20
for s in range(20):
	policy[s] = agent(np.array([s]))[0]
print(env.calc_rewards(policy, 1000))

for c in range(100):
	
	tf.get_variable_scope().reuse_variables()
	agent = deepq.learn(
		env,
		network='mlp',
		lr=1e-3,
		total_timesteps=2000,
		buffer_size=50000,
		exploration_fraction=0.1,
		exploration_final_eps=0.0,
		print_freq=30,
		gamma = 1,
		callback=callback,
		load_path = 'save-test.txt'
	)
	agent.save("save-test.txt")
	print("\n\n== test ", c + 1, " ==\n")
	policy = [0] * 20
	for s in range(20):
		policy[s] = agent(np.array([s]))[0]
	print(env.calc_rewards(policy, 1000))
	print("\n\n\n\n")

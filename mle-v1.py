import gym
from baselines import deepq
import virtual
import numpy as np
import tensorflow as tf

class Argument():
	def __init__(self, env):
		self.model = 'MLP'
		self.n_states = env.nS
		self.n_actions = env.nA
		self.d_hidden = 100
		self.n_layers = 0
		self.dropout = 0.5
		self.lr = 3e-2
		self.n_epochs = 1000
		self.batch_size = 40000
		self.log_every = 10
		self.save_every = 0
		self.eval_every = 10
       
def callback(lcl, _glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved

def get_action(env, obs, agent):
	if agent == None:
		#print(env.action_space.sample())
		return env.action_space.sample()
	else:
		#print('\n\ntype of agent is \n', type(agent), '\n\n')
		#print('observation is ', obs)
		#obs = [obs]
		#print(np.shape(np.array([obs])))
		return agent(np.array([obs]))[0]
		#assert False
		
if __name__ == '__main__':	
	env = gym.make('RLEnv-v0', nS = 5, nA = 5, gamma = 0.9, terminal_steps = 200)
	agent = None
	nsamples = 2
	ntrials = 2000
	
	args = Argument(env)
	freq = [[dict() for _ in range(env.nA)] for _ in range(env.nS)]
	for s in range(env.nS):
		for a in range(env.nA):
			for ns in range(env.nS):
				freq[s][a][ns] = 0
	for __ in range(ntrials):
		obs = env.reset()
		for ___ in range(nsamples):
			action = get_action(env, obs, agent)
			nobs, reward, stop, info = env.step(action)
			freq[obs][action][nobs] += 1
	
	for s in range(env.nS):
		for a in range(env.nA):
			tot = 0
			for ns in range(env.nS):
				tot += freq[s][a][ns] 
			for ns in range(env.nS):
				freq[s][a][ns] /= tot
			print("(", s, a, ") -> ", freq[s][a])

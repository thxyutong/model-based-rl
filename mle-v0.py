import gym
from baselines import deepq
import virtual
import numpy as np
import tensorflow as tf
import random
import torch

class Argument():
	def __init__(self, env):
		self.model = 'MLP'
		self.n_states = env.nS
		self.n_actions = env.nA
		self.d_hidden = 100
		self.n_layers = 1
		self.dropout = 0.5
		self.lr = 3e-4
		self.n_epochs = 2
		self.batch_size = 4000
		self.log_every = 10
		self.save_every = 0
		self.eval_every = 10
       
def callback(lcl, _glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved

def get_action(env, obs, agent):
	if agent == None:
		return env.action_space.sample()
	else:
		#print('\n\ntype of agent is \n', type(agent), '\n\n')
		#print('observation is ', obs)
		#obs = [obs]
		#print(np.shape(np.array([obs])))
		return agent(np.array([obs]))[0]
		#assert False
		
def test_agent(env, agent):
	policy = []
	for s in range(env.nS):
		policy.append(agent(np.array([s]))[0])
	env.calc_rewards(policy)
		
if __name__ == '__main__':	
	tf.set_random_seed(0)
	random.seed(233)	
	np.random.seed(1)
	torch.manual_seed(123)
	torch.cuda.manual_seed(12312)
	
	env = gym.make('RLEnv-v0', nS = 10, nA = 10, gamma = 0.9, terminal_steps = 200)
	agent = None
	nsamples = 20
	ntrials = 2000
	
	args = Argument(env)
	model = virtual.VirtualEnv(args, env)
	for _ in range(10):     #times of cycles
		print("\n\n===  Cycle ", _, "\n\n")
		states = []
		actions = []
		_states = []
		for __ in range(ntrials):
			obs = env.reset()
			for ___ in range(nsamples):
				action = get_action(env, obs, agent)
				nobs, reward, stop, info = env.step(action)
				states.append(obs)
				actions.append(action)
				_states.append(nobs)
				if stop:
					break
				obs = nobs
		data = [states, actions, _states]
		model.add_data(data)
		model.train()

		my_env = gym.make('RLImitateEnv-v0', real_env = env, model = model)
		my_env.reset()
		
		if _ > 0:
			tf.get_variable_scope().reuse_variables()
			
		agent = deepq.learn(
			my_env,
			network='mlp',
			lr=1e-3,
			total_timesteps=4000,
			buffer_size=50000,
			exploration_fraction=0.1,
			exploration_final_eps=0.02,
			train_freq = 10,
			print_freq=10,
			callback=callback,
			load_path = None#'mle-models/agent-mle-v0-' + str(_ - 1) + '.pkl' if _ > 0 else None
		)
		#agent.save('mle-models/agent-mle-v0-' + str(_) + '.pkl')
		
		test_agent(env, agent)
		print("End Cycle")

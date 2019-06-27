import gym
from baselines import deepq
import virtual
import numpy as np
import tensorflow as tf

class Argument():
	def __init__(self, env):
		self.model = 'MLP'
		self.n_states = 20
		self.n_actions = 20
		self.d_hidden = 500
		self.n_layers = 1
		self.dropout = 0.5
		self.lr = 1e-4
		self.n_epochs = 1000
		self.batch_size = 32
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
		
if __name__ == '__main__':	
	env = gym.make('RLEnv-v0', nS = 20, nA = 20, gamma = 0.9, terminal_steps = 200)
	agent = None
	nsamples = 200
	ntrials = 200
	
	args = Argument(env)
	model = virtual.VirtualEnv(args)
	for _ in range(10):     #times of cycles
		print("\n\n===  Cycle ", _, "\n\n")
		states = []
		actions = []
		_states = []
		for __ in range(ntrials):
			obs = env.reset()
			for ___ in range(nsamples):
				#print(_, __, ___)
				action = get_action(env, obs, agent)
				#if _ > 0:
				#	print('pair is (', obs, ', ', action, ') ', type(action))
				#	print(np.shape(action))
				#	print(action.tolist()[0])
				nobs, reward, stop, info = env.step(action)
				states.append(obs)
				actions.append(action)
				_states.append(nobs)
				if stop:
					break
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
			load_path = 'mle-models/agent-mle-v0-' + str(_ - 1) + '.pkl' if _ > 0 else None
		)
		#print(_, 'agent-mle-v0-' + str(_) + '.pkl')
		#assert(_==0)
		agent.save('mle-models/agent-mle-v0-' + str(_) + '.pkl')
		
		print("End Cycle")

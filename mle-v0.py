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
        self.n_layers = 1
        self.dropout = 0.5
        self.lr = 1e-4
        self.n_epochs = 2
        self.batch_size = 2
        self.log_every = 10
        self.save_every = 0
       
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
	env = gym.make('RLEnv-v0', nS = 20, nA = 20, gamma = 0.8)
	agent = None
	nsamples = 100
	ntrials = 10
	for _ in range(1):     #times of cycles
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
					
		args = Argument(env)
		data = [states, actions, _states]
		model = virtual.VirtualEnv(args)
		model.add_data(data)
		model.train()

		my_env = gym.make('RLImitateEnv-v0', real_env = env, model = model)
		my_env.reset()
		
		if _ > 0:
			#list1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
			tf.get_variable_scope().reuse_variables()
			#list2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
			#print(list1, '\n')
			#print(list2)
			
		print("\n\n==================\n\n")
		agent = deepq.learn(
			my_env,
			network='mlp',
			lr=1e-3,
			total_timesteps=100,
			buffer_size=50000,
			exploration_fraction=0.1,
			exploration_final_eps=0.02,
			print_freq=10,
			callback=callback,
			load_path = None#'agent-mle-v0-' + str(_ - 1) + '.pkl' if _ > 0 else None
		)
		#print(_, 'agent-mle-v0-' + str(_) + '.pkl')
		#assert(_==0)
		agent.save('agent-mle-v0-' + str(_) + '.pkl')
		print('\n\n\n cycle ', _)

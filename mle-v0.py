import gym
from baselines import deepq
import virtual

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
		return agent(obs)

env = gym.make('RLEnv-v0', nS = 20, nA = 20, gamma = 0.8)
agent = None
nsamples = 100
ntrials = 10
for _ in range(2):     #times of cycles
	states = []
	actions = []
	_states = []
	for __ in range(ntrials):
		obs = env.reset()
		for ___ in range(nsamples):
			print(_, __, ___)
			action = get_action(env, obs, agent)
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
	
	agent = deepq.learn(
		my_env,
		network='mlp',
		lr=1e-3,
		total_timesteps=10,#00000,
		buffer_size=50000,
		exploration_fraction=0.1,
		exploration_final_eps=0.02,
		print_freq=10,
		callback=callback,
		load_path = 'agent-mle-v0-' + str(_ - 1) + '.pkl' if _ > 0 else None
	)
	#print(_, 'agent-mle-v0-' + str(_) + '.pkl')
	agent.save('agent-mle-v0-' + str(_) + '.pkl')
	print('lfakjsdflaks=================adslkjfasdjldjs')

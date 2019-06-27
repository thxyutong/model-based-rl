import gym
import baselines
import numpy as np
import random as rd
import h5py

env = gym.make("RLEnv-v0", nS = 20, nA = 20, gamma = 0.9, terminal_steps = 1000)

obs = []
act = []
rew = []
lens = []
for _ in range(5):
	s = env.reset()
	nobs = []
	nact = []
	nrew = []
	for __ in range(100):
		a = env.action_space.sample()
		ns, _, _, _ = env.step(a)
		nobs.append(s * env.nA + a)
		nact.append(ns)
		nrew.append(0)
	rew.append(nrew)
	#nobs.append(s * env.nA + env.action_space.sample())
	obs.append(nobs)
	act.append(nact)
	lens.append(100)

obs = np.array(obs)
act = np.array(act)
rew = np.array(rew)
traj = {"obs_B_T_Do": obs, "a_B_T_Do": act, "r_B_T": rew, "len_B": lens}
print(traj)

hf = h5py.File("/home/pascalprimer/pytorch-a2c-ppo-acktr-gail/gail_experts/GAILEnv-v0.h5", "w")
hf.create_dataset("obs_B_T_Do", data = obs)
hf.create_dataset("a_B_T_Do", data = act)
hf.create_dataset("r_B_T", data = rew)
hf.create_dataset("len_B", data = lens)
#np.save("../test/test-data", traj)

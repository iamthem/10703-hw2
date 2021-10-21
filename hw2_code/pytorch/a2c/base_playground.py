# %%
from importlib import reload
import torch
import numpy as np
import gym
import a2c
import net
batch = 3 
gamma = 0.99
test_episodes = 100
device = 'cpu'
lr = 5e-4
env = gym.make('CartPole-v0')
nA = env.action_space.n
nS = env.observation_space.shape[0]
test_episodes = 10

# %%
reload(net)
reload(a2c)
Reinforce_net = a2c.Reinforce(nA, device, lr, nS, baseline = True, baseline_lr = 5e-4)

# %%
for m in range(1500):
    Reinforce_net.train(env, batch, gamma=gamma)

    if m % 100 == 0:

        G = np.zeros(test_episodes)
        Loss = np.zeros(test_episodes)  

        for k in range(test_episodes):
            g = Reinforce_net.evaluate_policy(env, batch) 
            G[k] = g 

        reward_mean = G.mean()
        reward_sd = G.std()
        print("Reward sd == ", str(reward_sd), "\t Reward \\mu == ", str(reward_mean.mean()))


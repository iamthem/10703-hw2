# %%
from importlib import reload
import torch
import numpy as np
import gym
import a2c
import net
batch = 1 
gamma = 0.99
test_episodes = 100
device = 'cpu'
lr = 5e-4
env = gym.make('CartPole-v0')
nA = env.action_space.n
nS = env.observation_space.shape[0]
test_episodes = 10
critic_lr = 1e-3
baseline = True
n = 10

# %%
reload(net)
reload(a2c)
A2C_net = a2c.A2C(nA, device, lr, nS, critic_lr, baseline)

# %%
for m in range(1000):
    A2C_net.train(env, batch, gamma=gamma, n = n)

    if m % 100 == 0:

        G = np.zeros(test_episodes)

        for k in range(test_episodes):

            rewards, probs, base_out = A2C_net.evaluate_policy(env, batch, n) 
            G[k] = rewards 

        reward_mean = G.mean()
        reward_sd = G.std()

        print("rewards \\mu = ", reward_mean, " rewards sd = ", reward_sd, "\nPolicy_outputs", )


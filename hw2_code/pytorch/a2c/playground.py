# %%
import torch
import numpy as np
import gym
from a2c import Reinforce 
import torch.nn.functional as F
from net import NeuralNet, Reinforce_Loss
batch = 3 
gamma = 0.99
test_episodes = 100
device = 'cpu'
lr = 5e-4
env = gym.make('CartPole-v0')
nA = env.action_space.n
nS = env.observation_space.shape[0]
Reinforce_net = Reinforce(nA, device, lr, nS)
test_episodes = 30


# %%
for m in range(1500):
    Reinforce_net.train(env, batch, gamma=gamma)

    if m % 100 == 0:

        G = np.zeros(test_episodes)
        Loss = np.zeros(test_episodes)  

        for k in range(test_episodes):
            rewards, loss_eval, probs, actions = Reinforce_net.evaluate_policy(env, batch) 
            G[k] = rewards 
            Loss[k] = loss_eval

        reward_mean = G.mean()
        reward_sd = G.std()
        Loss_mean = Loss.mean()
        Loss_sd = G.std()
        print("Reward sd == ", str(reward_sd), "\t Reward \\mu == ", str(reward_mean.mean()))


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

# %%
class Reinforce_Loss_Debug(torch.nn.NLLLoss):
    def forward(self, input, target, G, T):
        nll_result = F.nll_loss(input, target, reduction='none')
        return torch.div(torch.dot(G, nll_result), T), torch.dot(G, nll_result), nll_result


# %%
states, actions, rewards, policy_outputs, T = Reinforce_net.generate_episode(env, batch, sampling = False, render=False)
G = Reinforce_net.naiveGt(gamma, T, torch.zeros((T)), rewards)
actions

# %%
for m in range(500):
    Reinforce_net.train(env, batch, gamma=gamma)
   
# %%
loss = Reinforce_Loss_Debug()
l, dot, nll = loss(policy_outputs, actions, G, T) 
l


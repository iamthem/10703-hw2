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
        return torch.div(torch.dot(G, nll_result), T)


# %%
for m in range(500):
    Reinforce_net.train(env, batch, gamma=gamma)
   
# %%
rewards, loss_eval, actions, states, probs = Reinforce_net.evaluate_policy(env, batch) 


# %%
class_sample_count = np.array([len(np.where(actions == t)[0]) for t in np.arange(nA)])
C_max = np.max(class_sample_count) 
W = np.array([C_max / class_sample_count[0], C_max / class_sample_count[1]])

# %%
loss = Reinforce_Loss(weight = torch.from_numpy(W).float())
loss(policy_outputs, actions, G, T) 
#torch.isclose(torch.dot(G, policy_outputs[:,actions]), dot)

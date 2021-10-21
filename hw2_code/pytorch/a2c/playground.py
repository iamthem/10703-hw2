# %%
import torch
import numpy as np
import gym
from a2c import Reinforce 
import torch.nn.functional as F

# %%
policy = NN(nS, nA, torch.nn.LogSoftmax())
state = env.reset()
state_tensor = torch.clone(torch.from_numpy(state).float()).repeat(batch).reshape((batch, nS))

# %%
out = policy(state_tensor)
out[0]

# %%
for i in range(batch):
    print(torch.isclose(out, out_b[7]))

# %%
out_b = out


# %%
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
shapes = 0 
for i in range(100):
    states, actions, rewards, policy_outputs, T = Reinforce_net.generate_episode(env, batch, sampling = False, render=False)
    G = Reinforce_net.naiveGt(gamma, T, torch.zeros((T)), rewards)
    states.shape
    torch.exp(policy_outputs)
    shapes += list(states.shape)[0]


# %%
class Reinforce_Loss_Debug(torch.nn.NLLLoss):
    def forward(self, input, target, G, T):
        nll_result = F.nll_loss(input, target, reduction='none')
        return torch.div(torch.dot(G, nll_result), T)



# %%
loss = Reinforce_Loss_Debug()
loss(policy_outputs, actions, G, T) 

# Pole Angle is more than 12 degrees.
# Cart Position is more than 2.4 (center of the cart reaches the edge of
# 0       Cart Position             -4.8                    4.8
# 1       Cart Velocity             -Inf                    Inf
# 2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
# 3       Pole Angular Velocity     -Inf                    Inf

# %%
term_state = np.array ([-0.03670745, -0.01818765, 0.22, -.1])
norm_state = np.array ([-0.03670745, -0.01818765, 0.01, -.1])
p = np.array([0.7, 0.3])
count0 = 0
count1 = 0
for i in range(1000):
    tmp = np.random.choice(np.array([0,1]), p = p)
    if tmp == 0:
        count0 += 1
    else: count1 += 1

print(count1)


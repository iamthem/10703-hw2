import sys
import argparse
import numpy as np
import logging
import torch
import torch.optim as optim
from net import NeuralNet
from math import pow
# (uncomment below in AWS) 
from torchinfo import summary
logger = logging.getLogger(__name__)

class Reinforce(object):
    # Implementation of REINFORCE 

    def __init__(self, nA, device, lr, baseline=False):
        self.type = "Baseline" if baseline else "Reinforce"
        self.nA = nA
        self.device = device
        
        # TODO Should we batch inputs, i.e. input is (B x 4)?
        self.policy = NeuralNet(4, 2, torch.nn.LogSoftmax(dim=1))
        self.policy.to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        # logger.debug("optimizer of %s ==> \n %s", self.type + " Policy", str(self.optimizer))


    def evaluate_policy(self, env):
        # TODO: Compute Accumulative trajectory reward(set a trajectory length threshold if you want)
        pass

    def generate_episode(self, env, render=False):
        # Generates an episode by executing the current policy in the given env.
        # Returns:
        # - a list of states, indexed by time step
        # - a list of actions, indexed by time step
        # - a list of rewards, indexed by time step
        
        # Currently ignoring initial state
        state = env.reset()
        
        # init states, actions, rewards as tensor
        actions, rewards = torch.zeros((2, 200), device = self.device)
        states = torch.zeros((200, 4), device = self.device)
        
        done = False
        t = 0

        while not done:

            ## TODO How would batching affect this call?
            action = torch.argmax(self.policy(torch.from_numpy(state).float().to(self.device)))
            new_state, reward, done, info = env.step(action)
            states[t], actions[t], rewards[t] = torch.from_numpy(new_state).float().to(self.device), action, reward
            state = new_state
            t += 1

        env.close()
        T = torch.count_nonzero(rewards) 
        return states[:T, :], actions[:T], torch.flatten(rewards[torch.nonzero(rewards)])

    # TODO perform this action in one loop
    # Calculate G_t 
    def naiveGt(self, gamma, T, G, rewards):
        for t in range(T):
            
            gammas = torch.tensor([pow(gamma, k - t) for k in range(t, T)], device = self.device)
            # logger.debug('gamma[%d] = %s', t, str(gammas))
            G[t] = torch.dot(rewards[t:], gammas)

        return G

    # TODO construct loss (Use NLLLoss)
    def loss(self, outputs):
        for t in range(outputs.size()):
            pass
        return 

    # TODO debug and check everything here (too slow?)
    def update_policy(self, states, actions):
                  
        outputs = [self.policy(state) for state in states]
        loss_train = self.loss(outputs, actions)

        self.optimizer.zero_grad()
        loss_train.backward() 
        self.optimizer.step()

    def train(self, env, gamma=0.99, n=10):
        states, actions, rewards = self.generate_episode(env)

        logger.debug("states ==> %s\n, actions %s\n, rewards %s\n", str(states), str(actions), str(rewards))
        # What happens in final state? / Does it need special consideration? 
        T = len(rewards)
        G = self.naiveGt(gamma, T, torch.zeros((T), device = self.device), rewards)

        # self.update_policy(states, actions)

        return 


class A2C(Reinforce):
    # Implementation of N-step Advantage Actor Critic.
    # This class inherits the Reinforce class, so for example, you can reuse
    # generate_episode() here for different methods.

    def __init__(self, actor, actor_lr, N, nA, critic, critic_lr):
        self.type = "A2C" 
        logger.debug('Type is ', self.type)

        ## arguments unclear (should we prepend self and declare argument above?)
        super(A2C, self).__init__(nA)

    def evaluate_policy(self, env):
        pass

    def generate_episode(self, env, render=False):
        # TODO: Call parent class (pretty sure) 
        pass

    def train(self, env, gamma=0.99, n=10):
        pass

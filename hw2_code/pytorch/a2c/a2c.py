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

    def __init__(self, nA, device, lr, dim_in, dim_out, baseline=False):
        self.type = "Baseline" if baseline else "Reinforce"
        self.nA = nA
        self.device = device
        
        self.policy = NeuralNet(dim_in, dim_out, torch.nn.LogSoftmax(dim = 0))
        self.policy.to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        # logger.debug("optimizer of %s ==> \n %s", self.type + " Policy", str(self.optimizer))


    def evaluate_policy(self, env):
        # TODO: Compute Accumulative trajectory reward(set a trajectory length threshold if you want)
        pass

    def generate_episode(self, env, batch = 1, render=False):
        # Generates an episode by executing the current policy in the given env.
        # Returns:
        # - a list of states, indexed by time step
        # - a list of actions, indexed by time step
        # - a list of rewards, indexed by time step
        
        # Currently ignoring initial state
        state = env.reset()
        
        # init states, actions, rewards as tensor
        actions, rewards = torch.zeros((2, 200), device = self.device)
        states = torch.zeros((200, batch, 4), device = self.device)
        policy_outputs = torch.zeros((200, batch, 2), device = self.device)
        
        done = False
        t = 0

        while not done:


            pi_a = self.policy(torch.from_numpy(state).float().to(self.device))
            action = int(torch.argmax(pi_a))
            new_state, reward, done, info = env.step(action)

            ## Potential slowdown here? 
            state_batch = torch.from_numpy(
                                           np.array(
                                                   [np.copy(new_state) for i in range(batch)]
                                                   )
                                          ).float().to(self.device)

            states[t], actions[t], rewards[t], policy_outputs[t] = state_batch, action, reward, pi_a
            state = new_state
            t += 1

        env.close()
        T = torch.count_nonzero(rewards) 
        return states[:T, :, :], actions[:T], torch.flatten(rewards[torch.nonzero(rewards)]),policy_outputs[:T, :, :]

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
    def update_policy(self, states, actions, outputs):
                  
        loss_train = self.loss(outputs, actions)

        self.optimizer.zero_grad()
        loss_train.backward() 
        self.optimizer.step()

    def train(self, env, batch=1, gamma=0.99, n=10):

        states, actions, rewards, policy_outputs = self.generate_episode(env, batch)
        #logger.debug("actions ==> %s\n, policy_outputs ==> %s", str(actions), str(policy_outputs))

        # What happens in final state? / Does it need special consideration? 
        T = len(rewards)
        G = self.naiveGt(gamma, T, torch.zeros((T), device = self.device), rewards)

        self.update_policy(states, actions, policy_outputs)

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

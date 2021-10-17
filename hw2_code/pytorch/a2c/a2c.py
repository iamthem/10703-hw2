import sys
import argparse
import numpy as np
import logging
import torch
from net import NeuralNet
from math import pow
# (uncomment below in AWS) 
from torchinfo import summary
logger = logging.getLogger(__name__)

class Reinforce(object):
    # Implementation of REINFORCE 

    def __init__(self, nA, device, baseline=False):
        self.type = "Baseline" if baseline else "Reinforce"
        self.nA = nA
        self.device = device
        
        # TODO Should we batch inputs, i.e. input is (B x 4)?
        self.policy = NeuralNet(4, 1, torch.nn.Sigmoid)
        logger.debug("Summary of %s ==> \n %s", self.type + " Policy", summary(self.policy))


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
        
        # TODO implement below loop 
        done = False
        t = 0

        while not done:

            ## TODO how to call current policy?
            action = env.action_space.sample()
            new_state, reward, done, info = env.step(action)
            states[t], actions[t], rewards[t] = torch.from_numpy(new_state), action, reward
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

    def train(self, env, gamma=0.99, n=10):
        # Trains the model on a single episode using REINFORCE or A2C/A3C.
        # TODO: Implement this method. It may be helpful to call the class
        #       method generate_episode() to generate training data.

        states, actions, rewards = self.generate_episode(env)

        # What happens in final state? / Does it need special consideration? 
        T = len(rewards)
        G = self.naiveGt(gamma, T, torch.zeros((T), device = self.device), rewards)
        return G 


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
        # TODO: Compute Accumulative trajectory reward(set a trajectory length threshold if you want)
        pass

    def generate_episode(self, env, render=False):
        # Generates an episode by executing the current policy in the given env.
        # Returns:
        # - a list of states, indexed by time step
        # - a list of actions, indexed by time step
        # - a list of rewards, indexed by time step
        # TODO: Implement this method.
        pass

    def train(self, env, gamma=0.99, n=10):
        # Trains the model on a single episode using REINFORCE or A2C/A3C.
        # TODO: Implement this method. It may be helpful to call the class
        #       method generate_episode() to generate training data.
        pass

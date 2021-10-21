#import numpy as np 
import logging
import torch
import torch.optim as optim
from net import NeuralNet, Reinforce_Loss
from math import pow
# (uncomment below in AWS) 
# from torchinfo import summary
logger = logging.getLogger(__name__)

class Reinforce(object):
    # Implementation of REINFORCE 

    def __init__(self, nA, device, lr, nS, baseline=False):
        self.type = "Baseline" if baseline else "Reinforce"
        self.nA = nA
        self.nS = nS
        self.device = device
        self.policy = NeuralNet(nS, nA, torch.nn.LogSoftmax())
        self.policy.to(self.device)
        self.loss = Reinforce_Loss() 
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        # logger.debug("optimizer of %s ==> \n %s", self.type + " Policy", str(self.optimizer))


    def evaluate_policy(self, env, batch = 1):
        # TODO: try different params to see if network improves performance 

        states, actions, rewards, policy_outputs, T = self.generate_episode(env, batch, sampling = True)
        
        # undiscounted return ==> gamma = 1 
        G = self.naiveGt(0.99, T, torch.zeros(T, device = self.device), rewards)
        loss = self.loss(policy_outputs, actions, G, T)

        return torch.sum(rewards), loss, actions, states, torch.exp(policy_outputs)

    def generate_episode(self, env, batch = 1, sampling = False, render=False):
        # Generates an episode by executing the current policy in the given env.
        # Returns:
        # - a list of states, indexed by time step
        # - a list of actions, indexed by time step
        # - a list of rewards, indexed by time step
        
        if render: 
            pass

        # Currently ignoring initial state
        state = env.reset()
        
        # init states, actions, rewards as tensor
        actions = torch.zeros((200), dtype = torch.long, device = self.device)
        rewards = torch.zeros((200), device = self.device)
        states = torch.zeros((200, batch, self.nS), device = self.device)
        policy_outputs = torch.zeros((200, batch, self.nA), device = self.device)
        
        done = False
        t = 0
        
        # State is now copied Batch times into a tensor with shape (Batch x nS)
        state = torch.clone(torch.from_numpy(state).float().to(self.device)).repeat(batch).reshape((batch, self.nS))
        new_state = None

        while not done:

            pi_a = self.policy(state)

            if sampling:
                p = torch.exp(pi_a[0])
                action = p.multinomial(num_samples=1, replacement=True)
                action = int(action)
            else: 
                action = int(torch.argmax(pi_a[0]))

            new_state_np, reward, done, _ = env.step(action)

            ## Debug 
            del new_state
            new_state = torch.clone(torch.from_numpy(new_state_np).float().to(self.device)).repeat(batch).reshape((batch, self.nS))

            states[t], actions[t], rewards[t], policy_outputs[t] = new_state, action, reward, pi_a
            state = new_state
            t += 1

        env.close()
        T = t 
        return states[:T, :, :], actions[:T], torch.flatten(rewards[:T]), policy_outputs[:T, 0, :], T 

    # TODO perform this action in one loop
    def naiveGt(self, gamma, T, G, rewards):
        # Calculate G_t 
        for t in range(T):
            
            gammas = torch.tensor([pow(gamma, k - t) for k in range(t, T)], device = self.device)
            # logger.debug('gamma[%d] = %s', t, str(gammas))
            G[t] = torch.dot(rewards[t:], gammas)

        return G

    # TODO debug and check everything here (too slow?)
    def update_policy(self, policy_outputs, actions, G, T):
                  
        loss_train = self.loss(policy_outputs, actions, G, T) 
        self.optimizer.zero_grad()
        loss_train.backward() 
        self.optimizer.step()
        return 

    def train(self, env, batch=1, gamma=0.99, n=10):

        _, actions, rewards, policy_outputs, T = self.generate_episode(env, batch)
        #logger.debug("Input shape ==> %s \n Target shape ==> %s", str(policy_outputs.shape), str(actions.shape))

        # What happens in final state? / Does it need special consideration? 
        G = self.naiveGt(gamma, T, torch.zeros((T), device = self.device), rewards)
        self.update_policy(policy_outputs, actions, G, T)
        return 


# class A2C(Reinforce):
#     # Implementation of N-step Advantage Actor Critic.
#     # This class inherits the Reinforce class, so for example, you can reuse
#     # generate_episode() here for different methods.

#     def __init__(self, actor, actor_lr, N, nA, critic, critic_lr):
#         self.type = "A2C" 
#         logger.debug('Type is ', self.type)

#         ## arguments unclear (should we prepend self and declare argument above?)
#         super(A2C, self).__init__(nA)

#     def evaluate_policy(self, env):
#         pass

#     def generate_episode(self, env, render=False):
#         # TODO: Call parent class (pretty sure) 
#         pass

#     def train(self, env, gamma=0.99, n=10):
#         pass

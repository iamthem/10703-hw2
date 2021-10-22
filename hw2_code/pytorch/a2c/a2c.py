import numpy as np 
import logging
import torch
import torch.optim as optim
from net import NeuralNet, Reinforce_Loss, NeuralNetSmall
from math import pow
# (uncomment below in AWS) 
# from torchinfo import summary
logger = logging.getLogger(__name__)

class Reinforce(object):
    # Implementation of REINFORCE 

    def __init__(self, nA, device, lr, nS, baseline=False, baseline_lr = 0.0):
        self.type = "Baseline" if baseline else "Reinforce"
        self.nA = nA
        self.nS = nS
        self.device = device
        self.policy = NeuralNet(nS, nA, torch.nn.LogSoftmax())
        self.policy.to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        if baseline:
            layers = [32,32,16]
            self.baseline = NeuralNetSmall(nS, 1, torch.nn.Linear(layers[2], 1))
            self.baseline.to(self.device)
            self.optimizer_base = optim.Adam(self.baseline.parameters(), lr = baseline_lr)
            self.loss_base = torch.nn.MSELoss()

        else:
            self.baseline = lambda state: 0 

    def evaluate_policy(self, env, batch = 1):

        _, actions, rewards, policy_outputs, T, base_out = self.generate_episode(env, batch, sampling = True)
        
        # G = self.naiveGt(0.99, T, torch.zeros(T, device = self.device), rewards)
        # W = self.getW(actions)
        # loss = Reinforce_Loss(weight = W) 
        # loss_eval = loss(policy_outputs, actions, G, T)

        # undiscounted return ==> gamma = 1 
        return torch.sum(rewards)

    def generate_episode(self, env, batch = 1, sampling = False, render=False, baseline = False):
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
        baseline_outputs = torch.zeros((200), device = self.device)
        
        done = False
        t = 0
        
        # State is now copied Batch times into a tensor with shape (Batch x nS)
        state = torch.clone(torch.from_numpy(state).float().to(self.device)).repeat(batch).reshape((batch, self.nS))
        new_state = None

        while not done:

            pi_a = self.policy(state)
            b_out = self.baseline(state[0])

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

            states[t], actions[t], rewards[t], policy_outputs[t], baseline_outputs[t] = new_state, action, reward, pi_a, b_out
            state = new_state
            t += 1

        env.close()
        T = t 
        return states[:T, :, :], actions[:T], torch.flatten(rewards[:T]), policy_outputs[:T, 0, :], T, baseline_outputs[:T]

    def naiveGt(self, gamma, T, G, rewards):
        # Calculate G_t 
        for t in range(T):
            
            gammas = torch.tensor([pow(gamma, k - t) for k in range(t, T)], device = self.device)
            # logger.debug('gamma[%d] = %s', t, str(gammas))
            G[t] = torch.dot(rewards[t:], gammas)

        return G

    def getW(self, actions):
        actions = actions.cpu()
        class_sample_count = np.array([len(np.where(actions == a)[0]) for a in np.arange(self.nA)])
        C_max = np.max(class_sample_count) 
        if C_max == np.sum(class_sample_count): 
            return None 

        W = np.array([C_max / class_sample_count[0], C_max / class_sample_count[1]])
        return torch.from_numpy(W).float().to(self.device)

    def update_policy(self, policy_outputs, actions, G, T, base_out):
                  
        W = self.getW(actions)

        if W is None: 
            return

        loss = Reinforce_Loss(weight = W)

        if self.type == "Baseline" or self.type == "A2C":

            with torch.no_grad():
                G = torch.sub(G, base_out)  

            loss_b = self.loss_base(base_out, G)
            self.optimizer_base.zero_grad()
            loss_b.backward() 
            self.optimizer_base.step()

        loss_train = loss(policy_outputs, actions, G, T) 
        self.optimizer.zero_grad()
        loss_train.backward() 
        self.optimizer.step()


        return 

    def train(self, env, batch=1, gamma=0.99, n=10):

        _, actions, rewards, policy_outputs, T, base_out = self.generate_episode(env, batch, sampling = True)

        with torch.no_grad():
            G = self.naiveGt(gamma, T, torch.zeros((T), device = self.device), rewards)

        self.update_policy(policy_outputs, actions, G, T, base_out)

        return 


class A2C(Reinforce):
    def __init__(self, nA, device, lr, nS, critic_lr, baseline = True):
        Reinforce.__init__(self, nA, device, lr, nS, baseline, baseline_lr = critic_lr)
        self.type = "A2C" 

    def naiveGt(self, gamma, T, G, rewards, n, base_out):
        # Calculate G_t 
        for t in range(T):

            if t + n >= T:
                gammas = torch.tensor([pow(gamma, k - t) for k in range(t, T)], device = self.device)
                G[t] = torch.dot(rewards[t:], gammas)
                
            elif t + n < T:
                V_end = base_out[t + n]
                R_ks = rewards[t : t + n]
                end_of_range = int(np.min(np.array([t + n, T], np.intc)))
                gammas = torch.tensor([pow(gamma, k - t) for k in range(t, end_of_range)], device = self.device)
                assert(gammas.size() ==  R_ks.size())
                G[t] = torch.dot(R_ks, gammas) + torch.mul(V_end, pow(gamma, n))

        return G

    def evaluate_policy(self, env, batch = 1, n = 1):

        states, actions, rewards, policy_outputs, T, base_out = self.generate_episode(env, batch, sampling = True)
        # undiscounted return ==> gamma = 1 
        return torch.sum(rewards)

    def update_policy(self, policy_outputs, actions, G, T, base_out):
                  
        W = self.getW(actions)

        # Might be problematic
        if W is None: 
            return

        loss = Reinforce_Loss(weight = W)

        loss_train = loss(policy_outputs, actions, torch.sub(G, base_out), T) 
        loss_b = self.loss_base(base_out, G)

        self.optimizer.zero_grad()
        loss_train.backward(retain_graph = True)
        self.optimizer.step()

        self.optimizer_base.zero_grad()
        loss_b.backward(retain_graph = True) 
        self.optimizer_base.step()

        return 

    def train(self, env, batch=1, gamma=0.99, n=10):

        states, actions, rewards, policy_outputs, T, base_out = self.generate_episode(env, batch, sampling = True)

        G = self.naiveGt(gamma, T, torch.zeros((T), device = self.device), rewards, n, base_out)

        self.update_policy(policy_outputs, actions, G, T, base_out)

        return 

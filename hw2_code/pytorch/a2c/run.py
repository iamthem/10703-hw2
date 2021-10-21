import sys
import argparse
import os
import logging
import torch
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '1'

import gym

from a2c import Reinforce

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tqdm


def parse_a2c_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', dest='env_name', type=str,
                        default='CartPole-v0', help="Name of the environment to be run.")
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=3500, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-4, help="The actor's learning rate.")
    parser.add_argument('--baseline-lr', dest='baseline_lr', type=float,
                        default=5e-4, help="The actor's learning rate.")
    parser.add_argument('--critic-lr', dest='critic_lr', type=float,
                        default=1e-4, help="The critic's learning rate.")
    parser.add_argument('--n', dest='n', type=int,
                        default=100, help="The value of N in N-step A2C.")

    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--render', dest='render',
                              action='store_true',
                              help="Whether to render the environment.")
    parser_group.add_argument('--no-render', dest='render',
                              action='store_false',
                              help="Whether to render the environment.")
    parser.set_defaults(render=False)

    return parser.parse_args()

def hists(Loss, Loss_mean, G, G_mean, m, reward_sd, Loss_sd, ignore = False):
    if ignore: 
        return

    size = 8
    f = "Episode_" + str(m)
    plt.rc('axes', labelsize=size)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=size)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=size)    # fontsize of the tick labels
    fig, axs = plt.subplots(2)
    fig.suptitle(f)
    axs[0].hist(G, 20)
    axs[0].vlines(G_mean, 0, 30)
    axs[0].set_title("Rewards Histogram. \\mu = " + str(round(G_mean, 3)) + " sd = " + str(round(reward_sd, 3)), fontsize = size)
    axs[1].hist(Loss, 20)
    axs[1].vlines(Loss_mean, 0, 30)
    axs[1].set_title("Loss Histogram. \\mu = " + str(round(Loss_mean, 3)) + " sd = " + str(round(Loss_sd, 3)), fontsize = size)
    plt.savefig("/home/junaikin/Notes/AI+ML/10703/hw2/hw2_code/pytorch/a2c/.plots/" + f)
    plt.clf()
    return

def main_a2c(args):

    # Parse command-line arguments.
    args = parse_a2c_arguments()
    env_name = args.env_name


    num_episodes = args.num_episodes
    lr = args.lr
    baseline_lr = args.baseline_lr
    critic_lr = args.critic_lr
    # render = args.render

    # Create the environment.
    env = gym.make(env_name)
    nA = env.action_space.n
    nS = env.observation_space.shape[0]

    # Plot average performance of 5 trials
    num_seeds = 5
    frozen_pi_per_trial = num_episodes//100
    res = np.zeros((num_seeds, frozen_pi_per_trial))

    gamma = 0.99
    batch = 3 
    test_episodes = 20

    ## defaults above this line  

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i in tqdm.tqdm(range(num_seeds)):
        reward_means = []

        Reinforce_net = Reinforce(nA, device, lr, nS)
        
        # Insert code from handout.py below 
        for m in tqdm.tqdm(range(num_episodes)):
            Reinforce_net.train(env, batch, gamma=gamma)
            if m % 100 == 0:
                logger.debug("Episode: {}".format(m))
                G = np.zeros(test_episodes)
                for k in range(test_episodes):
                    g = Reinforce_net.evaluate_policy(env, batch)
                    G[k] = g

                reward_mean = G.mean()
                reward_sd = G.std()
                logger.debug("The test reward for episode {0} is {1} with sd of {2}.".format(m, reward_mean, reward_sd))
                reward_means.append(reward_mean)
        res[i] = np.array(reward_means)
    ks = np.arange(frozen_pi_per_trial)*100
    avs = np.mean(res, axis=0)
    maxs = np.max(res, axis=0)
    mins = np.min(res, axis=0)

    plt.fill_between(ks, mins, maxs, alpha=0.1)
    plt.plot(ks, avs, '-o', markersize=1)

    plt.xlabel('Episode', fontsize = 15)
    plt.ylabel('Return', fontsize = 15)
    plt.title("Reinforce Learning Curve", fontsize = 24)
    if not os.path.exists('./plots'):
        os.mkdir('./plots')

    plt.savefig("./plots/Reinforce_curve.png")

if __name__ == '__main__':

    logfile = 'ac2.log'
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(logfile, mode = 'w')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    torch.set_printoptions(precision = 3, edgeitems=2, linewidth=75)
    main_a2c(sys.argv)

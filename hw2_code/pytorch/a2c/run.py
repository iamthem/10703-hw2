import sys
import argparse
import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '1'

import gym

from a2c import Reinforce, A2C
from net import NeuralNet

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

    # Plot average performance of 5 trials
    num_seeds = 5
    l = num_episodes//100
    res = np.zeros((num_seeds, l))

    gamma = 0.99

    # defaults above this line  

    for i in tqdm.tqdm(range(num_seeds)):
        reward_means = []

        # TODO: create networks and setup reinforce/a2c


        Reinforce_net = Reinforce(nA)
        
        # Insert code from handout.py below 
        



if __name__ == '__main__':

    logfile = 'ac2.log'
    loglevel = logging.WARNING
    logging.basicConfig(filename=logfile, filemode='a', level=loglevel)
    logging.info('\n-------------------- START --------------------')
    main_a2c(sys.argv)

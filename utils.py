import torch
import random
import numpy as np
from collections import deque

class ReplayBuffer:
    """
    Experience replay buffer for storing data for training RL agent
    """
    def __init__(self, size):
        self.size = size
        self.deque = deque(maxlen=self.size)

    def push(self, experience):
        """push into the buffer"""
        self.deque.append(experience)

    def sample(self, batchsize):
        """sample from the buffer"""
        return random.sample(self.deque, batchsize)

    def __len__(self):
        return len(self.deque)


class OUNoise:
    """
    Ornstein-Uhlenbeck (OU) Noise for DDPG Agent Exploration
    from https://github.com/songrotek/DDPG/blob/master/ou_noise.py
    """

    def __init__(self, action_dimension, scale=1.0, mu=0, theta=0.15, sigma=0.1):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return torch.tensor(self.state * self.scale).float()


def transpose_list(mylist):
    return list(map(list, zip(*mylist)))

def transpose_to_tensor(input_list):
    make_tensor = lambda x: torch.tensor(x, dtype=torch.float)
    return list(map(make_tensor, zip(*input_list)))

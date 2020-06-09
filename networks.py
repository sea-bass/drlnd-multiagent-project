import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def hidden_init(layer):
    """ Initialize hidden layer weights """
    fan_in = layer.weight.data.size()[0]
    lim = 1.0 / np.sqrt(fan_in)
    return (-lim, lim)


class ActorNetwork(nn.Module):
    """ 
    Defines an actor network for DDPG.
    The actor accepts an observation and outputs an action
    """
    def __init__(self, obs_dim=24, act_dim=2, hidden_dims=[128, 64, 32]):
        super(ActorNetwork, self).__init__()

        self.fc1 = nn.Linear(obs_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.fc4 = nn.Linear(hidden_dims[2], act_dim)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-1e-3, 1e-3)

    def forward(self, obs):
        h1 = F.relu(self.fc1(obs))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        act = F.tanh(self.fc4(h3))
        return act


class CriticNetwork(nn.Module):
    """ 
    Defines a critic network for DDPG.
    The actor accepts an observation and action and outputs a Q-value estimate
    """

    def __init__(self, num_agents=2, obs_dim=24, act_dim=2, hidden_dims=[128, 64, 32]):
        super(CriticNetwork, self).__init__()

        self.fc1 = nn.Linear(obs_dim*num_agents, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1] + act_dim*num_agents, hidden_dims[2])
        self.fc4 = nn.Linear(hidden_dims[2], 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-1e-3, 1e-3)

    def forward(self, obs_full, act_full):
        h1 = F.relu(self.fc1(obs_full))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(torch.cat([h2, act_full], dim=1)))
        q = self.fc4(h3)
        return q

import torch
from torch.optim import Adam
import numpy as np
from networks import ActorNetwork, CriticNetwork
from utils import OUNoise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MADDPGAgent:
    """
    Defines a Multi-Agent Deep Deterministic Policy Gradient (MADDPG) agent
    """
    def __init__(self, num_agents=2, obs_size=24, act_size=2, 
                 gamma=0.99, tau=1e-3, lr_actor=1.0e-4, lr_critic=1.0e-3, 
                 weight_decay_actor=1e-5, weight_decay_critic=1e-4, clip_grad=1.0):
        super(MADDPGAgent, self).__init__()

        self.actor = ActorNetwork(obs_size, act_size).to(device)
        self.critic = CriticNetwork(num_agents, obs_size, act_size).to(device)
        self.target_actor = ActorNetwork(obs_size, act_size).to(device)
        self.target_critic = CriticNetwork(num_agents, obs_size, act_size).to(device)
        self.noise = OUNoise(act_size, scale=1.0)

        # initialize targets same as original networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        # Write parameters
        self.num_agents = num_agents
        self.gamma = gamma
        self.tau = tau
        self.clip_grad = clip_grad

        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor, 
                                    weight_decay=weight_decay_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic, 
                                     weight_decay=weight_decay_critic)


    def act(self, obs, noise=0.0):
        obs = obs.to(device)
        action = self.actor(obs) + (noise*self.noise.noise()).to(device)
        action = torch.clamp(action, -1, 1)
        return action

    def target_act(self, obs, noise=0.0):
        obs = obs.to(device)
        action = self.target_actor(obs) + (noise*self.noise.noise()).to(device)
        action = torch.clamp(action, -1, 1)
        return action

    def update_targets(self):
        """
        Perform soft update of target network parameters based on latest actor/critic parameters
        """
        soft_update(self.target_critic, self.critic, self.tau)
        soft_update(self.target_actor, self.actor, self.tau)

    def train(self, samples):
        """
        Perform a training step
        """
     
        # Unpack data from replay buffer and convert to tensors
        obs = torch.tensor([exp[0] for exp in samples], 
            dtype=torch.float, device=device)
        act = torch.tensor([exp[1] for exp in samples],
            dtype=torch.float, device=device)
        reward = torch.tensor([exp[2] for exp in samples],
            dtype=torch.float, device=device)
        next_obs = torch.tensor([exp[3] for exp in samples],
            dtype=torch.float, device=device)
        done = torch.tensor([exp[4] for exp in samples],
            dtype=torch.float, device=device)
        obs_full = torch.tensor([exp[5] for exp in samples],
            dtype=torch.float, device=device)
        next_obs_full = torch.tensor([exp[6] for exp in samples],
            dtype=torch.float, device=device)
        act_full = torch.tensor([exp[7] for exp in samples],
            dtype=torch.float, device=device)

        # Critic update
        self.critic_optimizer.zero_grad()
        target_critic_obs = [next_obs_full[:,i,:].squeeze() \
                        for i in range(self.num_agents)]
        target_critic_obs = torch.cat(target_critic_obs, dim=1)
        target_act = [self.target_act(next_obs_full[:,i,:].squeeze()) \
                        for i in range(self.num_agents)]
        target_act = torch.cat(target_act, dim=1) 
        with torch.no_grad():
            q_next = self.target_critic(target_critic_obs, target_act)
        q_target = reward + self.gamma*q_next*(1-done)
        
        critic_obs = [obs_full[:,i,:].squeeze() \
                        for i in range(self.num_agents)]
        critic_obs = torch.cat(critic_obs, dim=1)
        critic_act = [act_full[:,i,:].squeeze() \
                        for i in range(self.num_agents)]
        critic_act = torch.cat(critic_act, dim=1) 
        q = self.critic(critic_obs, critic_act)

        critic_loss = torch.nn.functional.mse_loss(q, q_target.detach())
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.clip_grad)
        self.critic_optimizer.step()
    
        # Actor update using policy gradient
        self.actor_optimizer.zero_grad()
        actor_act = [self.act(obs_full[:,i,:].squeeze()) \
                     for i in range(self.num_agents)]
        actor_act = torch.cat(actor_act, dim=1) 
        actor_loss = -self.critic(critic_obs, actor_act).mean()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_grad)
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        self.update_targets()

        # Print losses
        # al = actor_loss.cpu().detach().item()
        # cl = critic_loss.cpu().detach().item()
        # print("Agent {} losses. Actor: {} Critic: {}".format(aidx, al, cl))


# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L11
def soft_update(target, source, tau):
    """
    Perform DDPG soft update (move target params toward source based on weight
    factor tau)
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
        tau (float, 0 < x < 1): Weight factor for update
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L15
def hard_update(target, source):
    """
    Copy network parameters from source to target
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

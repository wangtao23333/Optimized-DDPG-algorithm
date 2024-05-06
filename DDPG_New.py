import logging
import os

import numpy as np
import torch
import torch.nn.functional as F

#from Agent import Agent
from Buffer import Buffer


def setup_logger(filename):
    """ set up logger with filename. """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(filename, mode='w')
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s--%(levelname)s--%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger


class DDPG:
    def __init__(self, obs_dim_list, act_dim_list, capacity, actor_lr, critic_lr, res_dir=None, device=None):
        """
        :param obs_dim_list: list of observation dimension of each agent
        :param act_dim_list: list of action dimension of each agent
        :param capacity: capacity of the replay buffer
        :param res_dir: directory where log file and all the data and figures will be saved
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print(f'training on device: {self.device}')
        # sum all the dims of each agent to get input dim for critic
        global_obs_dim = sum(obs_dim_list + act_dim_list)
        # create all the agents and corresponding replay buffer
        self.agents = []
        self.buffers = []
        for obs_dim, act_dim in zip(obs_dim_list, act_dim_list):
            self.agents.append(Agent(obs_dim, act_dim, global_obs_dim, actor_lr, critic_lr, device=self.device))
            self.buffers.append(Buffer(capacity, obs_dim, act_dim, self.device))
        if res_dir is None:
            self.logger = setup_logger('ddpg.log')
        else:
            self.logger = setup_logger(os.path.join(res_dir, 'ddpg.log'))

    def add(self, obs, actions, rewards, next_obs, dones):
        """add experience to buffer"""
        for n, buffer in enumerate(self.buffers):
            buffer.add(obs[n], actions[n], rewards[n], next_obs[n], dones[n])

    def sample(self, batch_size, agent_index):
        """sample experience from all the agents' buffers, and collect data for network input"""
        # get the total num of transitions, these buffers should have same number of transitions
        total_num = len(self.buffers[0])
        indices = np.random.choice(total_num, size=batch_size, replace=False)


        # but only the reward and done of the current agent is needed in the calculation
        obs_list, act_list, next_obs_list, next_act_list = [], [], [], []
        reward_cur, done_cur, obs_cur = None, None, None
        for n, buffer in enumerate(self.buffers):
            obs, action, reward, next_obs, done = buffer.sample(indices)
            obs_list.append(obs)
            act_list.append(action)
            next_obs_list.append(next_obs)
            # calculate next_action using target_network and next_state
            next_act_list.append(self.agents[n].target_action(next_obs))
            if n == agent_index:  # reward and done of the current agent
                obs_cur = obs
                reward_cur = reward
                done_cur = done

        return obs_list, act_list, reward_cur, next_obs_list, done_cur, next_act_list, obs_cur

    def select_action(self, obs):
        actions = []
        for n, agent in enumerate(self.agents):  # each agent select action according to their obs
            o = torch.from_numpy(obs[n]).unsqueeze(0).float().to(self.device)  # torch.Size([1, state_size])
            # Note that the result is tensor, convert it to ndarray before input to the environment
            act = agent.action(o).squeeze(0).detach().cpu().numpy()
            actions.append(act)
            # self.logger.info(f'agent {n}, obs: {obs[n]} action: {act}')
        return actions

    def learn(self, batch_size, gamma):
        for i, agent in enumerate(self.agents):
            obs, act, reward_cur, next_obs, done_cur, next_act, obs_cur = self.sample(batch_size, i)
            # update critic
            critic_value = agent.critic_value(obs, act)

            # calculate target critic value
            next_target_critic_value = agent.target_critic_value(next_obs, next_act)
            target_value = reward_cur + gamma * next_target_critic_value * (1 - done_cur)

            critic_loss = F.mse_loss(critic_value, target_value.detach(), reduction='mean')
            agent.update_critic(critic_loss)

            # update actor
            # action of the current agent is calculated using its actor
            action, logits = agent.action(obs_cur, model_out=True)
            act[i] = action
            actor_loss = -agent.critic_value(obs, act).mean()
            actor_loss_pse = torch.pow(logits, 2).mean()
            agent.update_actor(actor_loss + 1e-3 * actor_loss_pse)
            # self.logger.info(f'agent{i}: critic loss: {critic_loss.item()}, actor loss: {actor_loss.item()}')

    def update_target(self, tau):
        def soft_update(from_network, to_network):
            """ copy the parameters of `from_network` to `to_network` with a proportion of tau"""
            for from_p, to_p in zip(from_network.parameters(), to_network.parameters()):
                to_p.data.copy_(tau * from_p.data + (1.0 - tau) * to_p.data)

        for agent in self.agents:
            soft_update(agent.actor, agent.target_actor)
            soft_update(agent.critic, agent.target_critic)


from copy import deepcopy
from typing import List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.optim import Adam

class Agent:
    def __init__(self, obs_dim, act_dim, global_obs_dim, actor_lr, critic_lr,  ssl_lr=0.0001, device='cpu'):
        self.actor = MLPNetwork(obs_dim, act_dim).to(device)
        self.critic1 = MLPNetwork(global_obs_dim, 1).to(device)
        self.critic2 = MLPNetwork(global_obs_dim, 1).to(device)
        self.ssl_head = MLPNetwork(global_obs_dim, obs_dim).to(device)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=critic_lr)
        self.ssl_optimizer = Adam(self.ssl_head.parameters(), lr=ssl_lr)

    def update_ssl(self, obs, next_obs):
        predicted_next_obs = self.ssl_head(obs)
        ssl_loss = F.mse_loss(predicted_next_obs, next_obs)
        self.ssl_optimizer.zero_grad()
        ssl_loss.backward()
        self.ssl_optimizer.step()
        return ssl_loss.item()

    @staticmethod
    def gumbel_softmax(logits, tau=1.0, eps=1e-20):
        # NOTE that there is a function like this implemented in PyTorch(torch.nn.functional.gumbel_softmax),
        # but as mention in the doc, it may be removed in the future, so i implement it myself
        epsilon = torch.rand_like(logits)
        logits += -torch.log(-torch.log(epsilon + eps) + eps)
        return F.softmax(logits / tau, dim=-1)

    def action(self, obs, *, model_out=False):
        """
        choose action according to given `obs`
        :param model_out: the original output of the actor, i.e. the logits of each action will be
        `gumbel_softmax`ed by default(model_out=False) and only the action will be returned
        if set to True, return the original output of the actor and the action
        """
        # this method is called in the following two cases:
        # a) interact with the environment
        # b) calculate action when update actor, where input(obs) is sampled from replay buffer with size:
        # torch.Size([batch_size, state_dim])

        logits = self.actor(obs)  # torch.Size([batch_size, action_size])
        action = self.gumbel_softmax(logits)
        if model_out:
            return action, logits
        return action

    def target_action(self, obs):
        # we use target actor to get next action given next states,
        # which is sampled from replay buffer with size torch.Size([batch_size, state_dim])

        logits = self.target_actor(obs)  # torch.Size([batch_size, action_size])
        action = self.gumbel_softmax(logits)
        return action.squeeze(0).detach()

    def critic_value(self, state_list: List[Tensor], act_list: List[Tensor]):
        x = torch.cat(state_list + act_list, 1)
        return self.critic(x).squeeze(1)  # tensor with a given length

    def target_critic_value(self, state_list: List[Tensor], act_list: List[Tensor]):
        x = torch.cat(state_list + act_list, 1)
        return self.target_critic(x).squeeze(1)  # tensor with a given length

    def update_actor(self, loss):
        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

    def update_critic(self, loss):
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()


class MLPNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64, non_linear=nn.ReLU()):
        super(MLPNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, out_dim),
        ).apply(self.init)

    @staticmethod
    def init(m):
        """init parameter of the module"""
        gain = nn.init.calculate_gain('relu')
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=gain)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        return self.net(x)


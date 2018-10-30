import random
import numpy as np
from position_bot.qnetwork import *

#torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class VanillaQAgent:
    def __init__(self, in_channel=6, nb_hidden=64, action_size=3, replay_memory = None, seed = 1412,
        lr = 1e-3, bs = 64,
        gamma=0.99, tau= 1/300, update_interval = 5):
        
        self.x = 0
        self.y = 0
        self.in_channel = in_channel
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.npseed = np.random.seed(seed)
        self.lr = lr
        self.bs = bs
        self.gamma = gamma
        self.update_interval = update_interval
        self.tau = tau
        self.losses = []

        #vanilla
        self.qnetwork_local = QNetwork(in_channel, nb_hidden, action_size).to(device)
        self.qnetwork_target = QNetwork(in_channel, nb_hidden, action_size).to(device)
        
        
        #optimizer
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)

        # replay memory
        self.memory = replay_memory
        # count time steps
        self.t_step = 0
        
    def get_eps(self, i, eps_start = 1.0, eps_end = 0.001, eps_decay = 0.999):
        eps = max(eps_start * (eps_decay ** i), eps_end)
        return(eps)
    
    def step(self, state, action, reward, next_state, done):
        #add transition to replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # learn every self.t_step
        self.t_step += 1
        if self.t_step % self.update_interval == 0:
            if len(self.memory) > self.bs:
                #vanilla
                transitions = self.memory.sample(self.bs)
                self.learn(transitions, self.gamma)

    def act(self, state, i):
        eps = self.get_eps(i)
        state = torch.from_numpy(state).float()
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        #epsilon greedy
        if random.random() > eps:
            return np.argmax(action_values.data.numpy(), axis = 1)
        else:
            return random.choice(np.arange(self.action_size))
        
    def vanilla_loss(self,q_targets,q_expected):
        loss = F.mse_loss(q_expected,q_targets)
        return(loss)
    
    def learn(self, transitions, gamma):
        #vanilla
        states, actions, rewards, next_states, dones = transitions
        
        #vanilla
        q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        #compute loss
        q_targets = rewards + (gamma * q_targets_next) * (1 - dones)
        q_expected = self.qnetwork_local(states).gather(1, actions.long())

        #vanilla
        loss = self.vanilla_loss(q_expected, q_targets)

        #append for reporting
        self.losses.append(loss.data.item())
        
        #backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        #update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
#         self.hard_update(self.qnetwork_local, self.qnetwork_target, 1/self.tau)
      
    def hard_update(self, local_model, target_model, target_interval=1e2):
        if self.t_step % target_interval==0:
            target_model.load_state_dict(local_model.state_dict())
            
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
    def save_agent(self,model_name,df_name):
        #save
        torch.save(a.qnetwork_local.state_dict(), model_name)
        self.df.write_csv(df_name,index=False)
        
    def load_agent(self,model_name,df_name):
        #load
        a.qnetwork_local.load_state_dict(torch.load(model_name))
        a.qnetwork_target.load_state_dict(torch.load(model_name))
        self.df = pd.read_csv(df_name)
            
class DoubleQAgent:
    def __init__(self, in_channel=6, nb_hidden=64, action_size=3, replay_memory = None, seed = 1412,
        lr = 1e-3, bs = 64,
        gamma=0.99, tau= 1/300, update_interval = 5):
        
        self.x = 0
        self.y = 0
        self.in_channel = in_channel
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.npseed = np.random.seed(seed)
        self.lr = lr
        self.bs = bs
        self.gamma = gamma
        self.update_interval = update_interval
        self.tau = tau
        self.losses = []

        #vanilla
        self.qnetwork_local = QNetwork(in_channel, nb_hidden, action_size).to(device)
        self.qnetwork_target = QNetwork(in_channel, nb_hidden, action_size).to(device)
        
        
        #optimizer
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)

        # replay memory
        self.memory = replay_memory
        # count time steps
        self.t_step = 0
        
    def get_eps(self, i, eps_start = 1.0, eps_end = 0.001, eps_decay = 0.999):
        eps = max(eps_start * (eps_decay ** i), eps_end)
        return(eps)
    
    def step(self, state, action, reward, next_state, done):
        #add transition to replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # learn every self.t_step
        self.t_step += 1
        if self.t_step % self.update_interval == 0:
            if len(self.memory) > self.bs:
                #vanilla
                transitions = self.memory.sample(self.bs)
                self.learn(transitions, self.gamma)

    def act(self, state, i):
        eps = self.get_eps(i)
        state = torch.from_numpy(state).float()
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        #epsilon greedy
        if random.random() > eps:
            return np.argmax(action_values.data.numpy(), axis = 1)
        else:
            return random.choice(np.arange(self.action_size))
        
    def vanilla_loss(self,q_targets,q_expected):
        loss = F.mse_loss(q_expected,q_targets)
        return(loss)
    
    def learn(self, transitions, gamma):
        #vanilla
        states, actions, rewards, next_states, dones = transitions
        
        #double
        max_actions_next = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
        q_targets_next = self.qnetwork_target(next_states).detach().gather(1, max_actions_next.long())
        
        #compute loss
        q_targets = rewards + (gamma * q_targets_next) * (1 - dones)
        q_expected = self.qnetwork_local(states).gather(1, actions.long())

        #vanilla
        loss = self.vanilla_loss(q_expected, q_targets)

        #append for reporting
        self.losses.append(loss.data.item())
        
        #backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        #update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
#         self.hard_update(self.qnetwork_local, self.qnetwork_target, 1/self.tau)
      
    def hard_update(self, local_model, target_model, target_interval=1e2):
        if self.t_step % target_interval==0:
            target_model.load_state_dict(local_model.state_dict())
            
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
    def save_agent(self,model_name,df_name):
        #save
        torch.save(a.qnetwork_local.state_dict(), model_name)
        self.df.write_csv(df_name,index=False)
        
    def load_agent(self,model_name,df_name):
        #load
        a.qnetwork_local.load_state_dict(torch.load(model_name))
        a.qnetwork_target.load_state_dict(torch.load(model_name))
        self.df = pd.read_csv(df_name)
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 09:24:44 2023

@author: supit
"""

import torch as T
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import numpy as np

class DeepQNet(nn.Module):
    # constructor
    def __init__(self, input_dim, fc1_dim, fc2_dim, n_actions, lr):
        super().__init__()
        self.input_dim = input_dim
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim
        self.n_actions = n_actions
        
        # layers
        self.fc1 = nn.Linear(*self.input_dim, self.fc1_dim)
        self.fc2 = nn.Linear(self.fc1_dim, self.fc2_dim)
        self.fc3 = nn.Linear(self.fc2_dim, self.n_actions)
        
        # optimizer: Adam
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
        # loss function: MSE
        self.loss = nn.MSELoss()
        
        # device
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state):
        # forward propagate
        x = f.relu(self.fc1(state))
        x = f.relu(self.fc2(x))
        actions = self.fc3(x)
        
        return actions

class Agent():
    def __init__(self, gamma, lr, epsilon, input_dim, n_actions,
                 batch_size, max_mem_size = 100_000,
                 eps_end = 0.01, eps_deg = 5e-4):
        # save var
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        
        self.action_space = [i for i in range (n_actions)]
        
        self.batch_size = batch_size
        self.mem_size = max_mem_size
        self.mem_ctn = 0 # memory counter
        
        self.eps_end = eps_end
        self.eps_deg = eps_deg
        
        # evaluation network
        self.qeval = DeepQNet(input_dim = input_dim,
                              fc1_dim = 256, fc2_dim = 256,
                              n_actions = n_actions, lr = self.lr)
        
        # mem array
        self.state_mem = np.zeros((self.mem_size, *input_dim), dtype=np.float32)
        self.new_state_mem = np.zeros((self.mem_size, *input_dim), dtype=np.float32)
        self.action_mem = np.zeros(self.mem_size, dtype=np.float32)
        self.reward_mem = np.zeros(self.mem_size, dtype=np.float32)
        self.terminated_mem = np.zeros(self.mem_size, dtype=np.bool)
    
    def store_transition(self, state, new_state, action, reward, terminated):
        # store in the mems
        index = self.mem_ctn%self.mem_size #wrapped-up index
        self.state_mem[index] = state
        self.new_state_mem[index] = new_state
        self.action_mem[index] = action
        self.reward_mem[index] = reward
        self.terminated_mem[index] = terminated
        
        self.mem_ctn += 1
    
    def choose_action(self, obs):
        # explore vs exploit
        if np.random.random() > self.epsilon:
            state = T.tensor([obs]).to(self.qeval.device)
            actions = self.qeval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        
        return action
    
    def learn(self):
        # wait for enoug data for batch size
        if self.mem_ctn < self.batch_size:
            return
        
        self.qeval.optimizer.zero_grad()
        
        # sample batch from mem
        mem_size = min(self.mem_ctn, self.mem_size)
        batch = np.random.choice(mem_size, self.batch_size,
                                 replace = False) # avoiding select the same sample
        
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        # to evaluate only the actions that are taken in each batch 
        
        # sample batch
        state_batch = T.tensor(self.state_mem[batch]).to(self.qeval.device)
        new_state_batch = T.tensor(self.new_state_mem[batch]).to(self.qeval.device)
        reward_batch = T.tensor(self.reward_mem[batch]).to(self.qeval.device)
        terminated_batch = T.tensor(self.terminated_mem[batch]).to(self.qeval.device)
        # print(terminated_batch)
        
        action_batch = self.action_mem[batch]
        
        # evaluate state & new state
        qval = self.qeval.forward(state_batch)[batch_index,action_batch]
        new_qval = self.qeval.forward(new_state_batch)
        
        # if new state terminated --> new qval = 0
        new_qval[terminated_batch] = 0.0
        
        q_target = reward_batch + self.gamma*T.max(new_qval, dim = 1)[0]
        
        loss = self.qeval.loss(q_target, qval).to(self.qeval.device)
        loss.backward()
        self.qeval.optimizer.step()
        
        # epsilon degradation
        self.epsilon = self.epsilon - self.eps_deg if self.epsilon > self.eps_end \
                        else self.eps_end
        
        
        
        
        
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 09:36:49 2023

@author: supit
"""

import os # for checkpoint file

import torch as T
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

class DeepQNet(nn.Module):
    def __init__(self, input_dim, fc1_dim, fc2_dim, n_actions, lr,
                 chkpt_dir, chkpt_name):
        # super constructor
        super().__init__()
        
        # checkpoint file name
        self.chkpt_dir = chkpt_dir
        self.chkpt_file = os.path.join(self.chkpt_dir, chkpt_name)
        
        # layers
        self.fc1 = nn.Linear(*input_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.fc3 = nn.Linear(fc2_dim, *n_actions)
        
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
    
    def save_checkpoint(self):
        print('...saving checkpoint...')
        T.save(self.state_dict(), self.chkpt_file)
    
    def load_checkpoint(self):
        print('...loading checkpoint...')
        self.load_state_dict(T.load(self.chkpt_file))
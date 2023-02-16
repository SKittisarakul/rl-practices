# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 10:10:49 2023

@author: supit
"""

import numpy as np

class ReplayMemory(object):
    def __init__(self, mem_size, input_dim, n_actions):
        self.mem_size = mem_size
        # self.input_dim = input_dim
        # self.n_actions = n_actions
        
        # mem counter
        self.mem_ctn = 0
        
        # mem arrays
        self.state_mem = np.zeros((self.mem_size, *input_dim), dtype=np.float32)
        self.new_state_mem = np.zeros((self.mem_size, *input_dim), dtype=np.float32)
        self.action_mem = np.zeros((self.mem_size, *n_actions), dtype=np.int32)
        self.reward_mem = np.zeros(self.mem_size, dtype=np.float32)
        self.done_mem = np.zeros(self.mem_size, dtype=np.bool_)
        
    def store_mem(self, state, new_state, action, reward, done):
        # mem index
        index = self.mem_ctn%self.mem_size
        
        # assign values into array
        self.state_mem[index] = state
        self.new_state_mem[index] = new_state
        
        self.action_mem[index] = 0
        self.action_mem[index, action] = 1
        
        self.reward_mem[index] = reward
        self.done_mem[index] = done
        
        # increment counter
        self.mem_ctn += 1
        
    def sample_buffer(self, batch_size):
        # specify batch size
        max_mem = min(self.mem_size, batch_size)
        # sample  batch_size samples from arange(max_mem)
        #: specify indices for this batch
        batch = np.random.choice(max_mem, batch_size)
        
        states = self.state_mem[batch]
        new_states = self.new_state_mem[batch]
        actions = self.action_mem[batch]
        rewards = self.reward_mem[batch]
        doneFs = self.done_mem[batch]

        return states, new_states, actions, rewards, doneFs        
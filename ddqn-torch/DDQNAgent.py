# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 10:08:56 2023

@author: supit
"""

import torch as T
import numpy as np
from ReplayMemory import ReplayMemory
from DeepQNet import DeepQNet

class DDQNAgent(object):
    def __init__(self, gamma, lr, epsilon, input_dim, n_actions,
                 batch_size, mem_size,
                 eps_end = 0.01, eps_dec = 5e-4,
                 replace_target = 1_000,
                 algo = 'ddqn', chkpt_dir = 'tmp'):
        # save var
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        
        self.action_space = [i for i in range (*n_actions)]
        
        self.batch_size = batch_size
        self.memmory = ReplayMemory(mem_size, input_dim, n_actions)
        self.mem_ctn = 0 # memory counter
        
        self.eps_end = eps_end
        self.eps_dec = eps_dec
        
        # target network replacement
        self.replace_ctn = replace_target
        self.learn_ctn = 0
        
        # checkpoint file directory & name
        self.chkpt_dir = chkpt_dir
        
        # evaluation network
        self.qEvalNet = DeepQNet(input_dim = input_dim,
                                 fc1_dim = 512, fc2_dim = 512,
                                 n_actions = n_actions, lr=self.lr,
                                 chkpt_dir = self.chkpt_dir,
                                 chkpt_name = algo+'_q_eval')
        
        # evaluation network
        self.qTargetNet = DeepQNet(input_dim = input_dim,
                                 fc1_dim = 512, fc2_dim = 512,
                                 n_actions = n_actions, lr=self.lr,
                                 chkpt_dir = self.chkpt_dir,
                                 chkpt_name = algo+'_q_target')
    
    def store_transition(self, state, new_state, action, reward, done):
        self.memmory.store_mem(state, new_state, action, reward, done)
    
    def batch_sampling(self):
        states_, new_states_, actions_, rewards_, doneFs_ = \
            self.memmory.sample_buffer(self.batch_size)
        
        actions = np.dot(actions_, self.action_space)
        
        # to tensor & device
        states = T.tensor(states_).to(self.qEvalNet.device)
        new_states = T.tensor(new_states_).to(self.qEvalNet.device)
        # actions = T.tensor(actions_).to(self.qEvalNet.device)
        rewards = T.tensor(rewards_).to(self.qEvalNet.device)
        doneFs = T.tensor(doneFs_).to(self.qEvalNet.device)
        
        return states, new_states, actions, rewards, doneFs
    
    def choose_action(self, obs):
        # explore vs exploit
        rand_eps = np.random.random()
        
        if rand_eps > self.epsilon:
            # exploit
            state = T.tensor([obs], dtype=T.float32).to(self.qEvalNet.device)
            actions = self.qEvalNet.forward(state)
            action = T.argmax(actions).item()
        else: # explore
            action = np.random.choice(self.action_space)
        
        return action
    
    def eps_decrement(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_end \
            else self.eps_end
    
    def replace_targetNet(self):
        # replace target network every replace_ctn times
        if self.replace_ctn is not None and\
            self.learn_ctn % self.replace_ctn == 0:
            self.qTargetNet.load_state_dict(self.qEvalNet.state_dict())
    
    def learn(self):
        # wait until batch is filled up
        if self.memmory.mem_ctn < self.batch_size:
            return
        
        # set all optimized tensors' gradient to zeros
        self.qEvalNet.optimizer.zero_grad()
        
        # replace target network
        self.replace_targetNet()
        
        # batch sampling
        states, new_states, actions, rewards, doneFs = \
            self.batch_sampling()
        batch_indices = np.arange(self.batch_size)
        
        # evaluate
        qVal = self.qEvalNet.forward(states)[batch_indices, actions]
        new_qVal = self.qEvalNet.forward(new_states)
        # find action with max new_qEval
        max_action = T.argmax(new_qVal, dim=1)
        # assign qVal =0 for terminated states
        qVal[doneFs] = 0.0
        
        # cal new_qTarget
        new_qTarget = self.qTargetNet.forward(new_states)
        qTarget = rewards + self.gamma*new_qTarget[batch_indices, max_action]
        
        # cal loss for update qEvalNet
        loss = self.qEvalNet.loss(qTarget, qVal).to(self.qEvalNet.device)
        loss.backward()
        
        self.qEvalNet.optimizer.step()
        self.learn_ctn += 1
        
        self.eps_decrement()
        
    def save_model(self):
        self.qEvalNet.save_checkpoint()
        self.qTargetNet.save_checkpoint()
    
    def load_model(self):
        self.qEvalNet.load_checkpoint()
        self.qTargetNet.load_checkpoint()
        
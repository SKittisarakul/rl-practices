# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 14:19:14 2023

@author: supit
"""

import gym
from DDQNAgent import DDQNAgent
import numpy as np

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    load_chkpt = False
    ddqn_agent = DDQNAgent(gamma = 0.99, lr = 0.0005, epsilon =1.0,
                           input_dim = [8], n_actions=[4],
                           batch_size = 64, mem_size=50_000)
    
    best_score = -np.inf
    
    if load_chkpt:
        ddqn_agent.load_model()
    # score & epsilon history
    score_hist, eps_hist = [], []
    
    n_games = 500
    
    for i in range (n_games):
        score = 0
        done = False
        obs = env.reset()
        
        while not done:
            action = ddqn_agent.choose_action(obs)
            new_obs, reward, done, _ = env.step(action)
            
            # increment score
            score += reward
            # store in mem if not load checkpoint
            if not load_chkpt:
                ddqn_agent.store_transition(obs, new_obs, action, reward, done)
                # learn
                ddqn_agent.learn()
                
            # iterate
            obs = new_obs
        score_hist.append(score)
        eps_hist.append(ddqn_agent.epsilon)
        
        # average acore
        avg_score = np.mean(score_hist[-100:])
        
        print('episode ',i,' score %.2f'%score, 'average score %.2f'%avg_score,
              'epsilon %.2f'%ddqn_agent.epsilon)
        
        if avg_score > best_score:
            best_score = avg_score
#----------------------------------------------        
import matplotlib.pyplot as plt
plt.plot(score_hist)
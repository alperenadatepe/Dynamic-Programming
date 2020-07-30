#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: alperenadatepe
"""

import numpy as np

class GridWorld():
    def __init__(self, delta_threshold, discount_rate, shape=[4, 4]):
        if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError("Shape argument must be a list or tuple with length of 2.")
        
        self.shape = shape
        self.delta_threshold = delta_threshold
        self.discount_rate = discount_rate
        
        self._create_actions()
        self._create_states()
        self._create_initials()
        
    def _create_states(self):        
        self.states = [(i, j) for i in range(0, self.shape[0]) for j in range(0, self.shape[1])]
        self.n_states = np.prod(self.shape)

        self.terminal_states = [(0, 0), (self.shape[0] - 1, self.shape[1] - 1)]
        
        self.state_transition = np.ones([self.n_states, self.n_actions]) / self.n_actions
    
    def _create_actions(self):        
        self.actions = ["Up", "Down", "Right", "Left"]
        self.n_actions = len(self.actions)
        
    def _create_initials(self):
        self.Policy = np.ones([self.n_states, self.n_actions]) / self.n_actions
        self.V = np.zeros((self.n_states))
        
    def state_hasher(self, state):
        return state[0] + state[1] * self.shape[1]
        
    def step(self, state, action):
        reward = 0
        old_state = state
        
        x_cord, y_cord = state
        
        if action == "Up":
            y_cord -= 1
            
            state = (x_cord, y_cord)
        elif action == "Down":
            y_cord += 1
            
            state = (x_cord, y_cord)
        elif action == "Right":
            x_cord += 1
            
            state = (x_cord, y_cord)
        elif action == "Left":
            x_cord -= 1
            
            state = (x_cord, y_cord)
        
        if state in self.states:
            
            if state in self.terminal_states:
                reward = 1
            else:
                reward = -1
        else:
            state = old_state
            reward = -1
        
        return state, reward
        
    def value_improve(self):
        while True:    
            delta = 0
            
            for state in self.states:
                hashed_state = self.state_hasher(state)
                
                value = self.V[hashed_state]
                
                expected_return_list = []
            
                for action in self.actions:
                    next_state, reward = self.step(state, action)
                
                    hashed_next_state = self.state_hasher(next_state)
                    action_index = self.actions.index(action)
                
                    expected_return = self.state_transition[hashed_state, action_index] * (reward + self.discount_rate * self.V[hashed_next_state])
                    expected_return_list.append(expected_return)
            
                best_expected_return = np.max(expected_return_list)
                self.V[hashed_state] = best_expected_return
                
                best_action_index = np.argmax(expected_return_list)
                self.Policy[hashed_state] = np.eye(self.n_actions)[best_action_index]
                
                delta = max(delta, np.absolute(value - self.V[hashed_state]))
                
            if delta < self.delta_threshold:
                break

    def value_iteration(self):
        self.value_improve()
        
        return self.V, self.Policy
        
    def print_policy(self):
        optimal_policy = np.reshape(np.argmax(self.Policy, axis=1), shape)
        print(f"\nThe optimal policy is:\n\n{optimal_policy}")
    
if __name__ == "__main__":
    delta_threshold = 1e-10
    discount_rate = 1
    shape = [4, 4]

    grid_world = GridWorld(delta_threshold, discount_rate, shape)

    V, Policy = grid_world.value_iteration()        
    grid_world.print_policy()
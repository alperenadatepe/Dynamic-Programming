#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: alperenadatepe
"""

import numpy as np
import matplotlib.pyplot as plt

class Gambler():
    def __init__(self, delta_threshold, discount_rate, probability_of_head, goal):
        if not probability_of_head <= 1.0 and probability_of_head >= 0:
            raise ValueError("Probability value must be between 0 and 1")
        
        self.delta_threshold = delta_threshold
        self.discount_rate = discount_rate
        
        self.probability_of_head = probability_of_head
        self.probability_of_tail = 1 - probability_of_head
        self.goal = goal
        
        self._create_states()
        self._create_initials()
        
    def _create_states(self):        
        self.states = [i for i in range(0, self.goal + 1)]
        self.n_states = len(self.states)
                    
    def _create_initials(self):
        self.Policy = np.zeros((self.n_states))
        self.V = np.zeros((self.n_states))
        
    def _create_actions(self, state):        
        self.actions = [i for i in range(0, min(state, self.goal - state) + 1)]
        self.n_actions = len(self.actions)
        
    def expected_return(self, state, action):
        expected_return = 0
        reward = 0
        
        next_state_for_winning_situation = int(min(state + action, self.goal))
        
        if next_state_for_winning_situation == self.goal:
            reward = 1
            expected_return += self.probability_of_head * (reward + self.discount_rate * self.V[next_state_for_winning_situation])
        else:
            reward = 0    
            expected_return += self.probability_of_head * (reward + self.discount_rate * self.V[next_state_for_winning_situation])
                
        reward = 0
        next_state_for_losing_situation = int(max(0, state - action))
        
        if next_state_for_losing_situation == 0:
            reward = 0
            expected_return += self.probability_of_tail * (reward + self.discount_rate * self.V[next_state_for_losing_situation])
        else:
            reward = 0    
            expected_return += self.probability_of_tail * (reward + self.discount_rate * self.V[next_state_for_losing_situation])
        
        return expected_return
    
    def choose_action_by_policy(self, state):
        action = self.Policy[state]
        
        return action
    
    def evaluate(self):
        while True:
            delta = 0
            
            for state in self.states[1:-1]:
                
                value = self.V[state]
    
                action = self.choose_action_by_policy(state)

                self.V[state] = self.expected_return(state, action)
                
                delta = max(delta, np.absolute(value - self.V[state]))
            
            if delta < self.delta_threshold:
                break
        
        return self.V
    
    def improve(self):
        policy_stable = True
        
        for state in self.states[1:-1]:
            
            old_action = self.choose_action_by_policy(state)
                        
            best_action_return = None
            best_action = -1
            
            self._create_actions(state)
            for action in self.actions:
                expected_return = self.expected_return(state, action)
                
                if best_action_return == None or best_action_return < expected_return:
                    best_action_return = expected_return
                    best_action = action
            
            self.Policy[state] = best_action
            self.V[state] = best_action_return
            
            if old_action != best_action:
                policy_stable = False
        
        return policy_stable
    
    def policy_iteration(self):
        self.evaluate()
        stable = self.improve()
        
        while not stable:
            self.evaluate()
            stable = self.improve()
        
        return self.V, self.Policy
    
    def plot_state_values(self):
        plt.figure("State Value Graph")
        plt.xlabel("Capital")
        plt.ylabel("State Value")
        plt.plot(self.V[1:self.goal])
        plt.show()
    
    def plot_policy(self):
        plt.figure("Policy Graph")
        plt.xlabel("Capital")
        plt.ylabel("Optimal Stake Values")
        plt.bar(range(self.goal - 1), self.Policy[1:self.goal], align='center', alpha=0.5)
        plt.show()
    
    def plot(self):
        self.plot_state_values()
        self.plot_policy()

if __name__ == "__main__":
    delta_threshold = 1e-10
    discount_rate = 1
    probability_of_head = 0.4
    goal = 100

    gambler = Gambler(delta_threshold, discount_rate, probability_of_head, goal)

    V, Policy = gambler.policy_iteration()   
    gambler.plot()
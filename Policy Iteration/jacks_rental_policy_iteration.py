#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: alperenadatepe
"""

import numpy as np
import seaborn as sns

class JacksCarRental():
    def __init__(self, delta_threshold, discount_rate, \
                 n_max_car, n_max_move,  \
                 loc_1_request_mean, loc_2_request_mean, loc_1_return_mean, loc_2_return_mean, \
                 moving_cost, rent_out_reward):
        
        self.delta_threshold = delta_threshold
        self.discount_rate = discount_rate
        
        self.n_max_car = n_max_car
        self.n_max_move = n_max_move
        
        self.loc_1_request_mean = loc_1_request_mean
        self.loc_2_request_mean = loc_2_request_mean
        self.loc_1_return_mean = loc_1_return_mean
        self.loc_2_return_mean = loc_2_return_mean
        
        self.moving_cost = moving_cost
        self.rent_out_reward = rent_out_reward
        
        self._create_actions()
        self._create_states()
        self._create_initials()
        
    def _create_actions(self):        
        self.actions = [i for i in range(-self.n_max_move, self.n_max_move + 1, 1)]
        self.n_actions = len(self.actions)

    def _create_states(self):        
        self.states = [(i, j) for i in range(0, self.n_max_car + 1) for j in range(0, self.n_max_car + 1)]
        self.n_states = len(self.states)            
        
    def _create_initials(self):
        self.Policy = np.zeros((self.n_max_car + 1, self.n_max_car + 1), dtype=int)
        
        self.V = np.zeros((self.n_max_car + 1, self.n_max_car + 1))
        
        self.loc_1_request_distribution, self.loc_1_request_lower_bound, self.loc_1_request_upper_bound = self._poisson_distribution(loc_1_request_mean)
        self.loc_2_request_distribution, self.loc_2_request_lower_bound, self.loc_2_request_upper_bound = self._poisson_distribution(loc_2_request_mean)
        self.loc_1_return_distribution, self.loc_1_return_lower_bound, self.loc_1_return_upper_bound = self._poisson_distribution(loc_1_return_mean)
        self.loc_2_return_distribution, self.loc_2_return_lower_bound, self.loc_2_return_upper_bound = self._poisson_distribution(loc_2_return_mean)
        
    def _poisson_probability(self, lamb, n):
        return ( lamb ** n / np.math.factorial(n) ) * np.e ** (-lamb)
    
    def _poisson_distribution(self, lamb):
        lower_bound = 0
        upper_bound = 0
        
        for_lower_part = True
        distribution_threshold = 0.01
        
        probabilities_sum = 0

        distribution_values = {}
        while True:
            if for_lower_part:
                probability = self._poisson_probability(lamb, lower_bound)
                
                if probability <= distribution_threshold:
                    lower_bound += 1
                else:
                    distribution_values[lower_bound] = probability
                    probabilities_sum += probability
                    upper_bound = lower_bound + 1
                    for_lower_part = False
            else:
                probability = self._poisson_probability(lamb, upper_bound)
                
                if probability > distribution_threshold:
                    distribution_values[upper_bound] = probability
                    probabilities_sum += probability
                    upper_bound += 1
                else:
                    break
                
        excess_value = (1 - probabilities_sum) / (upper_bound - lower_bound)
        
        for key in distribution_values:
            distribution_values[key] += excess_value
        
        return distribution_values, lower_bound, upper_bound
    
    def expected_return(self, state, action):
        
        expected_return = 0

        car_at_location_1 = state[0]
        car_at_location_2 = state[1]
        
        car_at_location_1 = max(min(car_at_location_1 - action, self.n_max_car), 0)
        car_at_location_2 = max(min(car_at_location_2 + action, self.n_max_car), 0)
        
        expected_return += self.moving_cost * abs(action)
                
        for loc_1_request in range(self.loc_1_request_lower_bound, self.loc_1_request_upper_bound):
            for loc_2_request in range(self.loc_2_request_lower_bound, self.loc_2_request_upper_bound):
                for loc_1_return in range(self.loc_1_return_lower_bound, self.loc_1_return_upper_bound):
                    for loc_2_return in range(self.loc_2_return_lower_bound, self.loc_2_return_upper_bound):
                            
                        request_probability = self.loc_1_request_distribution[loc_1_request] * self.loc_2_request_distribution[loc_2_request] 
                        return_probability = self.loc_1_return_distribution[loc_1_return] * self.loc_2_return_distribution[loc_2_return] 
                        transition_probability = request_probability * return_probability
                        
                        rent_at_location_1 = min(car_at_location_1, loc_1_request)
                        rent_at_location_2 = min(car_at_location_2, loc_2_request)
                        
                        reward = (rent_at_location_1 + rent_at_location_2) * self.rent_out_reward
                        
                        car_at_loc_1_after = max(min(car_at_location_1 - rent_at_location_1 + loc_1_return, self.n_max_car), 0)
                        car_at_loc_2_after = max(min(car_at_location_2 - rent_at_location_2 + loc_2_return, self.n_max_car), 0)
                        
                        next_state = (car_at_loc_1_after, car_at_loc_2_after)
                        
                        expected_return += transition_probability * (reward + self.discount_rate * self.V[next_state])
             
        return expected_return
                
    def choose_action_by_policy(self, state):
        action = self.Policy[state]
        
        return action
        
    def evaluate(self):
        while True:
            delta = 0
            
            for state in self.states:
                
                value = self.V[state]
    
                action = self.choose_action_by_policy(state)
                
                expected_return = self.expected_return(state, action)

                self.V[state] = expected_return
                
                delta = max(delta, np.absolute(value - self.V[state]))
            
            if delta < self.delta_threshold:
                break
        
        return self.V
    
    def improve(self):
        policy_stable = True
        
        for state in self.states:
            
            old_action = self.choose_action_by_policy(state)
                                    
            best_action_return = None
            best_action = -1
            
            action_loc_1_to_2 = min(state[0], self.n_max_move)
            action_loc_2_to_1 = -min(state[1], self.n_max_move)
            
            for action in range(action_loc_2_to_1, action_loc_1_to_2 + 1):
                
                expected_return = self.expected_return(state, action)
                
                if best_action_return == None or best_action_return < expected_return:
                    best_action_return = expected_return
                    best_action = action
            
            self.Policy[state] = best_action
            
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
    
    def plot_policy(self):
        ax = sns.heatmap(self.Policy, linewidth=0.5)
        ax.invert_yaxis()
                
if __name__ == "__main__":
    delta_threshold = 1e-4
    discount_rate = 0.9
    
    n_max_car = 20
    n_max_move = 5
    
    loc_1_request_mean = 3 
    loc_2_request_mean = 4
    loc_1_return_mean = 3 
    loc_2_return_mean = 2
    
    moving_cost = -2
    rent_out_reward = 10

    jacks_car_rental = JacksCarRental(delta_threshold, discount_rate, \
                             n_max_car, n_max_move,  \
                             loc_1_request_mean, loc_2_request_mean, loc_1_return_mean, loc_2_return_mean, \
                             moving_cost, rent_out_reward)

    V, Policy = jacks_car_rental.policy_iteration()        
    jacks_car_rental.plot_policy()
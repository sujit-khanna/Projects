__author__ = 'Sujit Khanna'
"""
Author: Sujit Khanna sk2913
This file contains code for the first assignment
"""
'''
This file contains the implementation of the Firefly Optimizer
'''

import numpy as np
import math as m
import random
from objective_func import *


class init_firefly:

    def __init__(self, obj, obj_dim,num_fireflies, ceiling, floor, seed, **kwargs):
        self.obj_dim = obj_dim
        self.objective = obj
        self.num_fireflies = num_fireflies
        self.ceiling = ceiling
        self.floor = floor
        self.seed = seed
        self.init_pop = self.initialize_population()
        self.intensity = self.calc_intensity()

    def calc_intensity(self):
        tmp = self.init_pop
        tmp_1 = self.init_pop.shape
        pop_intensity = np.asarray(np.zeros(self.init_pop.shape[0])).astype(float)
        for i in range(len(pop_intensity)):
            pop_intensity[i] = self.objective(self.init_pop[i])
        return pop_intensity


    def initialize_population(self):
        random.seed(self.seed)
        pop_list = np.asarray(np.zeros((self.num_fireflies, self.obj_dim))).astype(float)
        for i in range(self.num_fireflies):
            pop_list[i] = np.array([random.uniform(self.floor, self.ceiling) for _ in range(self.obj_dim)])
        return pop_list


class firefly:

    def __init__(self, obj, num_agents, num_gen, seed, obj_dim, alpha = 0.2, alpha_decay = 0.03, beta_0=1, gamma=0.97,floor=-5, ceiling= 5,**kwargs):
        self.obj = obj
        self.num_agents = num_agents
        self.population = None
        self.num_gen = num_gen
        self.seed = seed
        self.obj_dim = obj_dim
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.gamma=gamma
        self.beta0=beta_0
        self.beta = self.beta0
        self.randomize = None
        self.runner = None
        self.floor = floor
        self.ceiling = ceiling
        self.I_0 = None
        self.current_best=None

    def new_intensity(self, param):
        return self.obj(param)

    def init_optimization(self):
        self.runner = init_firefly(self.obj, self.obj_dim, self.num_agents, self.ceiling, self.floor, self.seed)
        self.population=self.runner.init_pop
        self.I_0 = self.runner.intensity

    def run_optimization(self):
        return None

    def custom_sort(self, arr):
        l = list(arr)
        l.sort(key=lambda crunch: self.new_intensity(crunch))
        sort_ary = np.asarray(l)
        return sort_ary

    def model_fit(self):
        self.init_optimization()
        for n in range(self.num_gen):
            for i in range(len(self.population)):
                tmp_param = np.copy(self.population[i])
                for j in range(len(self.population)):
                    if self.I_0[j]>self.I_0[i]:
                        r_ij = m.sqrt(np.sum((self.population[i] - self.population[j])**2))
                        self.beta = (self.beta/(1 + self.gamma*(r_ij**2)))
                        self.randomize = self.alpha*np.array([random.uniform(self.floor, self.ceiling) for _ in range(self.obj_dim)])
                        self.population[i] = self.population[i] + self.beta*(self.population[j] - self.population[i]) + self.randomize
                        self.I_0[i] = (self.I_0[i])*m.exp(-self.gamma*(r_ij**2))
                g_1, g_2 = self.new_intensity(tmp_param),self.new_intensity(self.population[i])
                if self.new_intensity(tmp_param)>self.new_intensity(self.population[i]):
                    self.population[i] = tmp_param

            self.alpha = self.alpha*(1-self.alpha_decay)
            self.population = self.custom_sort(self.population)
            self.current_best = self.population[len(self.population) - 1]

        return self.current_best


    def four_peak(self, params):
        x, y = params[0], params[1]
        obj_func = m.exp(-(x - 4)**2 - (y - 4)**2) + m.exp(-(x + 4)**2 - (y - 4)**2) + 2*( m.exp(-(x)**2 - (y)**2) + m.exp(-(x)**2 - (y+ 4)**2))
        return obj_func

if __name__ == '__main__':
    params = np.array([2,2])
    tmp_1 = params.shape
    obj = ackley(params)
    #tmp = obj.return_value

    model = firefly(obj.ackley_func,num_agents=50, num_gen=1800, seed=7, obj_dim=params.shape[0])
    best_param = model.model_fit()
    value = obj.return_value(best_param)
    print(float(value))
    print(best_param)
    print('end here ')
    '''
    seed=100; gives local minima on (4,4)
    seed = 10 gives local minima on (-4, 4)
    
    '''


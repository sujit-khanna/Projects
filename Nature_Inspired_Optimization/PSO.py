__author__ = 'Sujit Khanna'
"""
Author: Sujit Khanna sk2913
This file contains code for the first assignment
"""
'''
This file contains the implementation of both PSO and APSO Optimizers
'''

import numpy as np
import math as m
import random
from objective_func import *


class init_PSO:

    def __init__(self, obj, obj_dim,num_fireflies, ceiling, floor, seed, **kwargs):
        self.obj_dim = obj_dim
        self.objective = obj
        self.num_fireflies = num_fireflies
        self.ceiling = ceiling
        self.floor = floor
        self.seed = seed
        self.init_pop = self.initialize_population()
        self.fitness = self.calc_fitness()

    def calc_fitness(self):
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


class PSO:

    def __init__(self, obj, class_obj,num_agents, num_gen, seed, obj_dim, alpha = 2, alpha_decay=0.03, beta=2,floor = -5,celiling=5 ,**kwargs):
        self.obj = obj
        self.num_agents = num_agents
        self.population = None
        self.num_gen = num_gen
        self.seed = seed
        self.obj_dim = obj_dim
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.beta = beta
        self.class_obj = class_obj

        self.randomize = None
        self.runner = None
        self.floor = floor
        self.ceiling = celiling
        #self.floor = kwargs.get('floor', -5)
        #self.ceiling = kwargs.get('ceiling', 5)
        self.fitness = None
        self.current_best=None
        self.current_best_particle=None
        self.velocities = None
        self.err_tracker = []

        '''
        Include a host of other parameters; required to test out the firefly algorithm
        '''

    def new_intensity(self, param):
        return self.obj(param)

    def init_optimization(self):
        self.runner = init_PSO(self.obj, self.obj_dim, self.num_agents, self.ceiling, self.floor, self.seed)
        self.population=self.runner.init_pop
        self.fitness = self.runner.fitness

    def init_velocities(self):
        return None

    def run_optimization(self):
        return None

    def custom_sort(self, arr):
        l = list(arr)
        l.sort(key=lambda crunch: self.new_intensity(crunch))
        sort_ary = np.asarray(l)
        return sort_ary

    def model_fit(self):
        self.init_optimization()
        self.current_best = self.custom_sort(self.population)[len(self.population) - 1]
        self.current_best_particle=self.population
        self.velocities =np.asarray(np.zeros(self.population.shape)).astype(float)
        for n in range(self.num_gen):
            for i in range(len(self.population)):
                tmp_best = np.copy(self.current_best_particle[i])
                self.velocities[i] += (self.alpha*random.uniform(0, 1))*(self.current_best - self.population[i]) + \
                                     (self.beta*random.uniform(0, 1))*(self.current_best_particle[i] - self.population[i])
                self.population[i] += self.velocities[i]
                if self.new_intensity(self.population[i]) > self.new_intensity(tmp_best):
                    self.current_best_particle[i] = self.population[i]
                else:
                    self.current_best_particle[i] = tmp_best
            self.current_best = self.custom_sort(self.population)[len(self.population) - 1]
            self.err_tracker.append(abs(self.class_obj.return_value(self.current_best)  - self.class_obj.min))
        return self.current_best


class Accelerated_PSO:

    def __init__(self, obj, class_obj,num_agents, num_gen, seed, obj_dim,alpha = 1, alpha_decay = 0.03,beta=0.7, floor=-5, ceiling = 5 ,**kwargs):
        self.obj = obj
        self.num_agents = num_agents
        self.population = None
        self.num_gen = num_gen
        self.seed = seed
        self.obj_dim = obj_dim
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.beta = beta
        self.class_obj = class_obj

        self.randomize = None
        self.runner = None
        self.floor = floor
        self.ceiling = ceiling
        #self.floor = kwargs.get('floor', -5)
        #self.ceiling = kwargs.get('ceiling', 5)
        self.fitness = None
        self.current_best=None
        self.current_best_particle=None
        self.velocities = None
        self.err_tracker = []

        '''
        Include a host of other parameters; required to test out the firefly algorithm
        '''

    def new_intensity(self, param):
        return self.obj(param)

    def init_optimization(self):
        self.runner = init_PSO(self.obj, self.obj_dim, self.num_agents, self.ceiling, self.floor, self.seed)
        self.population=self.runner.init_pop
        self.fitness = self.runner.fitness

    def init_velocities(self):
        return None


    def run_optimization(self):
        return None

    def custom_sort(self, arr):
        l = list(arr)
        l.sort(key=lambda crunch: self.new_intensity(crunch))
        sort_ary = np.asarray(l)
        return sort_ary



    def model_fit(self):
        self.init_optimization()
        self.current_best = self.custom_sort(self.population)[0]
        self.current_best_particle=self.population
        self.velocities =np.asarray(np.zeros(self.population.shape)).astype(float)
        for n in range(self.num_gen):
            for i in range(len(self.population)):
                tmp_best = np.copy(self.population[i])
                self.velocities[i] += (self.alpha*(random.uniform(0, 1) - 0.5)) + \
                                      (self.beta)*(self.current_best - self.population[i])
                self.population[i] += self.velocities[i]
                if self.new_intensity(self.population[i]) > self.new_intensity(tmp_best):
                    self.current_best_particle[i] = self.population[i]
                else:
                    self.current_best_particle[i] = tmp_best
            self.current_best = self.custom_sort(self.population)[len(self.population) - 1]
            self.err_tracker.append(abs(self.class_obj.return_value(self.current_best)  - self.class_obj.min))
            self.alpha=self.alpha*m.exp(-self.alpha_decay*n)
        return self.current_best



if __name__ == '__main__':

    '''basic testing of PSO Optimizer'''

    params = np.array([0, 0])
    tmp_1 = params.shape
    obj = easom(params)
    model = PSO(obj.easom_func, obj,num_agents=60, num_gen=1500,seed=10,obj_dim=params.shape[0])
    best_param = model.model_fit()
    print("PSO Results")
    print(best_param)
    value = obj.return_value(best_param)
    #global_val_1 = tmp([0, 0])
    #global_val_2 = tmp([-0.7, -4])
    print(value)
    print('PSO ends here ')

    '''basic testing of PSO Optimizer'''

    model = Accelerated_PSO(obj.easom_func, obj,num_agents=60, num_gen=500,seed=7,obj_dim=params.shape[0])
    best_param = model.model_fit()
    print("Accelerated PSO Results")
    print(best_param)
    value = obj.return_value(best_param)
    #global_val_1 = tmp([0, 0])
    #global_val_2 = tmp([0, -4])
    print(value)
    print('Accelerated PSO ends here ')

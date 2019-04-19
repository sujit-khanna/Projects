__author__ = 'Sujit Khanna'
"""
Author: Sujit Khanna sk2913
This file contains code for the first assignment
"""
'''
This file contains the implementation of the Harmony Search Algorithm
'''

import numpy as np
import math as m
import random
from objective_func import four_peak

class init_CS:

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

class CuckooSearch:

    def __init__(self, obj, class_obj,num_agents, num_gen, seed, obj_dim, alpha=0.01, p=0.25,beta=1.5, floor=-5, ceiling = 5,**kwargs):
        self.obj = obj
        self.num_agents = num_agents
        self.class_obj = class_obj
        self.population = None
        self.num_gen = num_gen
        self.seed = seed
        self.obj_dim = obj_dim
        self.alpha = alpha
        self.p = p
        self.beta = beta
        self.randomize = None
        self.runner = None
        self.floor = floor
        self.ceiling = ceiling
        self.fitness = None
        self.current_best=None
        self.current_best_particle=None
        self.velocities = None
        self.param_bound = [self.floor, self.ceiling]
        self.levy = None
        self.sigma_v, self.sigma_u = None, None
        self.err_tracker = []

    def new_intensity(self, param):
        return self.obj(param)

    def init_optimization(self):
        self.runner = init_CS(self.obj, self.obj_dim, self.num_agents, self.ceiling, self.floor, self.seed)
        self.population=self.runner.init_pop
        self.fitness = self.runner.fitness

    def scale_params(self, param_list):
        for i in range(len(param_list)):
            if param_list[i]>self.param_bound[1]:
                param_list[i] = self.param_bound[1]
            elif param_list[i]<self.param_bound[0]:
                param_list[i] = self.param_bound[0]

        return param_list

    def Calc_Levy_Flights(self):
        self.sigma_u = (m.gamma(1+self.beta)*m.sin(m.pi*self.beta/2)/(m.gamma((1+self.beta)/2)*self.beta*2**((self.beta-1)/2)))**(1/self.beta)
        self.sigma_v = 1

        for i in range(len(self.population)):
            U = np.random.normal(0, self.sigma_u, size=self.obj_dim)
            V = np.random.normal(0, 1, size=self.obj_dim)
            s = U/(abs(V)**(1/self.beta))
            self.population[i]+=self.alpha*s*(self.population[i]-self.current_best)*np.random.randn(len(self.population[i]))
            self.population[i]=self.scale_params(self.population[i])

    def run_optimization(self):
        return None

    def custom_sort(self, arr):
        l = list(arr)
        l.sort(key=lambda crunch: self.new_intensity(crunch))
        sort_ary = np.asarray(l)
        return sort_ary

    def update_population(self, tmp_population):
        for i in range(len(self.population)):
            if self.new_intensity(tmp_population[i])>self.new_intensity(self.population[i]):
                self.population[i] = tmp_population[i]
        self.current_best = self.custom_sort(self.population)[len(self.population) - 1]


    def bad_nests(self):
        tmp_population = np.copy(self.population)
        for i in range(len(self.population)):
            if np.random.uniform(0, 1)<self.p:
                [k1, k2] = random.sample(list(self.population), 2)
                x_diff = [k1_i - k2_i for k1_i, k2_i in zip(k1, k2)]
                '''
                [k1, k2, k3, k4] = random.sample(list(self.population), 4)
                x_diff_1 = [k1_i - k2_i for k1_i, k2_i in zip(k1, k2)]
                x_diff_2 = [k3_i - k4_i for k3_i, k4_i in zip(k3, k4)]

                self.velocity = np.array([best_i + self.F*(x_diff_1_i + x_diff_2_i) for best_i, x_diff_1_i, x_diff_2_i in zip(self.current_best, x_diff_1, x_diff_2)])
                '''
                add_val = np.array([self.alpha*random.uniform(0, 1)*x_diff_i for x_diff_i in x_diff])
                self.population[i]+=add_val


    def model_fit(self):
        self.init_optimization()
        self.current_best = self.custom_sort(self.population)[len(self.population) - 1]
        for n in range(self.num_gen):
            tmp_population = np.copy(self.population)
            self.Calc_Levy_Flights()
            self.update_population(tmp_population)
            tmp_population_2 = np.copy(self.population)
            self.bad_nests()
            self.update_population(tmp_population_2)
            self.err_tracker.append(abs(self.class_obj.return_value(self.current_best)  - self.class_obj.min))
        return self.current_best



if __name__ == '__main__':
    '''basic testing of Cuckoo Search'''
    params = np.array([0, 0])
    tmp_1 = params.shape
    obj = four_peak(params)
    tmp = obj.four_peak_func
    model = CuckooSearch(obj.four_peak_func, obj,num_agents=50, num_gen=100, seed=100,obj_dim=params.shape[0])
    best_param = model.model_fit()
    value = tmp(best_param)
    global_val_1 = tmp([0, 0])
    global_val_2 = tmp([0, -4])
    print(value)
    print(best_param)
    print('end here')





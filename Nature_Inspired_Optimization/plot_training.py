__author__ = 'Sujit Khanna'
"""
Author: Sujit Khanna sk2913
This file contains code for the first assignment
"""
'''
This file contains the code to plot the training error propogation for DE,PSO,APSO and Cuckoo Search 
'''

import numpy as np
import math as m
import random
from FireFly import *
from PSO import *
from Differential_Evolution import *
from Cuckoo_Search import *
from objective_func import *
import statistics
import pandas as pd
from matplotlib import pyplot as plt



def list_DE():
    scaling = 0.3
    crossover = 0.40
    pop_size, num_gen, num_trials = 60, 1500, 30
    params = np.array([2, 2])
    obj = ackley(params)
    obj_func = obj.ackley_func


    train_err = []
    for i in range(num_trials):
        set_seed = random.randint(1, 1000)
        model = Differential_Evolution(obj_func, obj,num_agents= pop_size, num_gen=num_gen, scaling=scaling,seed= set_seed,cross_prob = crossover, obj_dim = params.shape[0],floor = obj.floor, ceiling=obj.ceiling)
        best_param = model.model_fit()
        best_val = obj.return_value(best_param)
        print(best_val)
        err = abs(best_val - obj.min)
        train_err.append(model.err_tracker)

    df = pd.DataFrame(train_err)
    df_ary = np.asarray(df).astype(float)
    mean_ary, var_ary = np.mean(df_ary, axis=0), np.std(df_ary, axis=0)


    return mean_ary, var_ary


def list_PSO():
    alpha = 1.5
    alpha_decay = 0.003
    beta = 1.5
    pop_size, num_gen, num_trials = 40, 1500, 30
    params = np.array([2, 2])
    obj = ackley(params)
    obj_func = obj.ackley_func


    train_err = []
    for i in range(num_trials):
        set_seed = random.randint(1, 1000)
        model = PSO(obj_func, obj ,num_agents= pop_size, num_gen=num_gen, seed= set_seed, obj_dim = params.shape[0],
                    alpha=alpha,alpha_decay=alpha_decay,beta = beta,floor = obj.floor, ceiling=obj.ceiling)

        best_param = model.model_fit()
        best_val = obj.return_value(best_param)
        print(best_val)
        err = abs(best_val - obj.min)
        train_err.append(model.err_tracker)

    df = pd.DataFrame(train_err)
    df_ary = np.asarray(df).astype(float)
    mean_ary, var_ary = np.mean(df_ary, axis=0), np.std(df_ary, axis=0)


    return mean_ary, var_ary


def list_Acc_PSO():

    alpha = 0.1
    alpha_decay = 0
    beta = 0.1
    pop_size, num_gen, num_trials = 60, 1500, 30
    params = np.array([2, 2])
    obj = ackley(params)
    obj_func = obj.ackley_func


    train_err = []
    for i in range(num_trials):
        set_seed = random.randint(1, 1000)
        model = Accelerated_PSO(obj_func, obj ,num_agents= pop_size, num_gen=num_gen, seed= set_seed, obj_dim = params.shape[0],
                    alpha=alpha,alpha_decay=alpha_decay,beta = beta,floor = obj.floor, ceiling=obj.ceiling)

        best_param = model.model_fit()
        best_val = obj.return_value(best_param)
        print(best_val)
        err = abs(best_val - obj.min)
        train_err.append(model.err_tracker)

    df = pd.DataFrame(train_err)
    df_ary = np.asarray(df).astype(float)
    mean_ary, var_ary = np.mean(df_ary, axis=0), np.std(df_ary, axis=0)

    return mean_ary, var_ary

def list_cs():
    alpha = 0.01
    p = 0.25
    beta = 1.5
    pop_size, num_gen, num_trials = 60, 1500, 30
    params = np.array([2, 2])
    obj = ackley(params)
    obj_func = obj.ackley_func

    train_err = []
    for i in range(num_trials):
        set_seed = random.randint(1, 1000)
        model = CuckooSearch(obj_func, obj ,num_agents= pop_size, num_gen=num_gen, seed= set_seed, obj_dim = params.shape[0],
                                alpha=alpha,p=p,beta = beta,floor = obj.floor, ceiling=obj.ceiling)

        best_param = model.model_fit()
        best_val = obj.return_value(best_param)
        print(best_val)
        err = abs(best_val - obj.min)
        train_err.append(model.err_tracker)

    df = pd.DataFrame(train_err)
    df_ary = np.asarray(df).astype(float)
    mean_ary, var_ary = np.mean(df_ary, axis=0), np.std(df_ary, axis=0)
    return mean_ary, var_ary




if __name__ == '__main__':
    '''The code snippet will plot the training curves of DE,PSO and APSO together,
       and at the same time it will create another plot for Cuckoo Search
    '''
    num_gen = 1500
    mean_cs, var_cs = list_cs()
    mean_de, var_de = list_DE()
    mean_pso, var_pso = list_PSO()
    mean_apso, var_apso = list_Acc_PSO()
    x = np.linspace(0, num_gen, num_gen)
    plt.plot(x, mean_de, color='#CC4F1B', label='Differential_Evolution')
    plt.legend()
    plt.fill_between(x, mean_de - var_de, mean_de + var_de, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    plt.plot(x, mean_pso, color='#1B2ACC', label='PSO')
    plt.legend()
    plt.fill_between(x, mean_pso - var_pso, mean_pso + var_pso, alpha=0.5, edgecolor='#1B2ACC', facecolor='#089FFF',
                     linestyle='dashdot', antialiased=True)
    plt.plot(x, mean_apso, color='#3F7F4C', label='Accelerated_PSO')
    plt.legend()
    plt.fill_between(x, mean_apso - var_apso, mean_apso + var_apso, alpha=0.5, edgecolor='#3F7F4C', facecolor='#7EFF99')
    plt.savefig('plot_training_err')
    print('hold')
    plt.close()
    plt.plot(x, mean_cs, color='#CC4F1B', label='Cuckoo_search')
    plt.legend()
    plt.fill_between(x, mean_cs - var_cs, mean_cs + var_cs, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    plt.savefig('plot_training_err_Cuckoo')
    plt.close()




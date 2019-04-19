__author__ = 'Sujit Khanna'
"""
Author: Sujit Khanna sk2913
This file contains code for the first assignment
"""
'''
This file contains all test cases for Firefly, Differential_Evolution, PSO, APSO and Cuckoo Search Algorithms
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

NUM_TRIALS=3
def test_3A():

    alpha_list = [0.1, 0.3, 0.5, 0.8]
    beta_list = [0.5, 0.7, 1, 1.2]
    gamma_list = [0.01, 0.1, 1, 10]
    #pop_size, num_gen, num_trials = [10, 25, 50], 500, 30
    pop_size, num_gen, num_trials = [10, 25, 50], 500, NUM_TRIALS
    params = np.array([2, 2])
    tmp_1 = params.shape
    obj = four_peak(params)
    tmp = obj.four_peak_func

    res_std_err, res_mean = [[] for i in range(len(alpha_list)*len(pop_size))], [[] for i in range(len(alpha_list)*len(pop_size))]
    global_best = [[] for i in range(len(alpha_list)*len(pop_size))]
    l = -1
    for a,b,g in zip(alpha_list, beta_list, gamma_list):
        for pop in pop_size:
            l+=1
            header = 'alpha_' + str(a) + '_beta_' + str(b) + 'gamma_' + str(g) + '_pop_size_' + str(pop)
            res_std_err[l].append(header), res_mean[l].append(header)
            for i in range(num_trials):
                set_seed = random.randint(1, 1000)
                model = firefly(obj.four_peak_func, num_agents=pop, num_gen=num_gen, seed=set_seed ,obj_dim=params.shape[0], alpha=a,beta_0=b, gamma=g,
                                floor = obj.floor, ceiling=obj.ceiling)
                best_param = model.model_fit()
                best_val = obj.return_value(best_param)
                err = abs(best_val - obj.max)
                res_std_err[l].append(err), res_mean[l].append(best_val)
                tmp_param = []
                for h in range(len(best_param)):
                    tmp_param.append(best_param[h])

                global_best[l].append(tmp_param)
    best_gstar =[]
    for err,best in zip(res_std_err, global_best):
        idx = np.argmin(np.asarray(err[1:]))
        best_gstar.append(best[idx])


    '''Calculate mean and std err for all the values'''
    fin_res = []
    config, val_mean, val_err, err_mean, err_std,  =  [], [], [], [], []
    for mean, err,g in zip(res_mean, res_std_err, global_best):
        config.append(mean[0])
        val_err.append((statistics.stdev(mean[1:])/num_trials)), val_mean.append(statistics.mean(mean[1:]))
        err_std.append((statistics.stdev(err[1:])/num_trials)), err_mean.append(statistics.mean(err[1:]))

    tempor = [config, val_mean, val_err, err_mean, err_std, best_gstar]
    fin_res.append(tempor)
    return tempor

def test_3B():

    alpha_list = [0.1, 0.3, 0.5, 0.8]
    beta_list = [0.5, 0.7, 1, 1.2]
    gamma_list = [0.01, 0.1, 1, 10]
    #pop_size, num_gen, num_trials = [10, 25, 50], 500, 30
    pop_size, num_gen, num_trials = [25, 50, 75], 500, NUM_TRIALS
    params_1, params_2, params_3 = np.array([2, 2]), np.array([2, 2]), np.array([2, 2, 2, 2, 2, 2, 2, 2]) ###change this

    obj_1, obj_2, obj_3 = egg_crate(params_1), one_dim_func(params_2), one_dim_func(params_3)
    obj_1_func, obj_2_func, obj_3_func = obj_1.egg_crate_func, obj_2.sphere_func, obj_3.sphere_func
    obj_name = ['egg_crate', '2D_Sphere_func', '8D_Sphere_func']
    obj_list, func_list, param_list = [obj_1, obj_2, obj_3], [obj_1_func, obj_2_func, obj_3_func],[params_1, params_2, params_3]

    fin_res = [[] for i in range(len(obj_list))]
    bar=-1
    for obj, func, params in zip(obj_list, func_list, param_list):
        bar+=1
        res_std_err, res_mean = [[] for i in range(len(alpha_list)*len(pop_size))], [[] for i in range(len(alpha_list)*len(pop_size))]
        global_best = [[] for i in range(len(alpha_list)*len(pop_size))]
        l = -1
        for a,b,g in zip(alpha_list, beta_list, gamma_list):
            for pop in pop_size:
                l+=1
                header =obj_name[bar] + '_alpha_' + str(a) + '_beta_' + str(b) + 'gamma_' + str(g) + '_pop_size_' + str(pop)
                print(header)
                res_std_err[l].append(header), res_mean[l].append(header)
                for i in range(num_trials):
                    set_seed = random.randint(1, 1000)
                    model = firefly(func, num_agents=pop, num_gen=num_gen, seed=set_seed ,obj_dim=params.shape[0], alpha=a,beta_0=b, gamma=g,
                                    floor = obj.floor, ceiling=obj.ceiling)
                    best_param = model.model_fit()
                    best_val = obj.return_value(best_param)
                    err = abs(best_val - obj.min)
                    res_std_err[l].append(err), res_mean[l].append(best_val)
                    tmp_param = []
                    for h in range(len(best_param)):
                        tmp_param.append(best_param[h])

                    global_best[l].append(tmp_param)

        best_gstar =[]
        for err,best in zip(res_std_err, global_best):
            idx = np.argmin(np.asarray(err[1:]))
            best_gstar.append(best[idx])

        '''Calculate mean and std err for all the values'''
        config, val_mean, val_err, err_mean, err_std = [], [], [], [], []
        for mean, err in zip(res_mean, res_std_err):
            config.append(mean[0])
            val_err.append((statistics.stdev(mean[1:])/num_trials)), val_mean.append(statistics.mean(mean[1:]))
            err_std.append((statistics.stdev(err[1:])/num_trials)), err_mean.append(statistics.mean(err[1:]))
        fin_res[bar].append([config, val_mean, val_err, err_mean, err_std, best_gstar])

    return fin_res


def test_differential_evolution():

    scaling_list = [0.3, 0.5, 0.7, 0.9]
    crossover_list = [0.4,0.50,0.6, 0.7]
    gamma_list = [0.01, 0.1, 1, 10]
    pop_size, num_gen, num_trials = [20, 40, 60], 500, NUM_TRIALS
    params_1, params_2, params_3, params_4, params_5 = np.array([2, 2]), np.array([2, 2, 2, 2, 2, 2, 2, 2]), np.array([2, 2]), \
                                   np.array([2, 2]), np.array([2, 2, 2, 2, 2, 2, 2, 2]) ###change this

    obj_1, obj_2, obj_3, obj_4, obj_5 = ackley(params_1), ackley(params_2), easom(params_3), rosenbrock(params_4), rosenbrock(params_5)
    obj_1_func, obj_2_func, obj_3_func, obj_4_func, obj_5_func = obj_1.ackley_func, obj_2.ackley_func, obj_3.easom_func,obj_4.rosenbrock_func, obj_5.rosenbrock_func
    obj_list, func_list = [obj_1, obj_2, obj_3, obj_4, obj_5], [obj_1_func, obj_2_func, obj_3_func, obj_4_func, obj_5_func]
    obj_name, params_list = ['2DAckley', '8DAckely', 'Easom', '2DRoesnbrock', '8DRoesnbrock'], [params_1, params_2, params_3, params_4, params_5]


    fin_res = [[] for i in range(len(obj_list))]
    bar=-1
    for obj, func, params in zip(obj_list, func_list, params_list):
        bar +=1
        res_std_err, res_mean = [[] for i in range(len(pop_size)*len(scaling_list))], [[] for i in range(len(pop_size)*len(scaling_list))]
        global_best, training_err = [[] for i in range(len(pop_size)*len(scaling_list))], [[] for i in range(len(pop_size)*len(scaling_list))]
        l=-1
        for scaling, crossover in zip(scaling_list, crossover_list):
            for pop in pop_size:
                l+=1
                header = obj_name[bar] + 'F_' + str(scaling) + 'Cr_Prob' + str(crossover) + 'num_agents_' + str(pop)
                print(header)
                res_std_err[l].append(header), res_mean[l].append(header)
                for i in range(num_trials):
                    set_seed = random.randint(1, 1000)
                    model = Differential_Evolution(func, obj,num_agents= pop, num_gen=num_gen, scaling=scaling,seed= set_seed,cross_prob = crossover, obj_dim = params.shape[0],floor = obj.floor, ceiling=obj.ceiling)
                    best_param = model.model_fit()
                    best_val = obj.return_value(best_param)
                    training_err[l].append(model.err_tracker)
                    print(best_val)
                    err = abs(best_val - obj.min)
                    res_std_err[l].append(err), res_mean[l].append(best_val)
                    tmp_param = []
                    for h in range(len(best_param)):
                        tmp_param.append(best_param[h])

                    global_best[l].append(tmp_param)
        best_gstar =[]
        for err,best in zip(res_std_err, global_best):
            idx = np.argmin(np.asarray(err[1:]))
            best_gstar.append(best[idx])

        '''Calculate mean and std err for all the values'''
        config, val_mean, val_err, err_mean, err_std = [], [], [], [], []
        for mean, err in zip(res_mean, res_std_err):
            config.append(mean[0])
            val_err.append((statistics.stdev(mean[1:])/num_trials)), val_mean.append(statistics.mean(mean[1:]))
            err_std.append((statistics.stdev(err[1:])/num_trials)), err_mean.append(statistics.mean(err[1:]))
        fin_res[bar].append([config, val_mean, val_err, err_mean, err_std, best_gstar])

    return fin_res



def test_pso():

    alpha_list = [1, 1.5, 2, 2.5]
    alpha_decay = 0
    beta_list = [1, 1.5, 2, 2.5]
    gamma_list = [0.01, 0.1, 1, 10]
    pop_size, num_gen, num_trials = [20, 40, 60], 500, NUM_TRIALS
    params_1, params_2, params_3, params_4, params_5 = np.array([2, 2]), np.array([2, 2, 2, 2, 2, 2, 2, 2]), np.array([2, 2]), \
                                                       np.array([2, 2]), np.array([2, 2, 2, 2, 2, 2, 2, 2]) ###change this

    obj_1, obj_2, obj_3, obj_4, obj_5 = ackley(params_1), ackley(params_2), easom(params_3), rosenbrock(params_4), rosenbrock(params_5)
    obj_1_func, obj_2_func, obj_3_func, obj_4_func, obj_5_func = obj_1.ackley_func, obj_2.ackley_func, obj_3.easom_func,obj_4.rosenbrock_func, obj_5.rosenbrock_func
    obj_list, func_list = [obj_1, obj_2, obj_3, obj_4, obj_5], [obj_1_func, obj_2_func, obj_3_func, obj_4_func, obj_5_func]
    obj_name, params_list = ['2DAckley', '8DAckely', 'Easom', '2DRoesnbrock', '8DRoesnbrock'], [params_1, params_2, params_3, params_4, params_5]

    fin_res = [[] for i in range(len(obj_list) + 1)]
    bar=-1
    for obj, func, params in zip(obj_list, func_list, params_list):
        bar +=1
        res_std_err, res_mean = [[] for i in range(len(pop_size)*len(alpha_list))], [[] for i in range(len(pop_size)*len(alpha_list))]
        global_best = [[] for i in range(len(pop_size)*len(alpha_list))]
        l=-1
        for alpha, beta in zip(alpha_list,beta_list):
            for pop in pop_size:
                l+=1
                header = obj_name[bar] + 'alpha_' + str(alpha) + 'alpha_decay' + str(alpha_decay) +'beta' + str(beta)  +'num_agents_' + str(pop)
                print(header)
                res_std_err[l].append(header), res_mean[l].append(header)
                for i in range(num_trials):
                    set_seed = random.randint(1, 1000)
                    model = PSO(func, obj,num_agents= pop, num_gen=num_gen, seed= set_seed, obj_dim = params.shape[0],
                                alpha=alpha,alpha_decay=alpha_decay,beta = beta,floor = obj.floor, ceiling=obj.ceiling)
                    best_param = model.model_fit()
                    best_val = obj.return_value(best_param)
                    print(best_val)
                    err = abs(best_val - obj.min)
                    res_std_err[l].append(err), res_mean[l].append(best_val)
                    tmp_param = []
                    for h in range(len(best_param)):
                        tmp_param.append(best_param[h])

                    global_best[l].append(tmp_param)


        best_gstar =[]
        for err,best in zip(res_std_err, global_best):
            idx = np.argmin(np.asarray(err[1:]))
            best_gstar.append(best[idx])

        '''Calculate mean and std err for all the values'''
        config, val_mean, val_err, err_mean, err_std = [], [], [], [], []
        for mean, err in zip(res_mean, res_std_err):
            config.append(mean[0])
            val_err.append((statistics.stdev(mean[1:])/num_trials)), val_mean.append(statistics.mean(mean[1:]))
            err_std.append((statistics.stdev(err[1:])/num_trials)), err_mean.append(statistics.mean(err[1:]))
        fin_res[bar].append([config, val_mean, val_err, err_mean, err_std, best_gstar])

    return fin_res

def test_accelerated_pso():

    alpha_list = [0.4, 0.5, 0.6, 0.7]
    alpha_decay = 0.003
    beta_list = [0.4, 0.5, 0.6, 0.7]
    gamma_list = [0.01, 0.1, 1, 10]
    pop_size, num_gen, num_trials = [20, 40, 60], 500, NUM_TRIALS
    params_1, params_2, params_3, params_4, params_5 = np.array([2, 2]), np.array([2, 2, 2, 2, 2, 2, 2, 2]), np.array([2, 2]), \
                                                       np.array([2, 2]), np.array([2, 2, 2, 2, 2, 2, 2, 2]) ###change this

    obj_1, obj_2, obj_3, obj_4, obj_5 = ackley(params_1), ackley(params_2), easom(params_3), rosenbrock(params_4), rosenbrock(params_5)
    obj_1_func, obj_2_func, obj_3_func, obj_4_func, obj_5_func = obj_1.ackley_func, obj_2.ackley_func, obj_3.easom_func,obj_4.rosenbrock_func, obj_5.rosenbrock_func
    obj_list, func_list = [obj_1, obj_2, obj_3, obj_4, obj_5], [obj_1_func, obj_2_func, obj_3_func, obj_4_func, obj_5_func]
    obj_name, params_list = ['2DAckley', '8DAckely', 'Easom', '2DRoesnbrock', '8DRoesnbrock'], [params_1, params_2, params_3, params_4, params_5]

    fin_res = [[] for i in range(len(obj_list) + 1)]
    bar=-1
    for obj, func, params in zip(obj_list, func_list, params_list):
        bar +=1
        res_std_err, res_mean = [[] for i in range(len(pop_size)*len(alpha_list))], [[] for i in range(len(pop_size)*len(alpha_list))]
        global_best = [[] for i in range(len(pop_size)*len(alpha_list))]
        l=-1
        for alpha,beta in zip(alpha_list,beta_list):
            for pop in pop_size:
                l+=1
                header = obj_name[bar] + 'Alpha' + str(alpha) + 'alpha_decay' + str(alpha_decay) + 'beta' + str(beta) +'num_agents_' + str(pop)
                print(header)
                res_std_err[l].append(header), res_mean[l].append(header)
                for i in range(num_trials):
                    set_seed = random.randint(1, 1000)
                    model = Accelerated_PSO(func, obj,num_agents= pop, num_gen=num_gen, seed= set_seed, obj_dim = params.shape[0],
                                alpha=alpha,alpha_decay=alpha_decay,beta = beta,floor = obj.floor, ceiling=obj.ceiling)
                    best_param = model.model_fit()
                    best_val = obj.return_value(best_param)
                    print(best_val)
                    err = abs(best_val - obj.min)
                    res_std_err[l].append(err), res_mean[l].append(best_val)
                    tmp_param = []
                    for h in range(len(best_param)):
                        tmp_param.append(best_param[h])

                    global_best[l].append(tmp_param)


        best_gstar =[]
        for err,best in zip(res_std_err, global_best):
            idx = np.argmin(np.asarray(err[1:]))
            best_gstar.append(best[idx])

        '''Calculate mean and std err for all the values'''
        config, val_mean, val_err, err_mean, err_std = [], [], [], [], []
        for mean, err in zip(res_mean, res_std_err):
            config.append(mean[0])
            val_err.append((statistics.stdev(mean[1:])/num_trials)), val_mean.append(statistics.mean(mean[1:]))
            err_std.append((statistics.stdev(err[1:])/num_trials)), err_mean.append(statistics.mean(err[1:]))
        fin_res[bar].append([config, val_mean, val_err, err_mean, err_std, best_gstar])

    return fin_res


def harmony_search():

    alpha_list = [0.01, 0.05, 0.1, 0.5, 0.6]
    p_list = [0.1, 0.25, 0.5, 0.7, 0.8]
    beta_list = [1, 1.2, 1.5, 1.8, 2]

    gamma_list = [0.01, 0.1, 1, 10]
    pop_size, num_gen, num_trials = [10, 25, 50], 1500, NUM_TRIALS
    params_1, params_2, params_3, params_4, params_5 = np.array([2, 2]), np.array([2, 2, 2, 2, 2, 2, 2, 2]), np.array([2, 2]), \
                                                       np.array([2, 2]), np.array([2, 2, 2, 2, 2, 2, 2, 2]) ###change this

    obj_1, obj_2, obj_3, obj_4, obj_5 = ackley(params_1), ackley(params_2), easom(params_3), rosenbrock(params_4), rosenbrock(params_5)
    obj_1_func, obj_2_func, obj_3_func, obj_4_func, obj_5_func = obj_1.ackley_func, obj_2.ackley_func, obj_3.easom_func,obj_4.rosenbrock_func, obj_5.rosenbrock_func
    obj_list, func_list = [obj_1, obj_2, obj_3, obj_4, obj_5], [obj_1_func, obj_2_func, obj_3_func, obj_4_func, obj_5_func]
    obj_name, params_list = ['2DAckley', '8DAckely', 'Easom', '2DRoesnbrock', '8DRoesnbrock'], [params_1, params_2, params_3, params_4, params_5]
    fin_res = [[] for i in range(len(obj_list) + 1)]
    bar=-1
    for obj, func, params in zip(obj_list, func_list, params_list):
        bar +=1
        res_std_err, res_mean = [[] for i in range(len(pop_size)*len(alpha_list))],[[] for i in range(len(pop_size)*len(alpha_list))],
        global_best = [[] for i in range(len(pop_size)*len(alpha_list))],
        l=-1
        for alpha, p, beta in zip(alpha_list, p_list, beta_list):
            for pop in pop_size:
                l+=1
                header = obj_name[bar] + 'alpha' + str(alpha) + 'P_val' + str(p) +'beta' + str(beta) +'num_agents_' + str(pop)
                print(header)
                res_std_err[l].append(header), res_mean[l].append(header)
                for i in range(num_trials):
                    set_seed = random.randint(1, 1000)
                    model = CuckooSearch(func, obj,num_agents= pop, num_gen=num_gen, seed= set_seed, obj_dim = params.shape[0],
                                            alpha=alpha,p=p,beta = beta,floor = obj.floor, ceiling=obj.ceiling)
                    best_param = model.model_fit()
                    best_val = obj.return_value(best_param)
                    print(best_val)
                    err = abs(best_val - obj.min)
                    res_std_err[l].append(err), res_mean[l].append(best_val)
                    tmp_param = []
                    for h in range(len(best_param)):
                        tmp_param.append(best_param[h])

                    global_best[l].append(tmp_param)


        best_gstar =[]
        for err,best in zip(res_std_err, global_best):
            idx = np.argmin(np.asarray(err[1:]))
            best_gstar.append(best[idx])

        '''Calculate mean and std err for all the values'''
        config, val_mean, val_err, err_mean, err_std = [], [], [], [], []
        for mean, err in zip(res_mean, res_std_err):
            config.append(mean[0])
            val_err.append((statistics.stdev(mean[1:])/num_trials)), val_mean.append(statistics.mean(mean[1:]))
            err_std.append((statistics.stdev(err[1:])/num_trials)), err_mean.append(statistics.mean(err[1:]))
        fin_res[bar].append([config, val_mean, val_err, err_mean, err_std, best_gstar])

    return fin_res




def  plot_table(vals, lable_rows, lable_cols, name):
    fig, ax = plt.subplots()
    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    ax.table(cellText=vals,rowLabels=lable_rows, colLabels=lable_cols, loc='center', colWidths=[0.1 for x in lable_cols])
    #fig.tight_layout()
    #plt.show()
    fig.savefig('table_' + name)
    plt.close()

def manage_list(alist, names):
    #alist = [[[['2DAckleyF_0.9Cr_Prob0.5num_agents_10', '2DAckleyF_0.9Cr_Prob0.5num_agents_25', '2DAckleyF_0.9Cr_Prob0.5num_agents_50'], [1.3670188791945006, 0.27307430630642215, 0.13840709784658262], [0.09969347866454367, 0.12212254236195168, 0.06189753587068449], [1.3670188791945006, 0.27307430630642215, 0.13840709784658262], [0.09969347866454367, 0.12212254236195168, 0.06189753587068449]]], [[['8DAckelyF_0.9Cr_Prob0.5num_agents_10', '8DAckelyF_0.9Cr_Prob0.5num_agents_25', '8DAckelyF_0.9Cr_Prob0.5num_agents_50'], [2.5769045611745174, 1.6487727797971372, 1.507166193481756], [0.297404107791294, 0.0298445376906444, 0.014091671686037236], [2.5769045611745174, 1.6487727797971372, 1.507166193481756], [0.297404107791294, 0.0298445376906444, 0.014091671686037236]]], [[['EasomF_0.9Cr_Prob0.5num_agents_10', 'EasomF_0.9Cr_Prob0.5num_agents_25', 'EasomF_0.9Cr_Prob0.5num_agents_50'], [-0.20006439711300625, -8.053443065412687e-05, -0.20006450802445475], [0.08943551928387974, 1.7011735193579102e-08, 0.08943550688360713], [0.7999356028869937, 0.9999194655693459, 0.7999354919755453], [0.08943551928387974, 1.7011735189764383e-08, 0.08943550688360713]]], [[['2DRoesnbrockF_0.9Cr_Prob0.5num_agents_10', '2DRoesnbrockF_0.9Cr_Prob0.5num_agents_25', '2DRoesnbrockF_0.9Cr_Prob0.5num_agents_50'], [2.0933797430182546, 4.930380657631324e-32, 9.860761315262648e-33], [0.9361878816219716, 1.7079339599344613e-32, 4.4098665221655036e-33], [2.0933797430182546, 4.930380657631324e-32, 9.860761315262648e-33], [0.9361878816219716, 1.7079339599344613e-32, 4.4098665221655036e-33]]], [[['8DRoesnbrockF_0.9Cr_Prob0.5num_agents_10', '8DRoesnbrockF_0.9Cr_Prob0.5num_agents_25', '8DRoesnbrockF_0.9Cr_Prob0.5num_agents_50'], [84634.88979874588, 181.71974038096118, 215.59846202180645], [11371.212987323896, 17.49489068237657, 25.223649796974392], [84634.88979874588, 181.71974038096118, 215.59846202180645], [11371.212987323896, 17.49489068237657, 25.223649796974392]]]]
    lable_cols = ['val_mean', 'val_err', 'err_mean', 'err_std']
    j=-1
    for sub_list in alist:
        j+=1
        for f_list in sub_list:
            lable_rows = f_list[0]
            tmp_vals=[]
            for i in range(1, len(f_list)):
                tmp_vals.append([round(elem, 3) for elem in f_list[i]])
            plt_vals = list(map(list, zip(*tmp_vals)))
            #print("Stop")
            plot_table(plt_vals, lable_rows, lable_cols,str(names[j]) + "__" + str(j))
            #print('da da da da')

def disect_list_all(alist, name, func_name):

    #alist = [[[['egg_crate_alpha_0.1_beta_0.5gamma_0.01_pop_size_10', 'egg_crate_alpha_0.1_beta_0.5gamma_0.01_pop_size_25', 'egg_crate_alpha_0.1_beta_0.5gamma_0.01_pop_size_50', 'egg_crate_alpha_0.3_beta_0.7gamma_0.1_pop_size_10', 'egg_crate_alpha_0.3_beta_0.7gamma_0.1_pop_size_25', 'egg_crate_alpha_0.3_beta_0.7gamma_0.1_pop_size_50', 'egg_crate_alpha_0.5_beta_1gamma_1_pop_size_10', 'egg_crate_alpha_0.5_beta_1gamma_1_pop_size_25', 'egg_crate_alpha_0.5_beta_1gamma_1_pop_size_50', 'egg_crate_alpha_0.8_beta_1.2gamma_10_pop_size_10', 'egg_crate_alpha_0.8_beta_1.2gamma_10_pop_size_25', 'egg_crate_alpha_0.8_beta_1.2gamma_10_pop_size_50'], [4.18422740773118e-14, 1.6141157860204464e-14, 4.5733794109017235e-15, 4.3895260836852536e-14, 6.531217802760642e-14, 4.212800402549681e-14, 8.320663038108811e-14, 4.6038392078861703e-14, 1.282726262226534e-13, 5.029577266021334e-13, 7.097321938659812e-14, 1.0376851469227086e-13], [2.861126338052447e-14, 1.0049774188930344e-14, 1.0030514401638828e-15, 2.840351221092202e-14, 4.50177088785033e-14, 1.3711389106915817e-14, 5.726694961981748e-14, 3.0149796973847887e-14, 5.617773229630438e-14, 8.576362949603903e-14, 2.1301173414018214e-14, 9.689189337727469e-15], [1.9999999999999583, 1.9999999999999838, 1.9999999999999956, 1.999999999999956, 1.9999999999999347, 1.9999999999999578, 1.999999999999917, 1.999999999999954, 1.9999999999998717, 1.999999999999497, 1.999999999999929, 1.999999999999896], [2.865418737097889e-14, 1.0048591735576161e-14, 1.0205600981444537e-15, 2.8418673502176328e-14, 4.506165356422434e-14, 1.3738309013483032e-14, 5.72298701190236e-14, 3.014577520672848e-14, 5.613080539794496e-14, 8.572704824413412e-14, 2.1274752815165153e-14, 9.656068620905217e-15], [[3.5785000122192774e-08, 4.3416581691126805e-08], [-1.3365167593505714e-08, -3.144553563282196e-08], [-1.50879321745683e-08, -1.6769963009184149e-09], [2.994739491280982e-08, -4.833611685610773e-08], [-5.888296830478508e-08, -3.864509601917327e-08], [2.0678035606612005e-08, -4.40287171973788e-08], [-7.887391079196915e-08, 9.69910446909695e-09], [5.5879278789678536e-08, 1.6974874093395862e-08], [3.965688775411397e-08, 8.010343067044636e-08], [-1.5396340784847082e-07, -1.7456329830134028e-08], [-6.235642377575653e-08, -2.1612714552853631e-10], [-5.748094409263628e-08, 3.4843366758519216e-08]]]], [[['2D_Sphere_func_alpha_0.1_beta_0.5gamma_0.01_pop_size_10', '2D_Sphere_func_alpha_0.1_beta_0.5gamma_0.01_pop_size_25', '2D_Sphere_func_alpha_0.1_beta_0.5gamma_0.01_pop_size_50', '2D_Sphere_func_alpha_0.3_beta_0.7gamma_0.1_pop_size_10', '2D_Sphere_func_alpha_0.3_beta_0.7gamma_0.1_pop_size_25', '2D_Sphere_func_alpha_0.3_beta_0.7gamma_0.1_pop_size_50', '2D_Sphere_func_alpha_0.5_beta_1gamma_1_pop_size_10', '2D_Sphere_func_alpha_0.5_beta_1gamma_1_pop_size_25', '2D_Sphere_func_alpha_0.5_beta_1gamma_1_pop_size_50', '2D_Sphere_func_alpha_0.8_beta_1.2gamma_10_pop_size_10', '2D_Sphere_func_alpha_0.8_beta_1.2gamma_10_pop_size_25', '2D_Sphere_func_alpha_0.8_beta_1.2gamma_10_pop_size_50'], [3.563597009807241e-20, 3.9194278834890023e-19, 3.3176061080364814e-20, 1.6870896529942171e-18, 5.708256316639601e-20, 9.92138050203446e-21, 3.1730805869473707e-18, 5.680774468149793e-20, 5.718437191110908e-20, 2.4752109016324493e-16, 5.907646908099254e-18, 1.0053781208087515e-18], [2.1374326869900317e-20, 2.334069827582602e-19, 3.648177992270954e-21, 9.488090017719048e-19, 2.1442171011155626e-20, 6.2615245844341105e-21, 2.0134073191056674e-18, 1.876472248647951e-20, 3.820476228778479e-20, 1.7417736809715285e-16, 4.110080310942367e-18, 6.075395503268486e-19], [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.9999999999999998, 2.0, 2.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5700924586837752e-16, 0.0, 0.0], [[-0.38586956529485616, 7.353983377778142e-11], [0.722737030205879, -2.4870759493747983e-10], [-1.3275309756587854, -1.6738207277287548e-10], [-3.529359855515819, 1.7403758822167477e-09], [5.899442345438072, -1.6358103226137345e-10], [-1.4063256536258746, -1.3702741875432718e-10], [-0.19880582651018947, 5.706948831791657e-10], [3.895365244850004, 1.7398396376376538e-10], [8.977524800426593, -5.616652861343382e-11], [-2.394166136577719, -2.2222625551106452e-08], [-2.7436957268555084, -3.423474583851409e-09], [2.4156148164786044, 3.823446731137635e-10]]]], [[['8D_Sphere_func_alpha_0.1_beta_0.5gamma_0.01_pop_size_10', '8D_Sphere_func_alpha_0.1_beta_0.5gamma_0.01_pop_size_25', '8D_Sphere_func_alpha_0.1_beta_0.5gamma_0.01_pop_size_50', '8D_Sphere_func_alpha_0.3_beta_0.7gamma_0.1_pop_size_10', '8D_Sphere_func_alpha_0.3_beta_0.7gamma_0.1_pop_size_25', '8D_Sphere_func_alpha_0.3_beta_0.7gamma_0.1_pop_size_50', '8D_Sphere_func_alpha_0.5_beta_1gamma_1_pop_size_10', '8D_Sphere_func_alpha_0.5_beta_1gamma_1_pop_size_25', '8D_Sphere_func_alpha_0.5_beta_1gamma_1_pop_size_50', '8D_Sphere_func_alpha_0.8_beta_1.2gamma_10_pop_size_10', '8D_Sphere_func_alpha_0.8_beta_1.2gamma_10_pop_size_25', '8D_Sphere_func_alpha_0.8_beta_1.2gamma_10_pop_size_50'], [2.8621838844173593, 3.746827771485544e-14, 1.7613871543879675e-14, 4.177268883572449e-13, 2.284364312564049e-13, 1.6031900280086054e-13, 9.593031330768615e-13, 5.52794617393559e-13, 2.6849710351040877e-13, 8.03059960315644e-12, 2.5483000219633905e-12, 1.6440919423931251e-12], [2.0237009654593154, 1.1114291595639655e-14, 4.1568550670960906e-15, 1.5537657001317028e-13, 2.241808724230343e-14, 1.4296483143889017e-14, 4.843123516859229e-13, 5.973653285312062e-14, 1.0172005062978059e-13, 3.0793847805833052e-12, 9.558780453768181e-13, 1.2836666228533671e-13], [2.8619453515400903, 1.9999999999999625, 1.9999999999999822, 1.9999999999995821, 1.9999999999997715, 1.9999999999998397, 1.9999999999990408, 1.999999999999447, 1.9999999999997315, 1.9999999999919695, 1.9999999999974518, 1.999999999998356], [0.6096560713012733, 1.1147656456654803e-14, 4.1607450155120036e-15, 1.5536064878675955e-13, 2.2452322159177985e-14, 1.4287841374022352e-14, 4.842950188810104e-13, 5.974201805291764e-14, 1.0174199132270863e-13, 3.0793438345935538e-12, 9.559507934696165e-13, 1.283550584973986e-13], [[-2.647537896870463, 0.00795104473296185, -0.006725234204676455, -0.0008132586774020867, -0.0034604257761114156, -0.006102953795223067, -0.007643871208060415, -0.004666287766030433], [4.969605120507698, 1.0087254734210778e-07, -3.071468079121829e-09, 1.0758599419683262e-07, -2.660839411078489e-08, -5.664957121207934e-08, -1.5942920411694422e-07, -4.5737917243585155e-08], [-5.034636211965584, -6.264318772223379e-08, 2.386282331051615e-08, -2.2878690309861738e-08, 1.8134696125844782e-08, -7.674795658467248e-08, 1.0986126577750557e-07, -1.3671167477745588e-08], [-2.7194801017968886, -2.523738269680588e-07, 6.262096497584011e-07, 1.6494069609801109e-07, -7.310001562803818e-08, 1.6931017850700444e-07, 3.427822495651366e-07, 5.400753909145315e-08], [-2.251280563456964, 3.2077477261790707e-07, 1.5646430157078173e-07, 6.1825528338865916e-09, -2.6873076121699614e-07, -2.0705483868877058e-07, -9.874151874560205e-08, -8.88077819247856e-08], [2.633433298504663, -1.3109684049438052e-08, 2.450399053152571e-07, 8.497121126515228e-08, 7.924558510436073e-08, 2.4741688168465133e-08, 2.649553325309845e-08, -3.248179955393624e-07], [-4.123109658478459, -8.088889168851987e-07, 6.104077609602487e-08, -3.0790692260931174e-07, -2.9101235232071766e-07, 3.9974130868902084e-07, -8.023927312832582e-07, 5.54517967895942e-08], [2.763717982367252, -6.264247568298941e-08, -3.4142841868406094e-07, 1.3903653918862884e-07, -2.2542213103536195e-07, -5.473110611822502e-07, -3.2973333642019293e-07, 1.958510154415539e-07], [-2.108493901518746, 1.0712656922268128e-08, -4.3203211819248034e-08, 4.1322634319443247e-07, 3.876417070205952e-07, 1.4302435962756608e-08, -2.3914912159329153e-07, -1.7874768356958884e-07], [-5.874066054515492, -3.044340931895156e-06, -3.804995379219509e-07, 8.418012139892902e-07, 4.4847798242968794e-07, 6.392325871491952e-07, 5.963990510423408e-07, -1.1395805379594102e-06], [3.00404206196727, -7.058330636759156e-07, -6.907659727835058e-07, -1.062573249599048e-06, 3.71097658554265e-08, -8.323872246146277e-07, -6.733818261701332e-07, 8.049889153107064e-07], [1.1883035029142948, 1.8056935440813183e-07, 5.648737266597391e-07, 6.291941818252812e-07, -6.440443541013427e-07, 2.7015725163305036e-07, -2.9975442166470785e-07, 7.074080123392133e-07]]]]]
    lable_cols = ['val_mean', 'val_err', 'err_mean', 'err_std', 'best_param']
    j=0
    for sub_list in alist:
        j+=1
        for f_list in sub_list:
            lable_rows = f_list[0]
            tmp_vals=[]
            tmp_vals.append(lable_rows)
            for i in range(1, len(f_list)):
                if i !=len(f_list) -1:
                    tmp_vals.append([round(elem, 3) for elem in f_list[i]])
                else:
                    tmp_vals.append(f_list[i])
            df_vals = pd.DataFrame(tmp_vals)
            df_vals1 = df_vals.T
            df_vals1.to_csv(name + func_name[j - 1]+ '.csv')
            print('DOne')


def manage_list_firefly(alist):
    #alist = [['alpha_0.1_beta_0.5gamma_0.01_pop_size_10', 'alpha_0.1_beta_0.5gamma_0.01_pop_size_25', 'alpha_0.1_beta_0.5gamma_0.01_pop_size_50', 'alpha_0.3_beta_0.7gamma_0.1_pop_size_10', 'alpha_0.3_beta_0.7gamma_0.1_pop_size_25', 'alpha_0.3_beta_0.7gamma_0.1_pop_size_50', 'alpha_0.5_beta_1gamma_1_pop_size_10', 'alpha_0.5_beta_1gamma_1_pop_size_25', 'alpha_0.5_beta_1gamma_1_pop_size_50', 'alpha_0.8_beta_1.2gamma_10_pop_size_10', 'alpha_0.8_beta_1.2gamma_10_pop_size_25', 'alpha_0.8_beta_1.2gamma_10_pop_size_50'], [2.0000002250707767, 2.000000225070779, 2.000000225070779, 2.0000002250707647, 2.000000225070775, 2.0000002250707785, 2.0000002250707603, 2.0000002250707696, 2.0000002250707754, 2.0000002250707154, 2.000000225070772, 2.0000002250707762], [3.1401849173675503e-16, 1.5700924586837752e-16, 3.1401849173675503e-16, 7.379434555813742e-15, 2.5121479338940403e-15, 3.1401849173675503e-16, 1.1775693440128313e-14, 7.222425309945365e-15, 0.0, 6.280369834735101e-16, 1.5700924586837752e-16, 7.850462293418876e-16], [2.2507077668265651e-07, 2.2507077912514717e-07, 2.2507077890310256e-07, 2.2507076447020324e-07, 2.2507077490629968e-07, 2.2507077845901335e-07, 2.2507076047340036e-07, 2.2507076957722916e-07, 2.2507077535038889e-07, 2.2507071539834556e-07, 2.250707715756306e-07, 2.250707760165227e-07], [3.1401849173675503e-16, 1.5700924586837752e-16, 3.1401849173675503e-16, 7.379434555813742e-15, 2.5121479338940403e-15, 3.1401849173675503e-16, 1.1775693440128313e-14, 7.222425309945365e-15, 0.0, 6.280369834735101e-16, 1.5700924586837752e-16, 7.850462293418876e-16], [[[-1.6800377368679596e-08, -4.153845657880271e-07], [-4.261432954829417e-08, -4.5657455127703675e-07]], [[1.3057601012034517e-09, -4.3292913902319467e-07], [5.677879522542848e-09, -4.2577402151850593e-07]], [[-6.635470987215746e-09, -4.6221503262146256e-07], [-2.6666764805388566e-08, -4.397934237810692e-07]], [[-2.9045532982423703e-09, -3.999999570938961], [-5.004355712223604e-08, -4.3981691345445155e-07]], [[-8.516219027254974e-09, -4.772988394561152e-07], [-6.518384132495457e-08, -4.465206519693853e-07]], [[-4.258316898858068e-09, -4.7104128811012336e-07], [9.881130167087601e-09, -4.213888708429265e-07]], [[-2.9661695295126737e-09, -3.9999994766822407], [-2.9670576668190145e-08, -4.7458099164647656e-07]], [[2.307138304988089e-09, -4.63875943613936e-07], [-3.371306801945095e-08, -5.455657516273145e-07]], [[-4.831378960085592e-08, -4.4621625909685405e-07], [-3.3667517075214463e-08, -4.8544906459626e-07]], [[-1.712008998768617e-07, -3.9891256577378467e-07], [-1.7759868607945582e-07, -4.846677302841061e-07]], [[1.0937039019593748e-08, -5.127162456895133e-07], [-5.452446242876858e-08, -4.118007620260123e-07]], [[4.605485712152896e-09, -5.00615083990195e-07], [1.7060185886936065e-08, -4.1489474494845046e-07]]]]
    #alist = [['alpha_0.1_beta_0.5gamma_0.01_pop_size_10', 'alpha_0.1_beta_0.5gamma_0.01_pop_size_25', 'alpha_0.1_beta_0.5gamma_0.01_pop_size_50', 'alpha_0.3_beta_0.7gamma_0.1_pop_size_10', 'alpha_0.3_beta_0.7gamma_0.1_pop_size_25', 'alpha_0.3_beta_0.7gamma_0.1_pop_size_50', 'alpha_0.5_beta_1gamma_1_pop_size_10', 'alpha_0.5_beta_1gamma_1_pop_size_25', 'alpha_0.5_beta_1gamma_1_pop_size_50', 'alpha_0.8_beta_1.2gamma_10_pop_size_10', 'alpha_0.8_beta_1.2gamma_10_pop_size_25', 'alpha_0.8_beta_1.2gamma_10_pop_size_50'], [2.000000225070779, 2.000000225070778, 2.00000022507078, 2.0000002250707754, 2.0000002250707762, 2.000000225070778, 2.0000002250707665, 2.000000225070777, 2.000000225070775, 2.000000225070754, 2.000000225070777, 2.0000002250707745], [3.1401849173675503e-16, 9.42055475210265e-16, 0.0, 1.5700924586837752e-16, 1.5700924586837752e-16, 1.5700924586837752e-16, 9.106536260365895e-15, 4.710277376051325e-16, 3.454203409104305e-15, 1.648597081617964e-14, 1.0990647210786425e-15, 1.7271017045521525e-15], [2.2507077890310256e-07, 2.2507077801492414e-07, 2.2507077979128098e-07, 2.2507077512834428e-07, 2.250707764606119e-07, 2.2507077823696875e-07, 2.250707664686047e-07, 2.2507077690470112e-07, 2.2507077490629968e-07, 2.250707538120622e-07, 2.2507077734879033e-07, 2.2507077468425507e-07], [3.1401849173675503e-16, 9.42055475210265e-16, 0.0, 1.5700924586837752e-16, 1.5700924586837752e-16, 1.5700924586837752e-16, 9.106536260365895e-15, 4.710277376051325e-16, 3.454203409104305e-15, 1.648597081617964e-14, 1.0990647210786425e-15, 1.7271017045521525e-15], [[-1.882420555018383e-09, -4.2322741299386737e-07], [-1.4875874640658395e-08, -4.864773429277118e-07], [1.06387326693176e-08, -4.532202556584225e-07], [-2.4626938020475473e-08, -4.95787834759889e-07], [-2.1404155174260013e-08, -4.864822805532237e-07], [2.68447641337741e-08, -4.638155457146558e-07], [-8.304516030199966e-08, -5.299082849475786e-07], [-3.138794620295869e-08, -4.1927534656015113e-07], [-7.061064997513279e-08, -4.4962288363784085e-07], [2.750809843363644e-08, -3.9999994436068573], [2.8104521569216022e-08, -4.151746934288679e-07], [8.03116458400787e-09, -3.8768297479362543e-07]]]
    lable_cols = ['val_mean', 'val_err', 'err_mean', 'err_std', 'best param']
    j=-1
    lable_rows = alist[0]
    tmp_vals=[]
    tmp_vals.append(alist[0])
    for i in range(1, len(alist)):
        
        j+=1
        if i != len(alist)-1:
            tmp_vals.append([round(elem, 12) for elem in alist[i]])
        else:
            tmp_vals.append(alist[i])

    plt_vals = list(map(list, zip(*tmp_vals)))
    df_vals = pd.DataFrame(tmp_vals)
    #print("Stop")
    #plot_table_firefly(plt_vals, lable_rows, lable_cols,"plot__" + str(j))
    #print('da da da da'
    return df_vals.T



if __name__ == '__main__':
    '''
    The snippet below will save output of Firefly on Four_Peak_Function
    '''
    FF1_list = test_3A()
    new_df = manage_list_firefly(FF1_list)
    new_df.to_csv('four_peak_firefly.csv')

    '''
    The snippet below will save output of Firefly on Egg Crate and D-Dim funcs
    '''
    ff_func = ['egg_crate', '2D Sphere', '8D Sphere']
    FF2_list = test_3B()
    FF_df = disect_list_all(FF2_list, 'Firefly_', ff_func)

    '''
    The snippet below will save output of DE Ackley, Easom and RosenBrock's Function
    '''
    Other_Func = ['2D_Ackley', '8D_Ackley', 'Easom', '2D_Rosenbrock', '8D_Rosenbrock']
    DE_list = test_differential_evolution()
    DE_df = disect_list_all(DE_list, 'Differential_Evolution_', Other_Func)

    '''
    The snippet below will save output of PSO Ackley, Easom and RosenBrock's Function
    '''
    PSO_list = test_pso()
    PSO_df = disect_list_all(PSO_list, 'PSO_list_', Other_Func)

    '''
    The snippet below will save output of APSO Ackley, Easom and RosenBrock's Function
    '''
    APSO_list = test_accelerated_pso()
    APSO_df = disect_list_all(APSO_list, 'APSO_list_', Other_Func)

    '''
    The snippet below will save output of Cuckoo Search on  Ackley, Easom and RosenBrock's Function
    '''
    Cuckoo_list = harmony_search()
    Cuckoo_df = disect_list_all(Cuckoo_list, 'Cuckoo_list_', Other_Func)


    print("Cortana Noo")

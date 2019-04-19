__author__ = 'Sujit Khanna'
"""
Author: Sujit Khanna sk2913
This file contains code for the first assignment
"""
'''
This file contains all the objective function to bse used for optimization
'''

import numpy as np
import math as m

class four_peak:

    def __init__(self, params):
        self.name = 'four_peak'
        self.params = params
        self.dim = self.params.shape[0]
        self.floor = -5
        self.ceiling = 5
        self.max=2
        self.min=2

    def four_peak_func(self, param_new):
        param_ary = np.asarray(np.zeros(self.dim)).astype(float)
        for i in range(self.dim):
            param_ary[i] = param_new[i]
        x, y = param_ary[0], param_ary[1]
        obj_func = m.exp(-(x - 4)**2 - (y - 4)**2) + m.exp(-(x + 4)**2 - (y - 4)**2) + 2*( m.exp(-(x)**2 - (y)**2) + m.exp(-(x)**2 - (y+ 4)**2))
        return obj_func

    def return_value(self, param_new):
        return self.four_peak_func(param_new)


class egg_crate:

    def __init__(self, params):
        self.name = 'egg_crate'
        self.params = params
        self.dim = self.params.shape[0]
        '''check floor and ceiling values for this function'''
        self.floor = -5
        self.ceiling = 5
        self.max=2
        self.min=0


    def egg_crate_func(self, param_new):
        param_ary = np.asarray(np.zeros(self.dim)).astype(float)
        for i in range(self.dim):
            param_ary[i] = param_new[i]
        x1, x2 = param_ary[0], param_ary[1]
        obj_func = x1**2 + x2**2 + 25*(m.sin(x1)**2 + m.sin(x2)**2)
        return (1/obj_func)

    def return_value(self, param_new):
        return 1/(self.egg_crate_func(param_new))


class one_dim_func:

    def __init__(self, params):
        self.name = 'sphere'
        self.params = params
        self.dim = self.params.shape[0]
        '''check floor and ceiling values for this function'''
        self.floor = -5
        self.ceiling = 5
        self.max=2
        self.min=0


    def sphere_func(self, param_new):
        param_ary = np.asarray(np.zeros(self.dim)).astype(float)
        for i in range(self.dim):
            param_ary[i] = param_new[i]
        obj_func=0
        for i in range(0, self.dim):
            obj_func += (param_ary[i])**2
        return 1/(obj_func)

    def return_value(self, param_new):
        return 1/(self.sphere_func(param_new))



class ackley:

    def __init__(self, params):
        self.name = 'ackley'
        self.params = params
        self.dim = self.params.shape[0]
        '''check floor and ceiling values for this function'''
        self.floor = -35
        self.ceiling = 35
        self.max=2
        self.min=0


    def ackley_func(self, param_new):
        param_ary = np.asarray(np.zeros(self.dim)).astype(float)
        for i in range(self.dim):
            param_ary[i] = param_new[i]
        tmp_root, tmp_cosine = 0, 0
        for i in range(self.dim):
            tmp_root +=(param_ary[i])**2
            tmp_cosine += m.cos(2*m.pi*param_ary[i])
        exp_term_1 = -0.02*m.sqrt((tmp_root/self.dim))
        exp_term_2 = tmp_cosine/self.dim

        obj_func = ( -20*m.exp(exp_term_1) - m.exp(exp_term_2) + 20 + m.exp(1))

        return 1/obj_func

    def return_value(self, param_new):
        return 1/(self.ackley_func(param_new))

class easom:

    def __init__(self, params):
        self.name = 'easom'
        self.params = params
        self.dim = self.params.shape[0]
        '''check floor and ceiling values for this function'''
        self.floor = -100
        self.ceiling = 100
        self.max=2
        self.min=-1


    def easom_func(self, param_new):
        param_ary = np.asarray(np.zeros(self.dim)).astype(float)
        for i in range(self.dim):
            param_ary[i] = param_new[i]
        x1, x2 = param_ary[0], param_ary[1]
        obj_func = -(m.cos(x1))*m.cos(x2)*m.exp( -1*(x1 - m.pi)**2 -(x2 - m.pi)**2 )
        return -1*(obj_func)

    def return_value(self, param_new):
        return -1*(self.easom_func(param_new))

class rosenbrock:

    def __init__(self, params):
        self.name = 'rosenbrock'
        self.params = params
        self.dim = self.params.shape[0]
        '''check floor and ceiling values for this function'''
        self.floor = -30
        self.ceiling = 30
        self.max=2
        self.min=0


    def rosenbrock_func(self, param_new):
        param_ary = np.asarray(np.zeros(self.dim)).astype(float)
        for i in range(self.dim):
            param_ary[i] = param_new[i]
        obj_func=0
        for i in range(0, self.dim):
            obj_func+= 100*(param_ary[i] - (param_ary[i-1])**2)**2  + (param_ary[i -1] - 1)**2

        return 1/(obj_func)


    def return_value(self, param_new):
        return 1/(self.rosenbrock_func(param_new))
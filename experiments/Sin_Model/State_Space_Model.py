import matplotlib.pyplot as plt
import random
import numpy as np
import os
import math

from tqdm import trange, tqdm
from scipy.stats import multivariate_normal

import warnings
warnings.filterwarnings("ignore")

class Stochastic_Sin_System:
    """
    Define a three stochastic dynamic system:
    x(i+1) = f(x(i)) + noise, noise ~ N(0, Q),
    y(i+1) = h(x(i+1)) + noise, noise ~ N(0, R),
    """
    def __init__(self, params_dict, m, n, q2, r2):
        self.alpha = params_dict['alpha']
        self.beta = params_dict['beta']
        self.phi = params_dict['phi']
        self.delta = params_dict['delta']
        self.a = params_dict['a']
        self.b = params_dict['b']
        self.c = params_dict['c']
        
        self.q2 = q2
        self.r2 = r2
        self.m = m
        self.n = n
    
    def f(self, x):
        return self.alpha * np.sin(self.beta * x + self.phi) + self.delta
            
    def h(self, x, obs_type='I'):
        if obs_type == 'I':
            return self.h_I(x)
        elif obs_type == 'NL':
            return self.h_nonlinear(x)
        elif obs_type == 'partial':
            return self.h_partial(x)
        elif obs_type == 'arctan':
            return self.h_arctan(x)
    
    def h_I(self, x):
        return x
    
    def h_nonlinear(self, x):
        return self.a * (self.b * x + self.c) ** 2
    
    def h_partial(self, x):
        return x[0]
    
    def h_arctan(self, x):
        return np.arctan(x[1] / x[0])
    
    def euler_forward(self, trans_noise_sample, obs_noise_sample, T, init_state, obs_type):
        states = [init_state]
        measurements = [self.h(init_state, obs_type).flatten() + obs_noise_sample[0]]
        
        for t in range(1, T):
            trans_noise = trans_noise_sample[t-1]
            obs_noise = obs_noise_sample[t-1]
            state = self.f(states[-1]) + trans_noise
            measurement = self.h(state, obs_type).flatten() + obs_noise
            states.append(state)
            measurements.append(measurement)
        
        states = np.array(states)
        obs = np.array(measurements)
        return states, measurements
    
    def generate_sample(self, T, T_test, N_train, N_test, obs_type):
        Q = self.q2 * np.eye(self.m)
        R = self.r2 * np.eye(self.n)
        
        init_state = multivariate_normal(mean=1 * np.ones(self.m), cov=Q)
        trans_noise = multivariate_normal(mean=np.zeros(self.m), cov=Q)
        obs_noise = multivariate_normal(mean=np.zeros(self.n), cov=R)
        
        x_train = np.zeros([N_train, T, self.m])
        y_train = np.zeros([N_train, T, self.n])
        x_test = np.zeros([N_test, T_test, self.m])
        y_test = np.zeros([N_test, T_test, self.n])
        
        print("==> Generate training samples:")
        i = 0
        with tqdm(total=N_train) as pbar:
            while i < N_train:
                trans_noise_sample_trn = trans_noise.rvs(size=[1, T-1])
                obs_noise_sample_trn = obs_noise.rvs(size=[1, T])
                init_state_sample_trn = init_state.rvs(size=[1])
                x_train_i, y_train_i = self.euler_forward(trans_noise_sample_trn, obs_noise_sample_trn, T,
                                                          init_state_sample_trn, obs_type)
                if len(np.where(np.isnan(x_train_i).flatten())[0]) > 0:
                    continue
                else:
                    x_train[i], y_train[i] = x_train_i, y_train_i
                    i += 1
                    pbar.update(1)
            
        print("==> Generate test samples:")
        j = 0
        with tqdm(total=N_test) as pbar:
            while j < N_test:
                trans_noise_sample_test = trans_noise.rvs(size=[1, T_test-1])
                obs_noise_sample_test = obs_noise.rvs(size=[1, T_test])
                init_state_sample_test = init_state.rvs(size=[1])
                x_test_j, y_test_j = self.euler_forward(trans_noise_sample_test, obs_noise_sample_test, T_test,
                                                        init_state_sample_test, obs_type)
                if len(np.where(np.isnan(x_test_j).flatten())[0]) > 0:
                    continue
                else:
                    x_test[j], y_test[j] = x_test_j, y_test_j
                    j += 1
                    pbar.update(1)
        return x_train, y_train, x_test, y_test
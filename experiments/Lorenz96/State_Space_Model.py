import random
import numpy as np
import os
import math

from scipy.stats import multivariate_normal
from tqdm import trange, tqdm

import warnings
warnings.filterwarnings("ignore")


class Stochastic_Lorenz96:    
    def __init__(self, F, q2, r2, delta_t, m, n):
        self.delta_t = delta_t
        self.F = F
        self.q2 = q2
        self.r2 = r2
        self.m = m
        self.n = n
    
    def f(self, x):
        """
        f(x) = u1 * (u2 - u3) - u4 + F,
        u1 = [x0, x1, ..., x_{m-1}],
        u2 = [x2, x3, ..., x_{m+1}],
        u3 = [x_{-1}, x0, ..., x_{m-2}],
        u4 = [x1, x2, ..., xm],
        """
        if len(x.shape) == 1:
            u_1 = np.concatenate([x[-1:], x[0:-1]])
            u_2 = np.concatenate([x[1:], x[0:1]])
            u_3 = np.concatenate([x[-2:-1], x[-1:], x[0:-2]])
            u_4 = x
            return u_1 * (u_2 - u_3) - u_4 + np.ones_like(x) * self.F
            
        elif len(x.shape) == 2: # (m, ensemble_size)
            u_1 = np.concatenate([x[-1:], x[0:-1]])
            u_2 = np.concatenate([x[1:], x[0:1]])
            u_3 = np.concatenate([x[-2:-1], x[-1:], x[0:-2]])
            u_4 = x
            return u_1 * (u_2 - u_3) - u_4 + np.ones_like(x) * self.F
        else:
            print("=> Dimension mismatch !")
                
    def h(self, x, obs_type):
        if obs_type == 'I':
            return self.h_I(x)
        elif obs_type == 'x2':
            return self.h_x2(x)
        elif obs_type == 'NL':
            return self.h_x3(x)
        elif obs_type == 'sin':
            return self.h_sin(x)
    
    def h_I(self, x):
        return x
    
    def h_x2(self, x): # (m, ensemble_size)
        return x ** 2
    
    def h_x3(self, x): # (m, ensemble_size)
        return x ** 3
    
    def h_sin(self, x): # (m, ensemble_size)
        return np.sin(x / 10)
    
    def runge_kutta_4_step(self, trans_noise, obs_noise, T, 
                           delta_t, Delta_t, init_state, obs_type):
        states_lst = [init_state]
        state_measure_lst = [init_state]
        measurement_lst = []
        
        n_iter = int(Delta_t / delta_t)
        t = 1
        while t < T:
            trans_noise_sample = trans_noise.rvs(size=[1, n_iter])
            for n in range(n_iter):
                trans_noise_n = trans_noise_sample[n]
                state = states_lst[-1] + self.f(states_lst[-1]) * delta_t + trans_noise_n
                k1 = self.f(states_lst[-1])
                k2 = self.f(states_lst[-1] + 0.5 * self.delta_t * k1)
                k3 = self.f(states_lst[-1] + 0.5 * self.delta_t * k2)
                k4 = self.f(states_lst[-1] + self.delta_t * k3)
                state = states_lst[-1] + 1/6 * self.delta_t * (k1 + 2 * k2 + 2 * k3 + k4) + trans_noise_n
                states_lst.append(state)
                
            state_measure_lst.append(state)
            t += 1
        
        assert len(state_measure_lst) == T, print("Sequence length error!")
        
        states = np.array(state_measure_lst)
        obs_noise_sample = obs_noise.rvs(size=[1, T])
        for t in range(T):
            obs_noise_t = obs_noise_sample[t]
            measurement = self.h(states[t], obs_type).flatten() + obs_noise_t
            measurement_lst.append(measurement)
        
        measurements = np.array(measurement_lst)
        return states, measurements
    
    def generate_sample(self, delta_t, Delta_t, T, T_test, N_train, N_test, init_state, obs_type):
        Q = self.q2 * np.eye(self.m)
        R = self.r2 * np.eye(self.n)
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
#                 trans_noise_sample_trn = trans_noise.rvs(size=[1, T-1])
#                 obs_noise_sample_trn = obs_noise.rvs(size=[1, T])
                x_train_i, y_train_i = self.runge_kutta_4_step(trans_noise, obs_noise, T, delta_t, 
                                                               Delta_t, init_state, obs_type)
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
#                 trans_noise_sample_test = trans_noise.rvs(size=[1, T_test-1])
#                 obs_noise_sample_test = obs_noise.rvs(size=[1, T_test])
                x_test_j, y_test_j = self.runge_kutta_4_step(trans_noise, obs_noise, T, delta_t, 
                                                             Delta_t, init_state, obs_type)
                if len(np.where(np.isnan(x_test_j).flatten())[0]) > 0:
                    continue
                else:
                    x_test[j], y_test[j] = x_test_j, y_test_j
                    j += 1
                    pbar.update(1)
        return x_train, y_train, x_test, y_test
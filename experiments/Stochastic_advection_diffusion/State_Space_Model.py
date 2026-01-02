import random
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import multivariate_normal
from tqdm import trange, tqdm
# from State_Space_Model import Stochastic_AD_System

import scipy.sparse as sp
import scipy.sparse.linalg as spla

import warnings
warnings.filterwarnings("ignore")

class Stochastic_AD_System:
    
    def __init__(self, params_dict, sigma, obs_type, sensor_index):
        L = params_dict["L"]
        T = params_dict["T"]
        self.D = params_dict["D"]            # Diffusion parameter
        self.k = params_dict["k"]       # Advection parameter
        self.Nx = params_dict["Nx"]
        self.Nt = params_dict["Nt"]
        
        self.dx = 2 * L / (self.Nx - 1)     # Spatial step size
        self.Dt = T / self.Nt           # assimilation time step
        self.dt = self.Dt / 10          # Time step size

        alpha = self.D * self.dt / self.dx**2  
        beta = self.k * self.dt / self.dx
        
        # Create spatial and temporal grids
        self.x = np.linspace(-L, L, self.Nx)

        self.sigma = sigma
        self.obs_type = obs_type
        self.sensor_index = sensor_index

        # Initial and boundary conditions
        self.s_0 = self.initial(self.x)
        
        # Sparse matrix for the implicit scheme
        diagonals = [
            -(alpha + beta) * np.ones(self.Nx - 1),       # Lower diagonal
            (1 + 2 * alpha + beta) * np.ones(self.Nx), # Main diagonal
            -alpha * np.ones(self.Nx - 1)       # Upper diagonal
        ]
        self.A = sp.diags(diagonals, offsets=[-1, 0, 1], format='csc')
        
        self.m = self.Nx
        
    def initial(self, x):
        return -np.sin(np.pi * x)
    
    def source(self, x):
        return 5 * (x + 1) * (x - 1)

    def h(self, x, obs_type):
        if obs_type == 'I':
            return self.h_I(x)
        elif obs_type == 'NL':
            return self.h_NL(x)
    
    def h_I(self, x):
        return x

    def h_NL(self, x):
        return np.exp(-x + 0.5)
        
    def forward_step(self, s_old, F, z):
        rhs = s_old + self.dt * F + z

        # Solve the linear system A * s_new = rhs
        rhs[0] = 0
        rhs[-1] = 0
        
        self.A[0, 0] = 1
        self.A[0, 1] = 0
        self.A[-1, -2] = 0
        self.A[-1, -1] = 1
        
        s_new = spla.spsolve(self.A, rhs)
        return s_new
    
    def solve_AD(self, s_0, dt, Dt, trans_noise):
        s_old = s_0    # Placeholder for the next time step
        F = self.source(self.x)

        n_iter = int(Dt / dt)
        S = np.zeros([self.Nt, self.m])
        S[0] = self.s_0

        for n in range(self.Nt-1):
            trans_noise_sample = trans_noise.rvs(size=[1, n_iter])
            
            for i in range(n_iter):
                # Right-hand side vector
                z = trans_noise_sample[i]
                s_new = self.forward_step(s_old, F, z)

                # Update the solution
                s_old = s_new.copy()
                
            S[n+1] = s_new
        return S        
    
    def generate_state_sample(self, N_train, N_test):
        Q = self.sigma ** 2 * self.dt * np.eye(self.Nx)
        trans_noise = multivariate_normal(mean=np.zeros(self.m), cov=Q)
        
        x_train = np.zeros([N_train, self.Nt, self.m])
        x_test = np.zeros([N_test, self.Nt, self.m])
        
        print("==> Generate training samples:")
        i = 0
        data_upper = 1e+3
        with tqdm(total=N_train) as pbar:
            while i < N_train:
                x_train_i = self.solve_AD(self.s_0, self.dt, self.Dt, trans_noise)
                if len(np.where(np.isnan(x_train_i).flatten())[0]) > 0:
                    continue
                else:
                    x_train[i] = x_train_i
                    i += 1
                    pbar.update(1)
                
        print("==> Generate test samples:")
        j = 0
        with tqdm(total=N_test) as pbar:
            while j < N_test:
                x_test_j = self.solve_AD(self.s_0, self.dt, self.Dt, trans_noise)
                if len(np.where(np.isnan(x_test_j).flatten())[0]) > 0:
                    continue
                else:
                    x_test[j] = x_test_j
                    j += 1
                    pbar.update(1)
        return x_train, x_test
    


















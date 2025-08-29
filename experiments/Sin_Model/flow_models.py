import torch
import math
import numpy as np
import time
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as distributions

from torchvision import datasets, transforms
from torchvision.utils import save_image
from scipy.stats import multivariate_normal
from torch.utils.data import TensorDataset
from torch.optim import lr_scheduler
from tqdm import tqdm, trange

class CouplingLayer(nn.Module):
    def __init__(self, input_dim, hid_dim, mask, cond_dim=None, s_tanh_activation=True, smooth_activation=False):
        super().__init__()
        
        if cond_dim is not None:
            total_input_dim = input_dim + cond_dim
        else:
            total_input_dim = input_dim

        self.s_fc1 = nn.Linear(total_input_dim, hid_dim)
        self.s_fc2 = nn.Linear(hid_dim, hid_dim)
        self.s_fc3 = nn.Linear(hid_dim, input_dim)
        self.t_fc1 = nn.Linear(total_input_dim, hid_dim)
        self.t_fc2 = nn.Linear(hid_dim, hid_dim)
        self.t_fc3 = nn.Linear(hid_dim, input_dim)
        self.mask = nn.Parameter(mask, requires_grad=False)
        self.s_tanh_activation = s_tanh_activation
        self.smooth_activation = smooth_activation

    def forward(self, x, cond_x=None, mode='direct'):
        x_m = x * self.mask
        if cond_x is not None:
            x_m = torch.cat([x_m, cond_x], -1)
        if self.smooth_activation:
            if self.s_tanh_activation:
                s_out = torch.tanh(self.s_fc3(F.elu(self.s_fc2(F.elu(self.s_fc1(x_m)))))) * (1-self.mask)
            else:
                s_out = self.s_fc3(F.elu(self.s_fc2(F.elu(self.s_fc1(x_m))))) * (1-self.mask)
            t_out = self.t_fc3(F.elu(self.t_fc2(F.elu(self.t_fc1(x_m))))) * (1-self.mask)
        else:
            if self.s_tanh_activation:
                s_out = torch.tanh(self.s_fc3(F.relu(self.s_fc2(F.relu(self.s_fc1(x_m)))))) * (1-self.mask)
            else:
                s_out = self.s_fc3(F.relu(self.s_fc2(F.relu(self.s_fc1(x_m))))) * (1-self.mask)
            t_out = self.t_fc3(F.relu(self.t_fc2(F.relu(self.t_fc1(x_m))))) * (1-self.mask)
        if mode == 'direct':
            y = x * torch.exp(s_out) + t_out
            log_det_jacobian = s_out.sum(-1, keepdim=True)
        else:
            y = (x - t_out) * torch.exp(-s_out)
            log_det_jacobian = -s_out.sum(-1, keepdim=True)
        return y, log_det_jacobian

    
class RealNVP(nn.Module):
    def __init__(self, input_dim, hid_dim = 256, n_layers = 2, cond_dim = None, s_tanh_activation = True, smooth_activation=False):
        super().__init__()
        assert n_layers >= 2, 'num of coupling layers should be greater or equal to 2'
        
        self.input_dim = input_dim
        mask = (torch.arange(0, input_dim) % 2).float()
        self.modules = []
        self.modules.append(CouplingLayer(input_dim, hid_dim, mask, cond_dim, s_tanh_activation, smooth_activation))
        for _ in range(n_layers - 2):
            mask = 1 - mask
            self.modules.append(CouplingLayer(input_dim, hid_dim, mask, cond_dim, s_tanh_activation, smooth_activation))
        self.modules.append(CouplingLayer(input_dim, hid_dim, 1 - mask, cond_dim, s_tanh_activation, smooth_activation))
        self.module_list = nn.ModuleList(self.modules)
        
    def forward(self, x, cond_x=None, mode='direct'):
        """ Performs a forward or backward pass for flow modules.
        Args:
            x: a tuple of inputs and logdets
            mode: to run direct computation or inverse
        """
        logdets = torch.zeros(x.size(), device=x.device).sum(-1, keepdim=True)

        assert mode in ['direct', 'inverse']
        if mode == 'direct':
            for module in self.module_list:
                x, logdet = module(x, cond_x, mode)
                logdets += logdet
        else:
            for module in reversed(self.module_list):
                x, logdet = module(x, cond_x, mode)
                logdets += logdet

        return x, logdets

    def log_probs(self, x, cond_x = None):
        u, log_jacob = self(x, cond_x)
        log_probs = (-0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi)).sum(
            -1, keepdim=True)
        return (log_probs + log_jacob).sum(-1, keepdim=True)

    def sample(self, num_samples, noise=None, cond_x=None):
        if noise is None:
            noise = torch.Tensor(num_samples, self.input_dim).normal_()
        device = next(self.parameters()).device
        noise = noise.to(device)
        if cond_x is not None:
            cond_x = cond_x.to(device)
        samples = self.forward(noise, cond_x, mode='inverse')[0]
        return samples
    
class MLP(nn.Module):
    def __init__(self, units, layers, in_dim, out_dim, activation='ReLU'):
        super(MLP, self).__init__()
        self.activation = {'ReLU': nn.ReLU(), 'Sigmoid': nn.Sigmoid(), 'Tanh': nn.Tanh()}[activation]
        self.module_list = nn.ModuleList()
        self.module_list.append(nn.Linear(in_dim, units))
        self.module_list.append(self.activation)
        for _ in range(layers - 2):
            self.module_list.append(nn.Linear(units, units))
            self.module_list.append(self.activation)
        self.module_list.append(nn.Linear(units, out_dim))
        
    def forward(self, x):
        for module in self.module_list:
            x = module(x)
        return x

class Flow_based_Bayesian_Filter:
    def __init__(self, arch_params, m, n, train_loader, test_loader, device):
    
        self.m = m
        self.n = n
        # NFs
        self.Tx, self.Ty = self.Build_NF_Tx_Ty(m, n, arch_params, device)
        self.A_net, self.B_net = self.Build_AB_nets(m, n, arch_params)
        self.C, self.D = self.Build_CD(m, n)
        # to device
        self.A_net = self.A_net.to(device)
        self.B_net = self.B_net.to(device)
        self.Px = 0.5 * torch.ones(self.m).to(device) # diagnoal
        self.C = self.C.to(device)
        self.D = self.D.to(device)
        self.Py = 0.5 * torch.ones(self.n).to(device) # diagnoal
        
        self.FBF_initialize(device)
        
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss_train = []
        self.loss_test = []
        
    def FBF_initialize(self, device):
        self.params_NFs = [p for p in self.Tx.parameters() if p.requires_grad] + \
                          [p for p in self.Ty.parameters() if p.requires_grad]
        for p in self.params_NFs:
            if p.ndimension() == 1:
                p.data = 0.1 * nn.init.xavier_normal_(p.data.unsqueeze(-1)).squeeze(-1).to(device)
            else:
                p.data = 0.1 * nn.init.xavier_normal_(p.data).to(device)
    
    def Build_NF_Tx_Ty(self, m, n, arch_params, device):
        Tx_units = arch_params['Tx_units']
        Tx_layers = arch_params['Tx_layers']
        Ty_units = arch_params['Ty_units']
        Ty_layers = arch_params['Ty_layers']
        Tx = RealNVP(input_dim=m, hid_dim=Tx_units, n_layers=Tx_layers).to(device)
        Ty = RealNVP(input_dim=n, hid_dim=Ty_units, n_layers=Ty_layers).to(device)
        return Tx, Ty

    def Build_AB_nets(self, m, n, arch_params):
        A_net_units = arch_params['A_net_units']
        A_net_layers = arch_params['A_net_layers']
        B_net_units = arch_params['B_net_units']
        B_net_layers = arch_params['B_net_layers']
        
        A_net = MLP(A_net_units, A_net_layers, n, m)
        B_net = MLP(B_net_units, B_net_layers, n, m ** 2)

        self.A_net_params = [p for p in A_net.parameters() if p.requires_grad]
        self.B_net_params = [p for p in B_net.parameters() if p.requires_grad]
        return A_net, B_net
    
    def Build_CD(self, m, n):
        return 0.1 * torch.randn(1, n), 0.1 * torch.randn(m, n)
        
    def FBF_compile(self, n_epochs, lr=5e-4, momentum = 0.9, decay = 0.99, final_decay=1e-2):
        trainable_params = self.params_NFs + self.A_net_params + self.B_net_params + \
                           [self.Px, self.C, self.D, self.Py]
        self.optimizer = optim.Adam(trainable_params, lr=lr, betas=(momentum, decay), eps=1e-7)
        gamma = (final_decay)**(1./n_epochs)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=gamma)
    
    def log_prob_state(self, zx_old, zx_new, A, B, Px_var, log_det_J):
        mean_zx_series = A + torch.matmul(B, zx_old)
        mean_zx_series = mean_zx_series.squeeze(-1)
        data_zx_series = zx_new
        var_zx_series = data_zx_series - mean_zx_series
        Px_var_inv = torch.inverse(Px_var)
        
        log_exp = torch.sum(var_zx_series ** 2 @ Px_var_inv, dim=1)
        Px_var_diag = torch.diag(Px_var)
        log_ll_state = -0.5 * torch.log(Px_var_diag.prod()) - 0.5 * log_exp + log_det_J.flatten()
        return log_ll_state.mean()
    
    def log_prob_obs(self, zx_old, zy_new, C, D, Py_var, log_det_J):
        zx_old = zx_old.squeeze(-1)
        mean_zy_series = C + torch.matmul(zx_old, D)
        data_zy_series = zy_new
        var_zy_series = data_zy_series - mean_zy_series
        Py_var_inv = torch.inverse(Py_var)
    
        log_exp = torch.sum(var_zy_series ** 2 @ Py_var_inv, dim=1)
        Py_var_diag = torch.diag(Py_var)
        log_ll_obs = -0.5 * torch.log(Py_var_diag.prod()) - 0.5 * log_exp + log_det_J.flatten()
        return log_ll_obs.mean()
    
    def print_logs(self, epoch, loss_value, loss_x, loss_y, training=True):
        if training == True:
            print("Average train loss: %.6f,\t loss state: %.6f,\t loss obs: %.6f" % (loss_value, loss_x, loss_y))
        else:
            print("Average test loss: %.6f,\t loss state: %.6f,\t loss obs: %.6f" % (loss_value, loss_x, loss_y))
    
    def running_steps(self, x_old, x_new, y_new):
        zx_old = self.Tx(x_old.reshape(-1, self.m))[0].unsqueeze(-1)
        zx_new, log_det_J_Tx = self.Tx(x_new.reshape(-1, self.m))
        zy_new, log_det_J_Ty = self.Ty(y_new.reshape(-1, self.n))
        
        A = self.A_net(y_new.reshape(-1, self.n)).reshape(-1, self.m).unsqueeze(-1)
        B = self.B_net(y_new.reshape(-1, self.n)).reshape(-1, self.m, self.m)
        Px_var = torch.diag(F.softplus(self.Px)).to(device)
        C, D = self.C, self.D
        Py_var = torch.diag(F.softplus(self.Py)).to(device)

        loss_state = - self.log_prob_state(zx_old, zx_new, A, B, Px_var, log_det_J_Tx)
        loss_obs = - self.log_prob_obs(zx_old, zy_new, C, D, Py_var, log_det_J_Ty)
        loss = loss_state + loss_obs
        return loss_state, loss_obs, loss
    
    def train(self, epoch, device):
        running_loss = 0.
        running_state_loss = 0.
        running_obs_loss = 0.
        
        for batch_idx, (x, y) in enumerate(self.train_loader):
            batch_idx += 1
            
            self.optimizer.zero_grad()
            x, y = x.float().to(device), y.float().to(device)
            x_old = x[:, 0:-1]
            x_new, y_new = x[:, 1:], y[:, 1:]
            
            loss_state, loss_obs, loss = self.running_steps(x_old, x_new, y_new)
            running_loss += loss.item()
            running_state_loss += loss_state.item()
            running_obs_loss += loss_obs.item()
            
            loss.backward()
            self.optimizer.step()

        mean_loss = running_loss / batch_idx
        mean_state_loss = running_state_loss / batch_idx
        mean_obs_loss = running_obs_loss / batch_idx
        
        if epoch % 1 == 0:
            print("==> Epoch:", epoch)
            self.print_logs(epoch, mean_loss, mean_state_loss, mean_obs_loss)
        self.loss_train.append(mean_loss)
    
    def test(self, epoch, device):
        running_loss = 0.
        running_state_loss = 0.
        running_obs_loss = 0.
        
        for batch_idx, (x, y) in enumerate(self.test_loader):
            batch_idx += 1
            
            x, y = x.float().to(device), y.float().to(device)
            x_old = x[:, 0:-1]
            x_new, y_new = x[:, 1:], y[:, 1:]
            
            with torch.no_grad():
                loss_state, loss_obs, loss = self.running_steps(x_old, x_new, y_new)
                running_loss += loss.item()
                running_state_loss += loss_state.item()
                running_obs_loss += loss_obs.item()

        mean_loss = running_loss / batch_idx
        mean_state_loss = running_state_loss / batch_idx
        mean_obs_loss = running_obs_loss / batch_idx
        
        if epoch % 1 == 0:
            self.print_logs(epoch, mean_loss, mean_state_loss, mean_obs_loss)
        self.loss_test.append(mean_loss)
        
    def save_model(self, model_path):
        torch.save(self.Tx, model_path + '/Tx.pt')
        torch.save(self.Ty, model_path + '/Ty.pt')
        torch.save(self.A_net, model_path + '/A_net.pt')
        torch.save(self.B_net, model_path + '/B_net.pt')
        torch.save(self.C, model_path + '/C')
        torch.save(self.D, model_path + '/D')
        torch.save(self.Px, model_path + '/Px')
        torch.save(self.Py, model_path + '/Py')
        
    def load_model(self, model_path, device):
        self.Tx = torch.load(model_path + '/Tx.pt', map_location=device)
        self.Ty = torch.load(model_path + '/Ty.pt', map_location=device)
        self.A_net = torch.load(model_path + '/A_net.pt', map_location=device)
        self.B_net = torch.load(model_path + '/B_net.pt', map_location=device)
        self.C = torch.load(model_path + '/C', map_location=device)
        self.D = torch.load(model_path + '/D', map_location=device)
        self.Px = torch.load(model_path + '/Px', map_location=device)
        self.Py = torch.load(model_path + '/Py', map_location=device)
    
    def calc_ensemble(self, ensemble_size, measurement, T, device):
        measure_data = measurement.float().to(device)
        zy_new_all = self.Ty(measure_data)[0]
        
        A = self.A_net(measure_data).reshape(-1, self.m).unsqueeze(-1)
        B = self.B_net(measure_data).reshape(-1, self.m, self.m)
        Px_var = torch.diag(F.softplus(self.Px))
        C, D = self.C, self.D
        Py_var = torch.diag(F.softplus(self.Py))
        Py_var_inv = torch.inverse(Py_var)
        
        D_Py_inv = D @ Py_var_inv
        D_Py_inv_D_T = D_Py_inv @ D.T
        
        ######### init samples #########
        m_0 = 1.1 * torch.ones(self.m).to(device)
        P_0 = 0.1 * torch.eye(self.m).to(device)
        init_Gaussian = distributions.MultivariateNormal(loc=m_0, covariance_matrix=P_0)
        x_old_sample = init_Gaussian.rsample([ensemble_size])
        zx_old_sample = self.Tx(x_old_sample)[0].T
        zx_old_mean = zx_old_sample.mean(1)
        zx_old_var = torch.diag(zx_old_sample.std(1))

        ensemble_sample_set = torch.zeros([T-1, ensemble_size, self.m])
#         pbar = trange(1, T)
#         for t in pbar:
        for t in range(1, T):
            ######### P_smooth #########
            zy_new = zy_new_all[t:t+1]
            zx_old_var_inv = torch.inverse(zx_old_var)
            zx_smooth_old_var = torch.inverse(zx_old_var_inv + D_Py_inv_D_T)
            zx_smooth_old_mean = zx_smooth_old_var @ (zx_old_var_inv @ zx_old_mean.reshape(-1, 1) + \
                                 D_Py_inv @ (zy_new - C)) 
            ######### P_filter #########
            zx_filter_new_mean = A[t] + B[t] @ zx_smooth_old_mean
            zx_filter_new_var = B[t].T @ zx_smooth_old_var @ B[t] + Px_var
            zx_filter_new_Gaussian = distributions.MultivariateNormal(loc=zx_filter_new_mean.flatten(), covariance_matrix=zx_filter_new_var)
            zx_filter_new_sample = zx_filter_new_Gaussian.rsample([ensemble_size])
            
            x_new_sample = self.Tx(zx_filter_new_sample, mode='inverse')[0]
            ensemble_sample_set[t-1] = x_new_sample
            zx_old_var = zx_filter_new_var
            zx_old_mean = zx_filter_new_mean
        return ensemble_sample_set
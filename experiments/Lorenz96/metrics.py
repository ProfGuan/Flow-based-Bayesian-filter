import numpy as np

from tqdm import tqdm, trange
from properscoring import crps_ensemble as crps_lib


def Calc_root_mean_square_error(s_true, s_pred):
    N_s = s_true.shape[1]
    return np.sqrt((np.linalg.norm(s_true - s_pred, 2, axis=1) ** 2).mean()/N_s)

def Calc_sample_covariance(s_ensemble_sample):
    ensemble_size = s_ensemble_sample.shape[1] # (T, N_ensemble, m)
    s_mean = s_ensemble_sample.mean(axis=1, keepdims=True)
    s_dev = s_ensemble_sample - s_mean
    return 1 / (ensemble_size - 1) * np.matmul(s_dev.transpose(0, 2, 1), s_dev)

def Calc_spread(s_ensemble_sample):
    N_s = s_ensemble_sample.shape[2]
    P_plus = Calc_sample_covariance(s_ensemble_sample)
    Tr_P_plus = np.array([np.trace(P_plus[i]) for i in range(len(P_plus))])
    return np.sqrt(Tr_P_plus.mean()/N_s)

def FBF_Rmse_test_set(x_test, y_test):
    print("Filtering over all test sets ...")
    print("Calculating Rmse ...")
    ensemble_size = 500
    Rmse_test = np.zeros(N_test)
    pbar = trange(N_test)
    for i in pbar:
        s_true = x_test[i][1:T].cpu().detach().numpy()
        data = y_test[i][0:T]
        s_ensemble_sample = model.calc_ensemble(ensemble_size, data.to(device), T, device).cpu().detach().numpy()
        s_pred = s_ensemble_sample.mean(1)
        Rmse_test[i] = Calc_root_mean_square_error(s_true, s_pred)
    return Rmse_test

def rbf_kernel(distance, sigma=2):
    """
    Computes the Radial Basis Function (RBF) kernel between two vectors x and y.
    """
    return np.exp(-distance / (2 * sigma ** 2))

def MMD_RBF(x, y):
    xx, yy, zz = x @ x.T, y @ y.T, x @ y.T
    N = len(x)
    rx = np.tile(np.diag(xx).reshape(1, -1), (N, 1))
    ry = np.tile(np.diag(yy).reshape(1, -1), (N, 1))

    dxx = rx.T + rx - 2.*xx
    dyy = ry.T + ry - 2.*yy
    dxy = rx.T + ry - 2.*zz

    XX, XY, YY = (np.zeros(xx.shape), np.zeros(xx.shape), np.zeros(yy.shape))
    XX += rbf_kernel(dxx)
    XY += rbf_kernel(dxy)
    YY += rbf_kernel(dyy)
    return np.mean(XX - 2.*XY + YY)

def MMD_RBF_alongtime(x_true, post_samples):
    N_sample = post_samples.shape[1]
    T = len(x_true)
    MMD = np.zeros([T])
    
    for t in range(T): 
        x_true_t = np.tile(x_true[t:t+1], (N_sample, 1))
        post_sample_t = post_samples[t]
        MMD[t] = MMD_RBF(post_sample_t, x_true_t)
    return MMD.mean()

def MMD_test_set(x_test, y_test):
    N_test = len(x_test)
    MMD_test = np.zeros([N_test])
    ensemble_size = 500
    pbar = trange(N_test)
    for i in pbar:
        s_true = x_test[i][1:T].cpu().detach().numpy()
        data = y_test[i][0:T]
        s_ensemble_sample = model.calc_ensemble(ensemble_size, data.to(device), T, device).cpu().detach().numpy()
        MMD_test[i] = MMD_RBF_alongtime(s_true, s_ensemble_sample)
    return MMD_test       

def calc_CRPS(x_true, post_samples):
    crps_value = np.zeros_like(x_true)
#     pbar = trange(len(x_true))
#     for i in pbar:
    for i in range(len(x_true)):
        crps_value[i] = crps_lib(x_true[i], post_samples[i].T)
    return crps_value.mean()

def CRPS_test_set(x_test, y_test):
    N_test = len(x_test)
    CRPS_test = np.zeros([N_test])
    ensemble_size = 500
                                                
    pbar = trange(N_test)
    for (j, idx) in enumerate(pbar):
        s_true = x_test[idx][1:T].cpu().detach().numpy()
        data = y_test[idx][0:T]
        s_ensemble_sample = model.calc_ensemble(ensemble_size, data.to(device), T, device).cpu().detach().numpy()
        CRPS_test[j] = calc_CRPS(s_true, s_ensemble_sample)
    return CRPS_test

















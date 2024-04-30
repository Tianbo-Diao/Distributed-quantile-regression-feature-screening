import numpy as np
from numba import njit

# loading other Python files
import fun
from conquer import low_dim, high_dim

# opt is the default parameters
opt = {'phi': 0.1, 'gamma': 1.25, 'max_iter': 5e2, 'tol': 1e-2}

# select_fun is the screening function
# @njit
def select_fun(X, Y, m, n, h, tau, kernel, k_index,  beta_initial, deriv_first_initial,  deriv_total_initial):
    beta0 = beta_initial
    phi = np.linalg.norm(X[0], ord=np.inf)**2/n
    # phi = opt['phi']
    ro =1
    count = 0
    while count <= opt['max_iter'] and ro > opt['tol'] :
        gradbeta0 = fun.deriv_obj_fun(X, Y, m, n, h, tau, beta0, kernel, deriv_first_initial,  deriv_total_initial)
        loss_eval0 = fun.obj_fun(X, Y, m, n, h, tau, beta0, kernel, deriv_first_initial, deriv_total_initial)
        beta1 = fun.hard_thresholding( beta0 - gradbeta0 / phi, k_index )
        r0 = np.linalg.norm(beta1 - beta0)
        loss_proxy = loss_eval0 + (beta1 - beta0) @ gradbeta0 + 0.5 * phi * r0**2
        loss_eval1 = fun.obj_fun(X, Y, m, n, h, tau, beta1, kernel, deriv_first_initial, deriv_total_initial)

        while loss_proxy < loss_eval1:
            phi *= opt['gamma']
            beta1 = fun.hard_thresholding(beta0 - gradbeta0 / phi, k_index)
            r0 = np.linalg.norm(beta1 - beta0)
            loss_proxy = loss_eval0 + (beta1 - beta0) @ gradbeta0 + 0.5 * phi * r0** 2
            loss_eval1 = fun.obj_fun(X, Y, m, n, h, tau, beta1, kernel, deriv_first_initial, deriv_total_initial)
        beta0, phi = beta1, np.linalg.norm(X[0], ord=np.inf)**2/n
        count += 1

    return {'beta': beta1, 'count': count, 'loss': loss_eval1, 'mse': r0}


# first_select is the screening result
def first_select(X, Y, m, n, p, h, tau, kernel, k_index):
    sqr = high_dim(X[0], Y[0], intercept=False)
    l1_model = sqr.l1(tau=tau)

    # beta_initial is the initial value of beta, lasso is the default penalty
    beta_initial = l1_model['beta']

    # deriv_loss_initial is the derivative of the loss function ∇L_i(β)
    deriv_loss_initial = fun.deriv_i(X, Y, m, n, h, tau, beta_initial, kernel)

    # deriv_loss_N is the summation of the derivative of the loss function: ∇L_N(β)
    deriv_loss_N = (1 / m) * np.sum(deriv_loss_initial, axis=0)

    ff = [0] * k_index
    for i in range(k_index):
        ff[i] = select_fun(X, Y, m, n, h, tau, kernel, i+1, beta_initial, deriv_loss_initial, deriv_loss_N)

    dev = np.zeros(k_index)
    for i in range(k_index):
        dev[i] = np.log(fun.cqr_check_sum(X, Y, m, tau, ff[i]['beta'])) + (i + 1) * np.log(np.log(m * n)) * np.log(p) / (n * m)
        # dev[i] = np.log( fun.obj_fun(X, Y, m, n, h, tau, ff[i]['beta'], kernel, deriv_loss_initial, deriv_loss_N) ) + (i + 1) * np.log(np.log(m * n)) * np.log(p) / (n * m)

    dev_index = np.argmin(dev)
    index = np.where(ff[dev_index]['beta'].flatten() != 0)[0]
    beta = ff[dev_index]['beta'].flatten()
    return {'index': index, 'beta_value': beta}

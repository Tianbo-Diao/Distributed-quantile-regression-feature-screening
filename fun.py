import numpy as np
import scipy as sp
from numba import njit


# check loss function 𝜌_𝜏(x)
def check_fun(tau, x):
    return np.abs(x) / 2 + (tau - 0.5) * x


# summation of check loss function
# tau is 1-d array, beta is column vector
def cqr_check_sum(X, Y, m, tau, beta):
    machine_index = np.zeros(m)
    for i in range(m):
        residuals = Y[i] - X[i] @ beta
        machine_index[i] = np.mean( check_fun(tau, residuals) )
    return np.mean(machine_index)


# kernel funtion K(u)
def kernel_fun(x, kernel):
    if kernel == 'Gaussian':
        return  sp.stats.norm.pdf(x)
    elif kernel == 'Laplacian':
        return np.exp(- np.abs(x)) / 2
    elif kernel == 'Logistic':
        return 1 / (np.exp(x) + np.exp(-x) + 2)
    elif kernel == 'Uniform':
        return np.where(abs(x) <= 1, 0.5, 0)
    elif kernel == 'Epanechnikov':
        return np.where( abs(x) <= 1, 0.75 * (1 - x ** 2), 0 )


# smooth convolution function 𝓁_h(u)
# tau is 1-d array
def convu_fun(h, tau, x, kernel):
    if kernel == 'Gaussian':
        G = lambda x: x * sp.stats.norm.pdf(x) + x * ( 1 - 2 * sp.stats.norm.cdf(-x) )
        return  h / 2 * G(x / h) + (tau - 0.5) * x
    elif kernel == 'Laplacian':
        return  check_fun(tau, x) + h / 2 * np.exp( -np.abs(x)/h  )
    elif kernel == 'Logistic':
        return tau * x + h * np.log( 1 + np.exp(-x/h) )
    elif kernel == 'Uniform':
        U = lambda x: np.where( abs(x) <= 1, 0.5 + 0.5 *  x ** 2, np.abs(x) )
        return h / 2 * U(x / h) + (tau - 0.5) * x
    elif kernel == 'Epanechnikov':
        E = lambda x: np.where( abs(x) <= 1,  0.75 * x ** 2 - (1/8) * x ** 4 + (3/8), np.abs(x) )
        return h / 2 * E(x / h) + (tau - 0.5) * x


# integrated kernel function:  \int_{−∞}^{u} K(t) dt
def kernel_integral(x, kernel):
    if kernel == 'Gaussian':
        return sp.stats.norm.cdf(x)
    elif kernel == 'Laplacian':
        return 0.5 + 0.5 * np.sign(x) * ( 1 - np.exp(-np.abs(x)) )
    elif kernel == 'Logistic':
        return 1 / (1 + np.exp(-x))
    elif kernel == 'Uniform':
        return np.where(x > 1, 1, 0) + np.where(abs(x) <= 1, 0.5 * (1 + x), 0)
    elif kernel == 'Epanechnikov':
        return 0.25 * (2 + 3 * x / 5 ** 0.5 - (x / 5 ** 0.5) ** 3) * (abs(x) <= 5 ** 0.5) + (x > 5 ** 0.5)



# hard thresholding function
def hard_thresholding(x, k_retain):
    ord = np.sort(np.abs(x),axis=None)[::-1]
    a = x * ( np.abs(x) >= ord[k_retain - 1] )
    return a

# soft thresholding function
def soft_thresholding(x, lambda_value):
    return np.sign(x) * np.maximum(np.abs(x) - lambda_value, 0)



# deriv_loss_i 是 损失函数的导数 ∇L_i(β)，X ,Y 是列表
# gradient_beta is a list of gradient vector, each element is a column vector R^p
def deriv_i(X, Y, m, n, h, tau, beta, kernel):
    gradient_beta = [0] * m
    kernel_term = [0] * m
    for i in range(m):
        kernel_term[i] = kernel_integral( (X[i] @ beta - Y[i]) / h, kernel ) - tau
        gradient_beta[i] = X[i].T @ kernel_term[i] / n
    return gradient_beta


# objective function, X and Y are lists
def obj_fun(X, Y, m, n, h, tau, beta, kernel, deriv_first_initial,  deriv_total_initial):
    residuals = Y[0] - X[0] @ beta
    first_term = np.sum( convu_fun(h, tau, residuals, kernel) ) / n
    second_term =  beta @ (deriv_first_initial[0] - deriv_total_initial)
    return  first_term  - second_term

# derivative of objective function, X and Y are lists
def deriv_obj_fun(X, Y, m, n, h, tau, beta, kernel, deriv_first_initial,  deriv_total_initial):
    kernel_term = kernel_integral((X[0] @ beta - Y[0]) / h, kernel) - tau
    first_term = X[0].T @ kernel_term / n
    second_term =  deriv_first_initial[0] - deriv_total_initial
    return  first_term  - second_term
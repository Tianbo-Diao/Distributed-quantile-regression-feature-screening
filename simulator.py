import numpy as np
from numba import njit

rng = np.random.default_rng(66)

# simulate X, X is a list
# @njit
def simu_X(m, n, p, example):
    list_x = [0] * m
    if example == 'iid':
        for i in range(m):
            list_x[i] = rng.normal(size=(n, p))
    if example == 'hete':
        cov_mat = 0.5 ** np.abs( np.arange(1, p + 1).reshape(-1,1) - np.arange(1, p + 1) )
        L = np.linalg.cholesky(cov_mat)
        for i in range(m):
            list_x[i] = rng.normal( size=(n, p) ) @ L.T
    elif example == 'wang2009':
        for i in range(m):
            z = rng.normal(size=(n, p))
            w = rng.normal(size=(n, p))
            x1 = (z + w) / np.sqrt(2)
            x2 = (z + np.sum(z[:, :5], axis=1).reshape(-1,1) )/ 2
            list_x[i] = np.hstack( (x1[:, :5], x2[:, 5:]) )
    return list_x


# simulate Y, Y is a list
def simu_Y(list_x, m, n, beta, error_type):
    list_y = [0] * m
    match error_type:
        case 'normal':
            for i in range(m):
                list_y[i] =  list_x[i] @ beta + rng.normal(size=n)
        case 't':
            for i in range(m):
                list_y[i] = list_x[i] @ beta + rng.standard_t(df=3, size=n)
    return list_y


# value of beta
def beta(n, p, type):
    beta_value = np.zeros(p)
    match type:
        case 'iid':
            u_value = rng.uniform(size=1)
            beta_value[:3] = np.array([ 1, 2 * np.where(u_value > 0.5, 1, 0)[0], np.exp( u_value**2 )[0] ])
            return beta_value
        case 'hete':
            beta_value[[0, 3, 6, 9]] = (-1) ** rng.binomial(n=1, p=0.6, size=4) * (2 + np.abs(rng.normal(size=4)))
            return beta_value
        case 'wang2009':
            return np.array([2, 4, 6, 8, 10] + [0] * (p - 5))
        case 'diverge':
            p0 = int( np.floor(np.log(n)) )
            beta_value[rng.choice(p, p0, replace=False)] = (-1) ** rng.binomial(n=1, p=0.4, size=p0) * ( 4 * np.log(n) / (n ** (1 / 4)) + np.abs(rng.normal(size=p0)) )
            return beta_value

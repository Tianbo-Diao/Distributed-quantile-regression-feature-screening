import numpy as np
from scipy import stats
from numba import njit

rng = np.random.default_rng(66)

# Pearson correlation function
def pearson_fun(x1, x2, x3, x4, x5):
    return np.abs((x1 - x2 * x3) / np.sqrt((x4 - x2 ** 2) * (x5 - x3 ** 2)))


# Pearson correlation method，X and Y are lists
def pearson_com(X, Y, N, m, p):
    pearson_mid1 = np.zeros((m, p))
    pearson_mid2 = np.zeros((m, p))
    pearson_mid3 = np.zeros((m, p))
    pearson_mid4 = np.zeros((m, p))
    pearson_mid5 = np.zeros((m, p))

    for i in range(m):
        pearson_mid1[i, :] = np.sum( X[i] * Y[i].reshape(-1, 1), axis=0 )
        pearson_mid2[i, :] = np.sum(X[i], axis=0)
        pearson_mid3[i, :] = np.repeat( np.sum(Y[i]), p )
        pearson_mid4[i, :] = np.sum(X[i] ** 2, axis=0)
        pearson_mid5[i, :] = np.repeat( np.sum(Y[i] ** 2), p )

    pearson_x1 = np.sum(pearson_mid1, axis=0) / N
    pearson_x2 = np.sum(pearson_mid2, axis=0) / N
    pearson_x3 = np.sum(pearson_mid3, axis=0) / N
    pearson_x4 = np.sum(pearson_mid4, axis=0) / N
    pearson_x5 = np.sum(pearson_mid5, axis=0) / N

    pearson_value = pearson_fun(pearson_x1, pearson_x2, pearson_x3, pearson_x4, pearson_x5)
    return pearson_value


# X and Y are lists
def pearson_threshold(X, Y, N, m, n, p, q_new):
    X_new = [
        rng.multivariate_normal(
            mean=np.zeros(q_new), cov=np.eye(q_new), size=n)
        for _ in range(m)
    ]
    threshold_new = pearson_com(X_new, Y, N, m, q_new)
    threshold_value = pearson_com(X, Y, N, m, p)

    index = np.where(threshold_value >= np.max(threshold_new))[0]
    modified_vector = np.zeros(p)
    modified_vector[index] = threshold_value[index]
    beta_value = modified_vector

    return {'index': index, 'beta_value': beta_value}


# Kendall τ function
def Kend_fun(x1):
    return np.abs(x1 - 1 / 4)


# Kendall τ method，X and Y are lists
# @njit
def kend_com(X, Y, N, m, n, p):
    Kendall_x1 = np.zeros(p)
    for q in range(p):
        mid = np.zeros(m)
        for i in range(m):
            mid[i] = np.sum((X[i][:, q].reshape(-1, 1) < X[i][:, q]) * (Y[i].reshape(-1,1) < Y[i]))
        Kendall_x1[q] = np.sum(mid) / (m * n * (n - 1))
    Kend_value = Kend_fun(Kendall_x1)
    return Kend_value


# X and Y are lists
# @njit
def Kendall_threshold(X, Y, N, m, n, p, q_new):
    X_new = [
        rng.multivariate_normal(
            mean=np.zeros(q_new), cov=np.eye(q_new), size=n )
        for _ in range(m)
    ]

    threshold_new = kend_com(X_new, Y, N, m, n, q_new)
    threshold_value = kend_com(X, Y, N, m, n, p)

    index = np.where(threshold_value >= np.max(threshold_new))[0]
    modified_vector = np.zeros(p)
    modified_vector[index] = threshold_value[index]
    beta_value = modified_vector

    return {'index': index, 'beta_value': beta_value}


# SIRS correlation method，X and Y are lists
# @njit
def SIRS_com(X, Y, N, m, n, p):
    SIRS_x1 = np.zeros(p)
    for q in range(p):
        mid = np.zeros(m)
        for i in range(m):
            mid1 = Y[i].reshape(-1,1) < Y[i]
            mid2 = np.sum( mid1 * X[i][:, q].reshape(-1, 1), axis=0 )
            mid[i] = np.sum(mid2 ** 2)
        SIRS_x1[q] = np.sum(mid) / (m * n * (n - 1) * (n - 2))
    return SIRS_x1


def SIRS_threshold(X, Y, N, m, n, p, q_new):
    X_new = [
        rng.multivariate_normal(
            mean=np.zeros(q_new), cov=np.eye(q_new), size=n)
        for _ in range(m)
    ]
    X_new = [ stats.zscore(X_new[i], ddof=1) for i in range(m) ]
    X = [ stats.zscore(X[i], ddof=1) for i in range(m) ]
    Y = [ stats.zscore(Y[i], ddof=1) for i in range(m) ]

    threshold_new = SIRS_com(X_new, Y, N, m, n, q_new)
    threshold_value = SIRS_com(X, Y, N, m, n, p)

    index = np.where(threshold_value >= np.max(threshold_new))[0]
    modified_vector = np.zeros(p)
    modified_vector[index] = threshold_value[index]
    beta_value = modified_vector

    return {'index': index, 'beta_value': beta_value}


# DC function
@njit
def DC_fun(x1, x2, x3, x4, x5, x6, x7, x8):
    value = (x1 + x2 * x3 - 2 * x4) / np.sqrt( (x5 + x2 ** 2 - 2 * x6) * (x7 + x3 ** 2 - 2 * x8) )
    return value


# DC method，X and Y are lists
# @njit
def DC_com(X, Y, N, m, n, p):
    DC_x1 = np.zeros(p)
    DC_x2 = np.zeros(p)
    DC_x3 = np.zeros(p)
    DC_x4 = np.zeros(p)
    DC_x5 = np.zeros(p)
    DC_x6 = np.zeros(p)
    DC_x7 = np.zeros(p)
    DC_x8 = np.zeros(p)

    for q in range(p):
        mid1 = np.zeros(m)
        mid2 = np.zeros(m)
        mid3 = np.zeros(m)
        mid4 = np.zeros(m)
        mid5 = np.zeros(m)
        mid6 = np.zeros(m)
        mid7 = np.zeros(m)
        mid8 = np.zeros(m)

        for i in range(m):
            x_q = X[i][:, q]
            y_i = Y[i]

            mid1[i] = np.sum(  np.abs(x_q.reshape(-1,1) - x_q)  * np.abs( y_i.reshape(-1,1) - y_i )   )
            mid2[i] = np.sum( np.abs(y_i.reshape(-1,1) - y_i) )
            mid3[i] = np.sum( np.abs(x_q.reshape(-1,1) - x_q) )
            mid4[i] = np.sum(
                np.sum( np.abs(x_q.reshape(-1,1) - x_q), axis=0 )
                * np.sum( np.abs(y_i.reshape(-1,1) - y_i), axis=0 )
            )
            mid5[i] = np.sum( (y_i.reshape(-1,1) - y_i)**2 )
            mid6[i] = np.sum(
                 np.sum( np.abs(y_i.reshape(-1,1) - y_i), axis=0 )**2
            )
            mid7[i] = np.sum( (x_q.reshape(-1,1) - x_q)**2 )
            mid8[i] = np.sum(
                 np.sum(np.abs(x_q.reshape(-1,1) - x_q), axis=0)**2
            )

        DC_x1[q] = np.sum(mid1) / (m * n * (n - 1))
        DC_x2[q] = np.sum(mid2) / (m * n * (n - 1))
        DC_x3[q] = np.sum(mid3) / (m * n * (n - 1))
        DC_x4[q] = np.sum(mid4) / (m * n * (n - 1) * (n - 2))
        DC_x5[q] = np.sum(mid5) / (m * n * (n - 1))
        DC_x6[q] = np.sum(mid6) / (m * n * (n - 1) * (n - 2))
        DC_x7[q] = np.sum(mid7) / (m * n * (n - 1))
        DC_x8[q] = np.sum(mid8) / (m * n * (n - 1) * (n - 2))

    return DC_fun(DC_x1, DC_x2, DC_x3, DC_x4, DC_x5, DC_x6, DC_x7, DC_x8)


# X and Y are lists
def DC_threshold(X, Y, N, m, n, p, q_new):
    X_new = [
        rng.multivariate_normal(
            mean=np.zeros(q_new), cov=np.eye(q_new), size=n)
        for _ in range(m)
    ]
    threshold_new = DC_com(X_new, Y, N, m, n, q_new)
    threshold_value = DC_com(X, Y, N, m, n, p)

    index = np.where(threshold_value >= np.max(threshold_new))[0]
    modified_vector = np.zeros(p)
    modified_vector[index] = threshold_value[index]
    beta_value = modified_vector

    return {'index': index, 'beta_value': beta_value}


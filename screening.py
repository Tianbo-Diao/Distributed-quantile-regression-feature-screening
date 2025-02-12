import numpy as np
from concurrent.futures import ThreadPoolExecutor

from sklearn.model_selection import KFold
from scipy.optimize import minimize

rng = np.random.default_rng(66)

# loading other Python files
import fun
from conquer import high_dim

# opt is the default parameters
opt = {'phi': 0.1, 'gamma': 1.25, 'max_iter': 5e2, 'tol': 1e-2}

# select_fun is the screening function
# @njit
def select_fun(X, Y, m, n, h, tau, beta_initial, kernel, k_index):
    deriv_first_initial = fun.deriv_i(X, Y, m, n, h, tau, beta_initial, kernel)
    deriv_total_initial = (1 / m) * np.sum(deriv_first_initial, axis=0)

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



# def final_select(X, Y, m, n, p, h, tau, beta_initial, kernel, k_index):
#     # sqr = high_dim(X[0], Y[0], intercept=False)
#     # l1_model = sqr.l1(tau=tau)
#     #
#     # # beta_initial is the initial value of beta, lasso is the default penalty
#     # beta_initial = l1_model['beta']
#
#     # deriv_loss_initial is the derivative of the loss function ∇L_i(β)
#     # deriv_loss_initial = fun.deriv_i(X, Y, m, n, h, tau, beta_initial, kernel)
#
#     # deriv_loss_N is the summation of the derivative of the loss function: ∇L_N(β)
#     # deriv_loss_N = (1 / m) * np.sum(deriv_loss_initial, axis=0)
#
#     ff = [0] * k_index
#     dev = np.zeros(k_index)
#     for i in range(k_index):
#         ff[i] = select_fun(X, Y, m, n, h, tau, beta_initial, kernel, i+1)
#         dev[i] = np.log(fun.cqr_check_sum(X, Y, m, tau, ff[i]['beta'])) + (i + 1) * np.log(np.log(m * n)) * np.log(p) / (n * m)
#         # dev[i] = np.log( fun.obj_fun(X, Y, m, n, h, tau, ff[i]['beta'], kernel, deriv_loss_initial, deriv_loss_N) ) + (i + 1) * np.log(np.log(m * n)) * np.log(p) / (n * m)
#
#     dev_index = np.argmin(dev)
#     index = np.where(ff[dev_index]['beta'].flatten() != 0)[0]
#     beta = ff[dev_index]['beta'].flatten()
#     return {'index': index, 'beta_value': beta}




# final_select is the screening result
def final_select(X, Y, m, n, p, h, tau, beta_initial, kernel, k_index):
    # Initialize minimum dev and corresponding index
    min_dev = float('inf')  # Infinite initially to guarantee the first comparison will be less
    dev_index = -1  # Placeholder for the index of the minimum value

    # Loop over the range of k_index to compute and compare dev values
    for i in range(k_index):
        # Calculate the current beta value from select_fun
        ff = select_fun(X, Y, m, n, h, tau, beta_initial, kernel, i+1)
        # Compute the dev value for the current index
        current_dev = np.log( fun.cqr_check_sum(X, Y, m, tau, ff['beta']) ) + (i + 1) * np.log(np.log(m * n)) * np.log(p) / (n * m)

        # Track the minimum dev and its corresponding index
        if current_dev < min_dev:
            min_dev = current_dev
            dev_index = i
            best_beta = ff['beta'].flatten()  # Store the beta associated with the minimum dev

    # Get the index positions where beta is non-zero
    index = np.where(best_beta != 0)[0]

    # Return the selected index and beta values
    return {'index': index, 'beta_value': best_beta}



# def compute_dev(X, Y, m, n, p, h, tau, beta_initial, kernel, k):
#     ff_k = select_fun(X, Y, m, n, h, tau, beta_initial, kernel, k + 1)
#     dev_value = np.log(fun.cqr_check_sum(X, Y, m, tau, ff_k['beta'])) + \
#                 (k + 1) * np.log(np.log(m * n)) * np.log(p) / (n * m)
#     return dev_value, ff_k
#
# def final_select(X, Y, m, n, p, h, tau, beta_initial, kernel, k_index):
#     results = []
#
#     # Parallel computation of dev values
#     with ThreadPoolExecutor() as executor:
#         futures = [executor.submit(compute_dev, X, Y, m, n, p, h, tau, beta_initial, kernel, k) for k in range(k_index)]
#         for future in futures:
#             results.append(future.result())
#
#     # Find the minimum dev and its corresponding beta
#     dev_values = [result[0] for result in results]
#     ff_results = [result[1] for result in results]
#     min_index = np.argmin(dev_values)
#
#     best_beta = ff_results[min_index]['beta'].flatten()
#     index = np.where(best_beta != 0)[0]
#
#     return {'index': index, 'beta_value': best_beta}

# Define the check function (quantile loss)
def quantile_loss(y_true, y_pred, tau):
    residuals = y_true - y_pred
    return np.mean(np.maximum(tau * residuals, (tau - 1) * residuals))

# Define the objective function (quantile loss + LASSO penalty)
def objective_function(w, X, y, tau, alpha):
    """Objective function: quantile loss + LASSO penalty."""
    y_pred = X @ w
    return quantile_loss(y, y_pred, tau) + alpha * np.sum(np.abs(w))

# Cross-validation to find the optimal alpha (lambda)
def quantile_lasso_cv(X, y, tau, alphas, n_folds=5):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    alpha_scores = []

    for alpha in alphas:
        fold_scores = []

        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Initial guess for weights
            initial_w = np.zeros(X.shape[1])

            # Optimize the objective function
            result = minimize(
                fun=objective_function,
                x0=initial_w,
                args=(X_train, y_train, tau, alpha),
                method='L-BFGS-B',
                options={'maxiter': 1000}
            )

            # Evaluate on the validation set
            y_pred_val = X_val @ result.x
            val_score = quantile_loss(y_val, y_pred_val, tau)
            fold_scores.append(val_score)

        # Average score across folds for this alpha
        alpha_scores.append(np.mean(fold_scores))

    # Find the alpha with the lowest score
    optimal_alpha = alphas[np.argmin(alpha_scores)]
    return optimal_alpha

# Define a range of alpha (lambda) values to test
alphas = np.logspace(-4, 0, 20)



# CQR is the  LASSO estimator adopted by:
# Communication-constrained distributed quantile regression with optimal statistical guarantees. Tan, Battey, and Zhou, 2022, Journal of Machine Learning  Research.
# @njit
def CQR(X, Y, m, n, N, tau, s, p, beta_initial, kernel):
    bandwidth = {'b': 0.75 * s ** (1 / 2) * (np.log(p) / n) ** (1 / 4), 'h': 0.75 * (s * np.log(p) / N) ** (1 / 4)}
    deriv_first_initial = fun.deriv_i(X, Y, m, n, bandwidth['b'], tau, beta_initial, kernel)
    deriv_first_initial_2 = fun.deriv_i(X, Y, m, n, bandwidth['h'], tau, beta_initial, kernel)
    deriv_total_initial = (1 / m) * np.sum(deriv_first_initial_2, axis=0)

    # lambda_sim = np.array([max(abs(X[0].T.dot(tau - (rng.uniform(0,1, n) <= tau))))
    #                                for b in range(200)]) / n
    # lambda_value = 0.75 * np.quantile(lambda_sim, 0.9)
    # lambda_max = np.linalg.norm(  fun.deriv_obj_fun(X, Y, m, n, bandwidth['b'], tau, np.zeros(p), kernel, deriv_first_initial, deriv_total_initial), ord=np.inf   )
    # lambda_value = 0.005 * lambda_max

    lambda_value = quantile_lasso_cv(X[0], Y[0], tau, alphas)

    beta0 = beta_initial
    phi = np.linalg.norm(X[0], ord=np.inf)**2/n
    # phi = opt['phi']
    ro =1
    count = 0
    while count <= opt['max_iter'] and ro > opt['tol'] :
        gradbeta0 = fun.deriv_obj_fun(X, Y, m, n, bandwidth['b'], tau, beta0, kernel, deriv_first_initial,  deriv_total_initial)
        loss_eval0 = fun.obj_fun(X, Y, m, n, bandwidth['b'], tau, beta0, kernel, deriv_first_initial, deriv_total_initial)
        beta1 = fun.soft_thresholding( beta0 - gradbeta0 / phi, lambda_value )
        r0 = np.linalg.norm(beta1 - beta0)
        loss_proxy = loss_eval0 + (beta1 - beta0) @ gradbeta0 + 0.5 * phi * r0**2
        loss_eval1 = fun.obj_fun(X, Y, m, n, bandwidth['b'], tau, beta1, kernel, deriv_first_initial, deriv_total_initial)

        while loss_proxy < loss_eval1:
            phi *= opt['gamma']
            beta1 = fun.soft_thresholding(beta0 - gradbeta0 / phi, lambda_value)
            r0 = np.linalg.norm(beta1 - beta0)
            loss_proxy = loss_eval0 + (beta1 - beta0) @ gradbeta0 + 0.5 * phi * r0** 2
            loss_eval1 = fun.obj_fun(X, Y, m, n, bandwidth['b'], tau, beta1, kernel, deriv_first_initial, deriv_total_initial)
        beta0, phi = beta1, np.linalg.norm(X[0], ord=np.inf)**2/n
        count += 1

    index = np.where(beta1 != 0)[0]

    return {'index': index, 'beta_value': beta1,  'count': count, 'loss': loss_eval1, 'mse': r0}





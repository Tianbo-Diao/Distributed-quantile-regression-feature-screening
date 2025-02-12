import numpy as np
import pandas as pd
from time import time

# loading other Python files
import simulator
import fun
import screening
import U_fun
from conquer import high_dim

# parameters
N = 2000
p = 5000
tau_value = 0.5
m = 20
n = int(N / m)
k = round(np.log(N) * (N ** (1 / 5)) / 3)
h = np.max([0.05, np.sqrt( tau_value * (1-tau_value) ) * (np.log(p) / n) ** (1 / 4)])
beta_value = simulator.beta(n, p, type='iid')
len_s = np.count_nonzero(beta_value)

Times = 100
# first column is SC, second column is CF, third column is AMS, fourth column is PSR, fifth column is FDR, sixth column is ME, seventh column is Computing Time (QR has first and second computing times)
QR_index = np.zeros([Times, 8])
pearson_index = np.zeros([Times, 7])
Kendall_index = np.zeros([Times, 7])
SIRS_index = np.zeros([Times, 7])
DC_index = np.zeros([Times, 7])
CQR_index = np.zeros([Times, 7])
pearson_racs_index = np.zeros([Times, 7])
kendall_racs_idnex = np.zeros([Times, 7])
SIRS_racs_idnex = np.zeros([Times, 7])
DC_racs_idnex = np.zeros([Times, 7])



for Ti in range(Times):
    # simulate data
    start_time = time()
    popu_X = simulator.simu_X(m, n, p, example='iid')
    popu_Y = simulator.simu_Y(popu_X, m, n, beta = beta_value, error_type='normal')
    print('simulate data time:', time() - start_time)

    # QR screening
    # beta_initial is the initial value of beta, lasso is the default penalty
    start_time = time()
    sqr = high_dim(popu_X[0], popu_Y[0], intercept=False)
    l1_model = sqr.l1(tau=tau_value)
    beta_initial = l1_model['beta']  # LASSO estimator
    # l1_model = sqr.irw(tau=tau_value, penalty="SCAD")
    # beta_initial = l1_model['beta']   # SCAD estimator
    # l1_model = sqr.irw(tau=tau_value, penalty="MCP")
    # beta_initial = l1_model['beta']  # MCP estimator
    print('beta_initial computing time:', time() - start_time)

    start_time = time()
    QR_first = screening.select_fun(popu_X, popu_Y, m, n, h, tau_value, beta_initial,kernel='Gaussian', k_index=k)
    QR_index[Ti, 6] = time() - start_time
    print('QR screening_first time:', QR_index[Ti, 6])

    start_time = time()
    QR_select = screening.final_select(popu_X, popu_Y, m, n, p, h, tau_value, beta_initial, kernel='Gaussian', k_index=k)
    QR_index[Ti, 7] = time() - start_time
    print('QR screening__second time:', QR_index[Ti, 7])

    # CQR_select, CQR_index[Ti, 6] = fun.measure_runtime(screening.CQR, popu_X, popu_Y, m, n, N, tau_value, len_s, p, beta_initial, kernel='Gaussian')

    # U-method screening
    pearson_select, pearson_index[Ti, 6] = fun.measure_runtime(U_fun.pearson_threshold, popu_X, popu_Y, N, m, n, p, 500)
    Kendall_select, Kendall_index[Ti, 6] = fun.measure_runtime(U_fun.kendall_threshold, popu_X, popu_Y, N, m, n, p, 500)
    SIRS_select, SIRS_index[Ti, 6] = fun.measure_runtime(U_fun.SIRS_threshold, popu_X, popu_Y, N, m, n, p, 500)
    DC_select, DC_index[Ti, 6] = fun.measure_runtime(U_fun.DC_threshold, popu_X, popu_Y, N, m, n, p, 500)

    # pearson_racs_select, pearson_racs_index[Ti, 6] = fun.measure_runtime(U_fun.pearson_racs, popu_X, popu_Y, N, m, n, p, q_new=500, partition=5)
    # Kendall_racs_select, kendall_racs_idnex[Ti, 6] = fun.measure_runtime(U_fun.Kendall_racs, popu_X, popu_Y, N, m, n, p, q_new=500, partition=5)
    # SIRS_racs_select, SIRS_racs_idnex[Ti, 6] = fun.measure_runtime(U_fun.SIRS_racs, popu_X, popu_Y, N, m, n, p, q_new=500, partition=5)
    # DC_racs_select, DC_racs_idnex[Ti, 6] = fun.measure_runtime(U_fun.DC_racs, popu_X, popu_Y, N, m, n, p, q_new=500, partition=5)

    beta_index = np.where(beta_value != 0)[0]
    QR_index[Ti, :6] = fun.method_metrics(QR_select, beta_index, beta_value)
    pearson_index[Ti, :6] = fun.method_metrics(pearson_select, beta_index, beta_value)
    Kendall_index[Ti, :6] = fun.method_metrics(Kendall_select, beta_index, beta_value)
    SIRS_index[Ti, :6] = fun.method_metrics(SIRS_select, beta_index, beta_value)
    DC_index[Ti, :6] = fun.method_metrics(DC_select, beta_index, beta_value)
    # CQR_index[Ti, :6] = fun.method_metrics(CQR_select, beta_index, beta_value)

    # pearson_racs_index[Ti, :6] = fun.method_metrics(pearson_racs_select, beta_index, beta_value)
    # kendall_racs_idnex[Ti, :6] = fun.method_metrics(Kendall_racs_select, beta_index, beta_value)
    # SIRS_racs_idnex[Ti, :6] = fun.method_metrics(SIRS_racs_select, beta_index, beta_value)
    # DC_racs_idnex[Ti, :6] = fun.method_metrics(DC_racs_select, beta_index, beta_value)

    if (Ti + 1) % 10 == 0: print(Ti + 1, 'repetitions')


# save results
qr_df= fun.mean_std(QR_index)
pearson_df = fun.mean_std(pearson_index)
kendall_df = fun.mean_std(Kendall_index)
sirs_df = fun.mean_std(SIRS_index)
dc_df = fun.mean_std(DC_index)
cqr_df = fun.mean_std(CQR_index)
# pearson_racs_df = fun.mean_std(pearson_racs_index)
# kendall_racs_df = fun.mean_std(kendall_racs_idnex)
# sirs_racs_df = fun.mean_std(SIRS_racs_idnex)
# dc_racs_df = fun.mean_std(DC_racs_idnex)

# print('qr_indx: \n', qr_df)
# print('pearson_indx: \n', pearson_df)
# print('kendall_indx: \n', kendall_df)
# print('sirs_indx: \n', sirs_df)
# print('dc_indx: \n', dc_df)
# print('cqr_indx: \n', cqr_df)


with pd.ExcelWriter('wang2009_normal_m=5.xlsx') as writer:
    qr_df.to_excel(writer, sheet_name='QR_index', index=False)
    pearson_df.to_excel(writer, sheet_name='pearson_index', index=False)
    kendall_df.to_excel(writer, sheet_name='Kendall_index', index=False)
    sirs_df.to_excel(writer, sheet_name='SIRS_index', index=False)
    dc_df.to_excel(writer, sheet_name='DC_index', index=False)
    # cqr_df.to_excel(writer, sheet_name='CQR_index', index=False)
    # pearson_racs_df.to_excel(writer, sheet_name='pearson_racs_index', index=False)
    # kendall_racs_df.to_excel(writer, sheet_name='Kendall_racs_index', index=False)
    # sirs_racs_df.to_excel(writer, sheet_name='SIRS_racs_index', index=False)
    # dc_racs_df.to_excel(writer, sheet_name='DC_racs_index', index=False)





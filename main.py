import numpy as np
import pandas as pd
from time import time

# loading other Python files
import simulator
import fun
import screening
import U_fun
from conquer import low_dim, high_dim

# parameters
N = 2000
p = 5000
tau_value = 0.5
m = 20
n = int(N / m)
k = round(np.log(N) * (N ** (1 / 5)) / 3)
h = np.max([0.01, np.sqrt( tau_value * (1-tau_value) ) * (np.log(p) / N) ** (1 / 4)])
beta_value = simulator.beta(N, p, type='hete')

Times = 50
# first column is sc, second column is cf, third column is ams, fourth column is psr, fifth column is fdr, sixth column is aee
QR_index = np.zeros([Times, 6])
pearson_index = np.zeros([Times, 6])
Kendall_index = np.zeros([Times, 6])
SIRS_index = np.zeros([Times, 6])
DC_index = np.zeros([Times, 6])

for Ti in range(Times):
    # simulate data
    start_time = time()
    popu_X = simulator.simu_X(m, n, p, example='hete')
    popu_Y = simulator.simu_Y(popu_X, m, n, beta = beta_value, error_type='normal')
    print('simulate data time:', time() - start_time)

    # QR screening
    start_time = time()
    QR_select = screening.first_select(popu_X, popu_Y, m, n, p, h, tau_value, kernel='Gaussian', k_index=k)
    print('QR screening time:', time() - start_time)

    # U-method screening
    start_time = time()
    pearson_select = U_fun.pearson_threshold(popu_X, popu_Y, N, m, n, p, 500)
    Kendall_select = U_fun.Kendall_threshold(popu_X, popu_Y, N, m, n, p, 500)
    SIRS_select = U_fun.SIRS_threshold(popu_X, popu_Y, N, m, n, p, 500)
    DC_select = U_fun.DC_threshold(popu_X, popu_Y, N, m, n, p, 500)
    print('U-method screening time:', time() - start_time)

    # compute results
    beta_index = np.where(beta_value != 0)[0]
    QR_index[Ti, 0] = int(np.all(np.in1d(beta_index, QR_select['index'])))
    QR_index[Ti, 1] = int(np.array_equal(QR_select['index'], beta_index))
    QR_index[Ti, 2] = len(QR_select['index'])
    QR_index[Ti, 3] = len(np.intersect1d(QR_select['index'], beta_index)) / len(beta_index)
    QR_index[Ti, 4] = len(np.setdiff1d(QR_select['index'], beta_index)) / len(QR_select['index'])
    QR_index[Ti, 5] = np.linalg.norm(beta_value - QR_select['beta_value'])

    pearson_index[Ti, 0] = int(np.all(np.in1d(beta_index, pearson_select['index'])))
    pearson_index[Ti, 1] = int(np.array_equal(pearson_select['index'], beta_index))
    pearson_index[Ti, 2] = len(pearson_select['index'])
    pearson_index[Ti, 3] = len(np.intersect1d(pearson_select['index'], beta_index)) / len(beta_index)
    pearson_index[Ti, 4] = len(np.setdiff1d(pearson_select['index'], beta_index)) / len(pearson_select['index'])
    pearson_index[Ti, 5] = np.linalg.norm(beta_value - pearson_select['beta_value'])

    Kendall_index[Ti, 0] = int(np.all(np.in1d(beta_index, Kendall_select['index'])))
    Kendall_index[Ti, 1] = int(np.array_equal(Kendall_select['index'], beta_index))
    Kendall_index[Ti, 2] = len(Kendall_select['index'])
    Kendall_index[Ti, 3] = len(np.intersect1d(Kendall_select['index'], beta_index)) / len(beta_index)
    Kendall_index[Ti, 4] = len(np.setdiff1d(Kendall_select['index'], beta_index)) / len(Kendall_select['index'])
    Kendall_index[Ti, 5] = np.linalg.norm(beta_value - Kendall_select['beta_value'])

    SIRS_index[Ti, 0] = int(np.all(np.in1d(beta_index, SIRS_select['index'])))
    SIRS_index[Ti, 1] = int(np.array_equal(SIRS_select['index'], beta_index))
    SIRS_index[Ti, 2] = len(SIRS_select['index'])
    SIRS_index[Ti, 3] = len(np.intersect1d(SIRS_select['index'], beta_index)) / len(beta_index)
    SIRS_index[Ti, 4] = len(np.setdiff1d(SIRS_select['index'], beta_index)) / len(SIRS_select['index'])
    SIRS_index[Ti, 5] = np.linalg.norm(beta_value - SIRS_select['beta_value'])

    DC_index[Ti, 0] = int(np.all(np.in1d(beta_index, DC_select['index'])))
    DC_index[Ti, 1] = int(np.array_equal(DC_select['index'], beta_index))
    DC_index[Ti, 2] = len(DC_select['index'])
    DC_index[Ti, 3] = len(np.intersect1d(DC_select['index'], beta_index)) / len(beta_index)
    DC_index[Ti, 4] = len(np.setdiff1d(DC_select['index'], beta_index)) / len(DC_select['index'])
    DC_index[Ti, 5] = np.linalg.norm(beta_value - DC_select['beta_value'])

    if (Ti + 1) % 10 == 0: print(Ti + 1, 'repetitions')


# save results
qr_df = pd.DataFrame( np.mean(QR_index, axis=0).reshape(1,-1), columns=['sc', 'cf', 'ams', 'psr', 'fdr', 'aee'])
pearson_df = pd.DataFrame( np.mean(pearson_index, axis=0).reshape(1,-1), columns=['sc', 'cf', 'ams', 'psr', 'fdr', 'aee'])
kendall_df = pd.DataFrame( np.mean(Kendall_index, axis=0).reshape(1,-1), columns=['sc', 'cf', 'ams', 'psr', 'fdr', 'aee'])
sirs_df = pd.DataFrame( np.mean(SIRS_index, axis=0).reshape(1,-1), columns=['sc', 'cf', 'ams', 'psr', 'fdr', 'aee'])
dc_df = pd.DataFrame( np.mean(DC_index, axis=0).reshape(1,-1), columns=['sc', 'cf', 'ams', 'psr', 'fdr', 'aee'])

# print('qr_indx: \n', qr_df)
# print('pearson_indx: \n', pearson_df)
# print('kendall_indx: \n', kendall_df)
# print('sirs_indx: \n', sirs_df)
# print('dc_indx: \n', dc_df)



with pd.ExcelWriter('hete_results_m=10.xlsx') as writer:
    qr_df.to_excel(writer, sheet_name='QR_index', index=False)
    pearson_df.to_excel(writer, sheet_name='pearson_index', index=False)
    kendall_df.to_excel(writer, sheet_name='Kendall_index', index=False)
    sirs_df.to_excel(writer, sheet_name='SIRS_index', index=False)
    dc_df.to_excel(writer, sheet_name='DC_index', index=False)



import matplotlib.pyplot as plt
import datetime

from POD_Lib import Calculation as calc
from POD_Lib import utils
from POD_Lib import models

def trainer():
    print('------------------------------------------------------------------------------------------------------------')
    now = datetime.datetime.now()
    time_now = now.strftime('%Y-%m-%d-%H:%M:%S')
    print(f'TRAINING START AT {time_now}')
    sol_mat = calc.sol_matrix(names='CD')
    print(f'Size of solution matrix is {sol_mat.shape}')
    sol_std, params_std= utils.arr_norm(sol_mat)
    mach_vf_array = utils.get_mach_vf_array()
    X_std, x_params_std = utils.arr_norm(mach_vf_array)
    U,s,V = calc.perform_SVD(matrix= sol_std)
    print(f'Size of U matrix from SVD is {U.shape}')
    k = calc.calc_K(s)
    U_hat = calc.calc_U_hat(U,k)
    print(f'Size of U hat matrix from SVD is {U_hat.shape}')
    delta_hat = calc.calc_delta_hat(U_hat,sol_mat= sol_std)
    print(f'Size of delta hat matrix is {delta_hat.shape}')
    history, eval = models.training(mach_vf_array, delta_hat,k)
    print('------------------------------------------------------------------------------------------------------------')
    end = datetime.datetime.now()
    time_end = end.strftime('%Y-%m-%d-%H:%M:%S')
    print(f'TRAINING DONE AT {time_end}')
    delta_time = end-now
    print(f'TIME NEEDED TO TRAIN IS {delta_time}')




if __name__=='__main__':
    trainer()
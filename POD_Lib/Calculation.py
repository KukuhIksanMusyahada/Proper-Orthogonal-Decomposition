import os
import numpy as np
import scipy as sc
import pandas as pd

from scipy import linalg

from POD_Lib import path_handling as ph
from POD_Lib.utils import get_mach_vf_array

def sol_matrix(names='CD', path=ph.get_raw_data()):
    sol_mat = list()
    for dir in os.listdir(path):
        dir_path = os.path.join(path, dir)
        for file in os.listdir(dir_path):
            if file.endswith('.csv'):
                file_path = os.path.join(dir_path, file)
                df = pd.read_csv(os.path.join(file_path), usecols=[names], nrows= 130).to_numpy()
                sol_mat.append(df)
    sol_mat = np.concatenate(sol_mat, axis=1)
    return sol_mat

def perform_SVD(matrix= sol_matrix()):
    U,s, V = linalg.svd(matrix, full_matrices= True)
    return U,s,V

def sigma_operation(array:list):
    value = 0
    for val in array:
        val *= val
        value +=val
    return value


def calc_K(sigma, tolerance= 0.005):
    E = 1-(tolerance**2)
    k = 0
    num = 0
    denum = sigma_operation(sigma)
    size_sigma= len(sigma)
    
    for value in sigma:
        k+=1
        value *= value
        num += value
        val = num/denum
        if val>= E:
            break
    return k


def calc_U_hat(matrix, k):
    return matrix[:,:k]


def calc_delta_hat(matrix, sol_mat= sol_matrix()):
    return np.matmul(matrix.T, sol_mat)


def prediction(delta_star, u_hat):
    return np.matmul(u_hat, delta_star)


    

    



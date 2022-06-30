import os
import numpy as np
import scipy as sc
import pandas as pd

from POD_Lib import path_handling as ph

def sol_matrix(names, path=ph.get_raw_data()):
    sol_mat = list()
    for dir in os.listdir(path):
        dir_path = os.path.join(path, dir)
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            df = pd.read_csv(os.path.join(file_path), usecols=[names], nrows= 130).to_numpy()
            sol_mat.append(df)
    sol_mat = np.concatenate(sol_mat, axis=1).T
    return sol_mat

def perform_SVD():
    pass




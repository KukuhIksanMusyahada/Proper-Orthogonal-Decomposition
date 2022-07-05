from ast import List
import re
from statistics import mean
import numpy as np
import os

from POD_Lib import path_handling as ph


def extract_mach_and_vf(file: str):
    pattern = r'M_([0-9\.]*)_VF_([0-9\.]*).csv'
    result  = re.match(pattern, file)

    return float(result.group(1)), float(result.group(2))

def get_mach_vf_array(path=ph.get_raw_data()):
    mach = []
    vf = []
    for dir in os.listdir(path):
        dir_path = os.path.join(path, dir)
        for file in os.listdir(dir_path):
            if file.endswith('.csv'):
                result = extract_mach_and_vf(file)
                mach.append(result[0])
                vf.append(result[1])
    list = [mach, vf]
    return np.array(list).T



def norm(data, min, max, mean, std, minmax= False):
    if minmax == True:
        return (data-min)/(max-min)
    else:
        return (data-mean)/std

def denorm(data, min, max, mean, std, minmax= False):
    if minmax == True:
        return data*(max-min) + min
    else:
        return (data * std) + mean


def arr_norm(data, minmax= False, params= None):
    if params==None:
        params={'min':[] , 'max': [], 'mean':[], 'std':[]}
        for col in range(data.shape[1]):
            params['min'].append(np.min(data[:,col]))
            params['max'].append(np.max(data[:,col]))
            params['mean'].append(np.mean(data[:,col]))
            params['std'].append(np.std(data[:,col]))

    for col in range(data.shape[1]):
        if minmax==True:
            data[:,col]= norm(data[:,col],min=params['min'][col],
                max=params['max'][col], mean=params['mean'][col],
                std=params['std'][col], minmax=True)
        else:
            data[:,col]= norm(data[:,col],min=params['min'][col],
                max=params['max'][col], mean=params['mean'][col],
                std=params['std'][col])
    return data, params


def arr_denorm(data,params, minmax= False):
    for col in range(data.shape[1]):
        if minmax==True:
            data[:,col]= denorm(data[:,col], min=params['min'][col],
                 max=params['max'][col], mean=params['mean'][col],
                  std=params['std'][col], minmax=True)
        else:
            data[:,col]= denorm(data[:,col], min=params['min'][col],
                 max=params['max'][col], mean=params['mean'][col],
                  std=params['std'][col])

    return data

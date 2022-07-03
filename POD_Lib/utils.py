from ast import List
import re
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
import os


def get_this_dir():
    return os.path.dirname( os.path.abspath(__file__) )

def get_raw_data():
    return os.path.join(get_this_dir(), os.pardir, 'Raw_New')

def get_M0_6_data():
    return os.path.join(get_raw_data(), 'M_0.6')

def get_M0_7_data():
    return os.path.join(get_raw_data(), 'M_0.7')

def get_M0_8_data():
    return os.path.join(get_raw_data(), 'M_0.8')

def get_M0_9_data():
    return os.path.join(get_raw_data(), 'M_0.9')

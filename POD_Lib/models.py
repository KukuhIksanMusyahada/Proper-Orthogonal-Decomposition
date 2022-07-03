import os
import numpy as np
import tensorflow as tf
import datetime
import pickle


from keras.models import Sequential
from keras.layers import Dense

from POD_Lib.utils import get_mach_vf_array
from POD_Lib import path_handling as ph

def layers(num_layer:int, num_neurons:int, min_neurons=0,  step_neurons=None, num_features=1):
    model= Sequential()
    model.add(Dense(200, input_dim=2))
    if step_neurons != None:
        min_neurons = num_neurons-(step_neurons*num_layer)
        if min_neurons<0:
            min_neurons=0
        num_neurons = range(num_neurons, min_neurons, -step_neurons)
    for neurons in num_neurons:
        model.add(Dense(neurons))
    model.add(Dense(num_features))

    return model

def loss_optim(model, optimizer='adam', loss='mse'):
    model.compile(optimizer=optimizer, loss=loss)

    return model

def fit_model(model,machVF, Udelta, max_epochs = 500):
    X = np.array(machVF.T)
    y = np.array(Udelta.T)
    history = model.fit(X,y, epochs= max_epochs, verbose=0)

    return history

def eval_model(model, machvf, Udelta):
    X = np.array(machvf.T)
    y = np.array(Udelta.T)
    eval = model.evaluate(model, X,y)
    return eval

def training(machVF,Udelta, k):
    model = layers(num_layer=0, num_neurons=200, step_neurons=100, num_features= k)
    model = loss_optim(model)
    X = np.array(machVF)
    y = np.array(Udelta)
    model.summary()
    history = fit_model(model, X, y)
    eval = model.evaluate(X.T,y.T)
    SaveModel(model, history)
    return history, eval

def SaveModel(model, history, optional_path: str=None):
    """Save both model and history"""
    now = datetime.datetime.now()
    time_now = now.strftime('%Y%m%d%H%M%S')
    folder_name = "ANN " + time_now
    if optional_path != None:
        model_directory = os.path.join (optional_path, folder_name)
    else:
        model_directory = os.path.join (ph.get_models_data(), folder_name)
    os.makedirs(model_directory)
    history_file = os.path.join(model_directory, 'history.pkl')


    model.save(model_directory)
    print ("\nModel saved to {}".format(model_directory))

    with open(history_file, 'wb') as f:
        pickle.dump(history.history, f)
    print ("Model history saved to {}".format(history_file))


def LoadModel(path_to_model):
    """Load Model and optionally it's history as well"""
    history_file = os.path.join(path_to_model, 'history.pkl')
    model = tf.keras.models.load_model(path_to_model)
    # model = tf.saved_model.load(path_to_model)
    print ("\nmodel loaded")

    with open(history_file, 'rb') as f:
        history = pickle.load(f)
    print ("model history loaded")

    return model, history
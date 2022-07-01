import numpy as np
import tensorflow as tf


from keras.models import Sequential
from keras.layers import Dense

from POD_Lib.utils import get_mach_vf_array

def layers(num_layer:int, num_neurons:int, min_neurons=0,  step_neurons=None, num_features=1):
    model= Sequential()
    model.add(Dense(20, input_dim=2))
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

def training(machVF,Udelta):
    model = layers(num_layer=0, num_neurons=200, step_neurons=100)
    model = loss_optim(model)
    X = np.array(machVF)
    y = np.array(Udelta)
    model.summary()
    history = fit_model(model, X, y)
    eval = model.evaluate(X.T,y.T)
    
    return history, eval

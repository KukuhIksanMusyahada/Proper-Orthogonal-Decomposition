from pyexpat import model
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense

def layers(num_layer:int, num_neurons:int, min_neurons=0,  step_neurons=None, num_features=1):
    model= Sequential()
    if step != None:
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

def training(model,machVF, Udelta):

    return model

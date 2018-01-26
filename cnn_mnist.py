"""
Created on Mon Jan 22 2018
Python 2.7

Convolutional Neural Network
Frameworks: Theano and Lasagne
Data base: mnist

@author: Lucas Buzuti
"""
from lasagne.layers import InputLayer, DenseLayer, Conv2DLayer, MaxPool2DLayer
from lasagne.nonlinearities import softmax
from Th_nN.train import plot_train, compile_train_function, fit
from Th_nN.check_train import saved_params, predict
from Th_nN.mnist_data import loading_dataset
from sklearn.model_selection import train_test_split

x_train, y_train, x_test, y_test = loading_dataset()
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.3, random_state=15000)


# Model Neural Network
def build_neural_network():
    net = {}

    net['input'] = InputLayer((None, 1, 28, 28))
    net['conv1'] = Conv2DLayer(net['input'], num_filters=8, filter_size=5)
    net['pool1'] = MaxPool2DLayer(net['conv1'], pool_size=2)
    net['conv2'] = Conv2DLayer(net['pool1'], num_filters=16, filter_size=5)
    net['pool2'] = MaxPool2DLayer(net['conv2'], pool_size=3)
    net['hid1'] = DenseLayer(net['pool2'], num_units=100)
    net['hid2'] = DenseLayer(net['hid1'], num_units=100)
    net['hid3'] = DenseLayer(net['hid2'], num_units=100)
    net['out'] = DenseLayer(net['hid3'], num_units=10, nonlinearity=softmax)
    return net

net = build_neural_network()
train_fn, valid_fn = compile_train_function(net, lr=0.0001, w_dacy=1e-5)
train_curves = fit(train_fn, valid_fn,
                   train_set=(x_train, y_train), valid_set=(x_valid, y_valid),
                   epochs=20, batch_size=8000)
saved_params(net, 'params.pkl')
plot_train(train_curves)

predict(net, valid_fn, x_test, y_test, 'params.pkl')


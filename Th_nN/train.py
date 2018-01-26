#coding: utf-8
from matplotlib import pyplot as plt
import numpy as np
import sys
import lasagne
import theano
from theano import tensor as T
from lasagne.regularization import regularize_layer_params, l2

def plot_train(train_curves):
    plt.figure()
    cost_history, acc_history, val_cost_history, val_acc_history = train_curves
    plt.plot(cost_history, 'b--', label='Tranining')
    plt.plot(val_cost_history, 'r-', label='Valid')
    plt.xlabel('Epochs', fontsize=15)
    plt.ylabel('Error rate', fontsize=15)
    plt.legend()
    print 'The best performance valid: %0.2f%%' % (np.max(val_acc_history)*100)
    plt.show()

def compile_train_function(neural_network, lr, w_dacy):
    input_var = neural_network['input'].input_var
    output_var = T.lvector()  # Variable symbolic

    predicted = lasagne.layers.get_output(neural_network['out'], inputs=input_var)  # Answer of output

    loss = lasagne.objectives.categorical_crossentropy(predicted, output_var)  # Function of error
    loss = loss.mean()
    """
    Regularize L2 (avoid over-fitting)
    Only to function of train
    
    Lreg = L + λ*∑(w^2)
    where:  L --> loss
            λ --> weight decay
            w --> weight
    """
    loss += w_dacy * regularize_layer_params(neural_network['out'], l2)  # Regularize L2

    # Accuracy rate
    y_pred = T.argmax(predicted, axis=1)
    acc = T.eq(y_pred, output_var)
    acc = acc.mean()

    valid_predicted = lasagne.layers.get_output(neural_network['out'], inputs=input_var)  # Validation answer of output
    valid_loss = lasagne.objectives.categorical_crossentropy(valid_predicted, output_var)  # Validation function of error
    valid_loss = valid_loss.mean()

    # Validation accuracy rate
    valid_y_pred = T.argmax(valid_predicted, axis=1)
    valid_acc = T.eq(valid_y_pred, output_var)
    valid_acc = valid_acc.mean()

    # Parameters updating
    params = lasagne.layers.get_all_params(neural_network['out'])
    updates = lasagne.updates.sgd(loss, params, lr)

    # Compile function
    train_fn = theano.function([input_var, output_var], [loss, acc], updates=updates)
    valid_fn = theano.function([input_var, output_var], [valid_loss, valid_acc])
    return train_fn, valid_fn

def _iterate_minibatches(x, y, batch_size):
    for batch_start in xrange(0, len(x), batch_size):
        yield x[batch_start:batch_start+batch_size], y[batch_start:batch_start+batch_size]

def fit(train_fn, valid_fn, train_set, valid_set, epochs, batch_size):
    x_train, y_train = train_set
    x_valid, y_valid = valid_set

    cost_history = []
    acc_history = []
    val_cost_history = []
    val_acc_history = []

    print('epoch\ttrain_err\tval_err')
    for i in range(epochs):
        epoch_cost = 0
        epoch_acc = 0
        train_batches = 0
        for x_batch, y_batch in _iterate_minibatches(x_train, y_train, batch_size):
            cost, acc = train_fn(x_batch, y_batch)
            epoch_cost += cost
            epoch_acc += acc
            train_batches += 1

        val_epoch_cost = 0
        val_epoch_acc = 0
        val_batches = 0
        for x_batch, y_batch in _iterate_minibatches(x_valid, y_valid, batch_size):
            val_cost, val_acc = valid_fn(x_batch, y_batch)
            val_epoch_cost += val_cost
            val_epoch_acc += val_acc
            val_batches += 1

        epoch_cost = epoch_cost / train_batches
        cost_history.append(epoch_cost)
        acc_history.append(epoch_acc / train_batches)

        val_epoch_cost = val_epoch_cost / val_batches
        val_cost_history.append(val_epoch_cost)
        val_acc_history.append(val_epoch_acc / val_batches)
        num_epochs = int(((i+1.)/epochs) * 100)
         
        sys.stdout.write('\033[2K' + '\r' + '%d\t%.4f\t\t%.4f' % (i + 1, epoch_cost, val_epoch_cost))
        sys.stdout.write('\n')
        sys.stdout.write('Epochs ' + str(i + 1) + '/' + str(epochs) +
                         ' | Progress ' + '#' * num_epochs + ' ' + str(num_epochs) + '%')
        sys.stdout.flush()

    print '\n\nValidation accuracy rate: %.2f%%' % (val_acc_history[-1] * 100)
    return cost_history, acc_history, val_cost_history, val_acc_history
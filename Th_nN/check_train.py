import lasagne
import cPickle
import os

def saved_params(neural_network, file):
    path = os.path.abspath('.')
    files = path[:]
    params = lasagne.layers.get_all_param_values(neural_network['out'])
    cPickle.dump(params, open(files+'/'+file, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
    print 'Saved file'

def predict(neural_network, valid_fn, x_test, y_test, file):
    path = os.path.abspath('.')
    files = path[:]
    loaded_params = cPickle.load(open(files+'/'+file, 'rb'))
    lasagne.layers.set_all_param_values(neural_network['out'], loaded_params)
    acc = valid_fn(x_test, y_test)[1]
    print 'Test accuracy rate: %.2f%%' % (acc*100)



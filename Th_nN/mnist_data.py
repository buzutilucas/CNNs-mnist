import numpy as np
import struct
import urllib
import os
import sys
import gzip
import StringIO

def _open_file(file_img, file_lable):
    with open(file_lable, 'rb') as flbl:
        magic, num = struct.unpack('>II', flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(file_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), 1, rows, cols)
    return img, lbl

def loading_dataset():
    baseURL = 'http://yann.lecun.com/exdb/mnist/'
    lst_data_url = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
                    't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
    lst_dataset = ['train-images.idx3-ubyte', 'train-labels.idx1-ubyte',
                   't10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte']

    path = os.path.abspath('.')

    # Download dataset if not yet done:
    for i, data in enumerate(lst_data_url):
        if not os.path.isfile(path+'/Datasets/'+lst_dataset[i]):
            sys.stdout.write('\r' + 'Downloading')
            resp = urllib.urlopen(baseURL + data)
            compressed_file = StringIO.StringIO()
            compressed_file.write(resp.read())
            compressed_file.seek(0)
            decompressed_file = gzip.GzipFile(fileobj=compressed_file, mode='rb')
            with open(path+'/Datasets/'+lst_dataset[i], 'wb') as out_file:
                out_file.write(decompressed_file.read())
    sys.stdout.write('\r')

    # Load the dataset
    train_file_img = path+'/Datasets/'+lst_dataset[0]
    train_file_lbl = path+'/Datasets/'+lst_dataset[1]
    test_file_img = path+'/Datasets/'+lst_dataset[2]
    test_file_lbl = path+'/Datasets/'+lst_dataset[3]
    train_img, train_lbl = _open_file(train_file_img, train_file_lbl)
    test_img, test_lbl = _open_file(test_file_img, test_file_lbl)
    return train_img, train_lbl, test_img, test_lbl
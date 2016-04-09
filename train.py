#!/usr/bin/env python
import os
import sys
import logging
import numpy as np
import cv2
import h5py
import time
from fuel.datasets.hdf5 import H5PYDataset
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream
import theano
from theano import tensor as T
import lasagne
from tqdm import tqdm

# for organizing an hdf5 file for streaming
class DataH5PyStreamer:
    # folds = None if dataset is separated into 'train', 'test'
    # folds = (10, 3) for ex if there are 10 total folds and #3 (zero_indexed) is validation set
    # folds = (10, 10) use all folds as train and no test
    def __init__(self, path, batch=1, fold=0):
        # probe path for folds
        with h5py.File(path, mode='r') as f:
            fold_names = set([x[0] for x in f.attrs['split'] if 'fold-' in x[0]])
            folds = len(fold_names)
            shape = f['images'].shape
            label_shape = f['labels'].shape
            logging.info("folds found in {} is {}".format(path, fold_names))
            logging.info("image shape of {} is {}".format(path, shape))
            logging.info("label shape of {} is {}".format(path, label_shape))
            self.shape = (batch,) + shape[1:]
        #sys.exit(0)
        tr_set = ['fold-{}'.format(i) for i in range(folds) if i != fold]
        assert set(tr_set).issubset(fold_names)
        tr_set = set(tr_set)
        te_set = ['fold-{}'.format(i)]
        assert set(te_set).issubset(fold_names)
        te_set = set(te_set)

        self.batch = batch
        self.tr_data = H5PYDataset(path, which_sets=tr_set)
        self.te_data = H5PYDataset(path, which_sets=te_set)
        self.tr_size = self.tr_data.num_examples
        self.te_size = self.te_data.num_examples

        self.tr_shuf = ShuffledScheme(examples=self.tr_size, batch_size=batch)
        self.te_shuf = SequentialScheme(examples=self.te_size, batch_size=batch)
        pass

    def train (self):
        return DataStream(self.tr_data, iteration_scheme = self.tr_shuf)

    def test (self):
        return DataStream(self.te_data, iteration_scheme = self.te_shuf)
    pass

def save_params(model, fn):
    if isinstance(model, list):
        param_vals = model
    else:
        param_vals = lasagne.layers.get_all_param_values(model)
    if 'npz' in fn:
        np.savez(fn, *param_vals)
    else:
        with open(fn, 'w') as wr:
            import pickle
            pickle.dump(param_vals, wr)

def run_epoch (stream, func, maxit, shape):
    err = None
    n = 0
    #print maxit
    for imb in tqdm(stream.get_epoch_iterator(), total=maxit):
        if imb[0].shape != shape:
            continue
        #imb = tr_transform(imb)
        if not isinstance(imb, tuple):
            imb = (imb,)
        e = np.array(func(*imb))
        if err is None:
            err = e
        else:
            err += e
        n += 1
        pass
    return err / n

def train (model, data, out_path, max_epoch, fold, batch):
    verbose = True
    streamer = DataH5PyStreamer(data, batch=batch, fold=fold)
    shape = streamer.shape
    logging.info('data shape is {}'.format(shape))

    import pkgutil
    loader = pkgutil.get_importer('models')
    # load network from file in 'models' dir
    model = loader.find_module(model).load_module(model)

    input_var = T.tensor4('input')
    label_var = T.tensor4('label')

    net, loss, scores = model.network(input_var, label_var, shape)

    params = lasagne.layers.get_all_params(net, trainable=True)
    lr = theano.shared(lasagne.utils.floatX(3e-3))

    updates = lasagne.updates.adam(loss, params, learning_rate=lr)

    train_fn = theano.function([input_var, label_var], loss, updates=updates)
    test_fn = theano.function([input_var, label_var], scores)

    best = None # (score, epoch, params)
    tr_stream = streamer.train()
    te_stream = streamer.test()
    for epoch in range(max_epoch):
        start = time.time()
        tr_err = run_epoch(tr_stream, train_fn, streamer.tr_size/batch, shape)
        te_err = run_epoch(te_stream, test_fn, streamer.te_size/batch, shape)
        s = te_err[0]
        if best is None or s < best[0]:
            best = (s, epoch, [np.copy(p) for p in (lasagne.layers.get_all_param_values(net))])
            pass
        if verbose:
            print('ep {}/{} - tl {} - vl {} - t {:.3f}s'.format(
                epoch, max_epoch, tr_err, te_err, time.time()-start))
        pass

    print "save best epoch: {:d}".format(best[1])
    save_params(best[2], out_path)
    pass

if __name__ == '__main__':
    import argparse
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser(description='import training images & labels')
    #parser.add_argument('--root', default='hdfs://washtenaw:19000/user/hive/warehouse/wdong_tri.db')
    parser.add_argument('model', nargs=1)
    parser.add_argument('data', nargs=1)
    parser.add_argument('params', nargs=1)
    parser.add_argument('--epoch', default=300, type=int)
    parser.add_argument('--fold', default=0, type=int)
    parser.add_argument('--batch', default=16, type=int)
    args = parser.parse_args()
    np.random.seed(1234)
    train(args.model[0], args.data[0], args.params[0], args.epoch, args.fold, args.batch)
    pass


#!/usr/bin/env python
import os
import sys
import logging
import numpy as np
import cv2
import time
import picpac
import theano
from theano import tensor as T
import lasagne
from tqdm import tqdm

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
    for it in tqdm(range(maxit)): #stream.get_epoch_iterator(), total=maxit):
        image, anno, pad = stream.next()
        e = np.array(func(image, anno))
        if err is None:
            err = e
        else:
            err += e
        n += 1
        pass
    return err / n

def train (model, data, out_path, max_epoch, K, fold, batch):
    verbose = True
    seed = 1996
    tr_stream = picpac.ImageStream(data, batch=batch, K=K, fold=fold, train=True, annotate='image', seed=seed, reshuffle=True)
    shape = tr_stream.next()[0].shape
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
    for epoch in range(max_epoch):
        start = time.time()
        tr_err = run_epoch(tr_stream, train_fn, tr_stream.size() / batch, shape)
        te_stream = picpac.ImageStream(data, batch=batch, K=K, fold=fold, train=False, annotate='image', seed=seed)
        te_err = run_epoch(te_stream, test_fn, te_stream.size() / batch, shape)
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
    parser.add_argument('-K', default=5, type=int)
    parser.add_argument('--batch', default=16, type=int)
    args = parser.parse_args()
    np.random.seed(1234)
    train(args.model[0], args.data[0], args.params[0], args.epoch, args.K, args.fold, args.batch)
    pass

